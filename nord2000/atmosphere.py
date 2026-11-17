"""Atmospheric properties: air absorption, sound speed and the weather model.

Air absorption follows ISO 9613-1:1993.  The sound speed profile is the
Nord2000 log-lin profile

    c(z) = A ln(z / z0 + 1) + B z + c0

which is converted to an *equivalent linear* profile c(z) = c0 + a z
before the propagation calculation (Nord2000 is formulated with
circular-arc rays in a linear sound speed profile).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

T0_KELVIN = 273.15
REFERENCE_PRESSURE_KPA = 101.325


def sound_speed(temperature_c: float) -> float:
    """Adiabatic speed of sound in air [m/s] at ``temperature_c`` [degC]."""
    return 331.3 * math.sqrt(1.0 + temperature_c / T0_KELVIN)


def air_absorption(
    frequency,
    temperature_c: float = 15.0,
    relative_humidity: float = 70.0,
    pressure_kpa: float = REFERENCE_PRESSURE_KPA,
) -> np.ndarray:
    """Pure-tone atmospheric absorption coefficient [dB/m] (ISO 9613-1)."""
    f = np.asarray(frequency, dtype=float)
    T = temperature_c + T0_KELVIN
    T20 = 293.15
    T01 = 273.16
    pa_pr = pressure_kpa / REFERENCE_PRESSURE_KPA

    c_sat = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    psat_pr = 10.0**c_sat
    h = relative_humidity * psat_pr / pa_pr  # molar concentration of water vapour [%]

    fr_o = pa_pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pa_pr * (T / T20) ** -0.5 * (
        9.0 + 280.0 * h * math.exp(-4.170 * ((T / T20) ** (-1.0 / 3.0) - 1.0))
    )

    alpha = (
        8.686
        * f**2
        * (
            1.84e-11 / pa_pr * (T / T20) ** 0.5
            + (T / T20) ** -2.5
            * (
                0.01275 * math.exp(-2239.1 / T) / (fr_o + f**2 / fr_o)
                + 0.1068 * math.exp(-3352.0 / T) / (fr_n + f**2 / fr_n)
            )
        )
    )
    return alpha


@dataclass
class Weather:
    """Meteorological conditions for a Nord2000 calculation.

    Parameters
    ----------
    temperature : air temperature at ground level [degC]
    humidity : relative humidity [%]
    pressure : static pressure [kPa]
    A : logarithmic sound speed profile coefficient [m/s]
    B : linear sound speed profile coefficient [1/s]
    z0 : aerodynamic roughness length [m]
    Cv2 : turbulence structure parameter for wind [m^(4/3) s^-2]
    CT2 : turbulence structure parameter for temperature [K^2 m^(-2/3)]
    """

    temperature: float = 15.0
    humidity: float = 70.0
    pressure: float = REFERENCE_PRESSURE_KPA
    A: float = 0.0
    B: float = 0.0
    z0: float = 0.1
    Cv2: float = 0.12
    CT2: float = 0.008

    @classmethod
    def from_meteo(
        cls,
        wind_speed: float = 0.0,
        wind_height: float = 10.0,
        wind_direction: float = 0.0,
        temperature_gradient: float = 0.0,
        z0: float = 0.1,
        **kwargs,
    ) -> "Weather":
        """Build a weather object from simple meteorological data.

        ``wind_direction`` is the angle [deg] between the wind vector and
        the source-to-receiver direction (0 = downwind, 180 = upwind).
        ``temperature_gradient`` is dT/dz [K/m].
        """
        w = cls(z0=z0, **kwargs)
        u = wind_speed * math.cos(math.radians(wind_direction))
        w.A = u / math.log(wind_height / z0 + 1.0)
        w.B = w.c0 / (2.0 * (w.temperature + T0_KELVIN)) * temperature_gradient
        return w

    @property
    def c0(self) -> float:
        """Sound speed at ground level [m/s]."""
        return sound_speed(self.temperature)

    @property
    def temperature_kelvin(self) -> float:
        return self.temperature + T0_KELVIN

    def sound_speed_profile(self, z) -> np.ndarray:
        z = np.maximum(np.asarray(z, dtype=float), 0.0)
        return self.A * np.log(z / self.z0 + 1.0) + self.B * z + self.c0

    def absorption(self, frequency) -> np.ndarray:
        return air_absorption(frequency, self.temperature, self.humidity, self.pressure)

    def _linear_fit(self, zmax: float) -> float:
        z = np.linspace(0.0, max(zmax, 1e-3), 200)
        c = self.sound_speed_profile(z)
        zc = z - z.mean()
        return float(np.sum(zc * (c - c.mean())) / np.sum(zc * zc))

    def equivalent_gradient(self, hs: float, hr: float, distance: float) -> float:
        """Equivalent linear sound speed gradient a = dc/dz [1/s].

        The log-lin profile is replaced by the least-squares linear profile
        over the height range relevant for the propagation path, taken as
        0 ... max(hs, hr) + 0.1 * distance (the strong near-ground part of
        the log profile would otherwise dominate).  For downward refraction
        the ray apex height depends on the gradient itself, so the apex
        height of the circular ray is added and the fit is iterated.
        """
        if self.A == 0.0 and self.B == 0.0:
            return 0.0
        zbase = max(hs, hr, 1.0) + 0.1 * distance
        a = self._linear_fit(zbase)
        for _ in range(5):
            if a <= 0.0:
                break
            radius = self.c0 / a
            bulge = distance**2 / (8.0 * radius)
            a = self._linear_fit(zbase + bulge)
        return a
