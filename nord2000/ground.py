"""Ground impedance and reflection coefficients.

Time convention throughout the package is exp(-i omega t); a spherical
wave is exp(i k r) / r.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import wofz

#: Nord2000 impedance classes: effective flow resistivity [kPa s m^-2].
GROUND_CLASSES = {
    "A": 12.5,      # very soft (snow, moss)
    "B": 31.5,      # soft forest floor
    "C": 80.0,      # uncompacted, loose ground (turf, grass)
    "D": 200.0,     # normal uncompacted ground (pasture, forest floor)
    "E": 500.0,     # compacted field and gravel
    "F": 2000.0,    # compacted dense ground (gravel road, parking lot)
    "G": 20000.0,   # hard surfaces (dense asphalt, concrete)
    "H": 200000.0,  # very hard and dense surfaces (water)
}

RIGID = math.inf


def flow_resistivity(ground) -> float:
    """Return the flow resistivity [kPa s m^-2] for a class letter or number."""
    if ground is None:
        return RIGID
    if isinstance(ground, str):
        try:
            return GROUND_CLASSES[ground.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown Nord2000 ground class {ground!r}") from exc
    return float(ground)


def impedance(frequency: float, sigma: float) -> complex:
    """Normalised surface impedance (Delany-Bazley, sigma in kPa s m^-2)."""
    if math.isinf(sigma):
        return complex(math.inf)
    x = frequency / sigma
    return complex(1.0 + 9.08 * x**-0.75, 11.9 * x**-0.73)


def plane_wave_reflection(sin_psi: float, Z: complex) -> complex:
    """Plane-wave reflection coefficient for grazing angle psi."""
    if np.isinf(Z):
        return 1.0 + 0j
    beta = 1.0 / Z
    return (sin_psi - beta) / (sin_psi + beta)


def spherical_reflection(
    frequency: float, sigma: float, r2: float, sin_psi: float, c: float = 340.0
) -> complex:
    """Spherical-wave reflection coefficient Q = Rp + (1 - Rp) F(w).

    ``r2`` is the length of the reflected path (image source to receiver)
    and ``sin_psi`` the sine of the grazing angle.
    """
    if math.isinf(sigma):
        return 1.0 + 0j
    sin_psi = min(max(sin_psi, 0.0), 1.0)
    Z = impedance(frequency, sigma)
    beta = 1.0 / Z
    rp = (sin_psi - beta) / (sin_psi + beta)
    k = 2.0 * math.pi * frequency / c
    w = np.sqrt(0.5j * k * r2) * (sin_psi + beta)
    F = 1.0 + 1j * math.sqrt(math.pi) * w * wofz(w)
    return complex(rp + (1.0 - rp) * F)
