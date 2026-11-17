"""Coherence reduction and partially coherent summation of ray contributions.

Nord2000 does not add the ray contributions fully coherently.  The
cross terms are multiplied by a coherence coefficient F = F_f * F_c,
where F_f accounts for the averaging over a 1/3-octave band and F_c for
the loss of coherence caused by atmospheric turbulence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .bands import THIRD_OCTAVE_RELATIVE_BANDWIDTH
from .geometry import polyline_mean_separation


@dataclass
class Contribution:
    """One ray (or diffracted ray) contribution at the receiver.

    ``amplitude`` is the complex pressure relative to the free-field
    pressure over the direct distance, ``length`` the path length and
    ``path`` the polyline followed by the ray (for turbulence decorrelation).
    """

    amplitude: complex
    length: float
    path: list


def frequency_coherence(f: float, delta_tau: float) -> float:
    """Coherence due to averaging over a 1/3-octave band (sinc, clipped at 0)."""
    x = math.pi * THIRD_OCTAVE_RELATIVE_BANDWIDTH * f * abs(delta_tau)
    if x < 1e-9:
        return 1.0
    if x >= math.pi:
        return 0.0
    return math.sin(x) / x


def turbulence_coherence(f: float, rho: float, distance: float, c0: float,
                         T: float, Cv2: float, CT2: float) -> float:
    """Coherence between two rays separated by ``rho`` (Kolmogorov turbulence)."""
    if rho <= 0.0:
        return 1.0
    k = 2.0 * math.pi * f / c0
    gamma = 0.375 * 0.364 * (CT2 / T**2 + 22.0 / 3.0 * Cv2 / c0**2)
    return math.exp(-gamma * k**2 * rho ** (5.0 / 3.0) * distance)


def partially_coherent_level(contribs: list[Contribution], f: float, c0: float,
                             T: float, Cv2: float, CT2: float,
                             x0: float, x1: float) -> float:
    """Level [dB] relative to free field of partially coherent contributions."""
    if not contribs:
        return -200.0
    d = abs(x1 - x0)
    energy = sum(abs(c.amplitude) ** 2 for c in contribs)
    n = len(contribs)
    for i in range(n):
        ci = contribs[i]
        if ci.amplitude == 0:
            continue
        for j in range(i + 1, n):
            cj = contribs[j]
            if cj.amplitude == 0:
                continue
            ff = frequency_coherence(f, (ci.length - cj.length) / c0)
            if ff == 0.0:
                continue
            rho = polyline_mean_separation(ci.path, cj.path, x0, x1)
            fc = turbulence_coherence(f, rho, d, c0, T, Cv2, CT2)
            energy += 2.0 * ff * fc * (ci.amplitude * np.conj(cj.amplitude)).real
    return 10.0 * math.log10(max(energy, 1e-20))
