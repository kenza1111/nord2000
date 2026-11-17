"""One-third octave frequency bands and frequency weightings used by Nord2000."""

from __future__ import annotations

import numpy as np

#: Nominal 1/3-octave centre frequencies used by Nord2000 (25 Hz - 10 kHz).
NOMINAL_FREQUENCIES = np.array(
    [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
    ],
    dtype=float,
)

#: Exact (base-10) 1/3-octave centre frequencies, f = 1000 * 10^(n/10).
EXACT_FREQUENCIES = 1000.0 * 10.0 ** (np.arange(-16, 11) / 10.0)

#: Relative bandwidth of a 1/3-octave band (f2 - f1) / fc.
THIRD_OCTAVE_RELATIVE_BANDWIDTH = 10 ** 0.05 - 10 ** -0.05  # ~0.2316


def a_weighting(frequencies) -> np.ndarray:
    """A-weighting in dB (IEC 61672-1) for the given frequencies."""
    f2 = np.asarray(frequencies, dtype=float) ** 2
    ra = (12194.0**2 * f2**2) / (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12194.0**2)
    )
    return 20.0 * np.log10(ra) + 2.0


def energetic_sum(levels, axis=None) -> float | np.ndarray:
    """Energetic sum 10 lg(sum 10^(L/10)) of levels in dB."""
    levels = np.asarray(levels, dtype=float)
    return 10.0 * np.log10(np.sum(10.0 ** (levels / 10.0), axis=axis))
