"""Python implementation of the Nord2000 outdoor sound propagation model."""

from .atmosphere import Weather, air_absorption, sound_speed
from .bands import EXACT_FREQUENCIES, NOMINAL_FREQUENCIES, a_weighting, energetic_sum
from .ground import GROUND_CLASSES, impedance, spherical_reflection
from .propagation import Nord2000, PropagationResult
from .terrain import TerrainProfile

__all__ = [
    "Nord2000",
    "PropagationResult",
    "TerrainProfile",
    "Weather",
    "GROUND_CLASSES",
    "NOMINAL_FREQUENCIES",
    "EXACT_FREQUENCIES",
    "a_weighting",
    "air_absorption",
    "energetic_sum",
    "impedance",
    "sound_speed",
    "spherical_reflection",
]

__version__ = "0.1.0"
