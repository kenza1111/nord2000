# nord2000

A Python implementation of the **Nord2000** outdoor sound propagation model.
It calculates the propagation of sound from a point source to a receiver
along a vertical terrain profile, in 1/3-octave bands from 25 Hz to 10 kHz,
including ground effect, screening, terrain shape and weather.

```
L_R = L_W + ΔL_d + ΔL_a + ΔL_t
```

| term   | meaning                                                         |
|--------|-----------------------------------------------------------------|
| `ΔL_d` | geometrical divergence, −10 lg(4πR²)                            |
| `ΔL_a` | air absorption, ISO 9613-1                                      |
| `ΔL_t` | terrain effect relative to free field: ground, screens, hills, refraction |

## Installation

```bash
pip install -e .            # numpy + scipy
pip install -e .[test]      # + pytest
```

## Usage

```python
import numpy as np
from nord2000 import Nord2000, TerrainProfile, Weather

# 100 m of grass (ground class D) with a 3 m rigid screen at x = 50 m
profile = TerrainProfile.flat(0, 100, "D").with_screen(50, 3.0)

model = Nord2000(Weather.from_meteo(wind_speed=3.0, wind_direction=0))
res = model.propagate(profile, source_x=0, source_height=0.5,
                      receiver_x=100, receiver_height=1.5)

res.frequencies      # 1/3-octave centre frequencies
res.terrain          # ΔL_t per band [dB]
res.attenuation      # ΔL_d + ΔL_a + ΔL_t per band
res.receiver_levels(np.full(27, 100.0))     # L_R per band for a given L_W spectrum
res.receiver_level_a(np.full(27, 100.0))    # total A-weighted level
```

Profiles are piecewise-linear `(x, z)` polylines. Each segment has a ground
class, which can differ from segment to segment:

```python
TerrainProfile([(0, 0), (20, 0), (60, 4), (200, 4)], ["G", "D", "C"])
profile.with_screen(x, height)                    # thin rigid screen
profile.with_screen(x, height, thickness=10)      # thick barrier or building
```

Run `python examples/basic_usage.py` for a complete example.

### Ground classes (Nord2000)

| class | σ [kPa s m⁻²] | typical surface                  |
|-------|---------------|----------------------------------|
| A     | 12.5          | very soft (snow, moss)           |
| B     | 31.5          | soft forest floor                |
| C     | 80            | uncompacted loose ground, turf   |
| D     | 200           | normal uncompacted ground, grass |
| E     | 500           | compacted field, gravel          |
| F     | 2000          | compacted dense ground           |
| G     | 20000         | hard surfaces (asphalt, concrete)|
| H     | 200000        | very hard and dense (water)      |

You can also pass a flow resistivity number directly, or `None` for an acoustically rigid surface.

### Weather

`Weather(A, B, z0, temperature, humidity, pressure, Cv2, CT2)` describes the
Nord2000 log-lin sound speed profile `c(z) = A ln(z/z0 + 1) + B z + c0` and
the turbulence. `Weather.from_meteo(wind_speed, wind_height, wind_direction,
temperature_gradient, z0)` builds one from simple meteorological data.

## Model components

* **Air absorption**: ISO 9613-1 pure-tone absorption coefficient.
* **Ground impedance**: Delany–Bazley model with the Nord2000 impedance classes A–H.
* **Ground reflection**: spherical-wave reflection coefficient
  `Q = Rp + (1 − Rp) F(w)`. `F(w)` is computed with the Faddeeva function.
* **Coherence**: contributions are summed partially coherently. Cross terms
  are multiplied by a frequency-band averaging factor and a turbulence
  coherence factor, `exp(−(3/8)·0.364·(C_T²/T² + 22/3·C_v²/c0²)·k²·ρ^(5/3)·d)`.
  Here ρ is the mean separation between the two ray paths.
* **Non-flat, unscreened terrain**: the direct ray plus one reflected ray per
  terrain segment. Each reflected ray is weighted by the fraction of its
  Fresnel zone that lies on the segment. This also handles mixed ground,
  such as a road next to grass.
* **Screening**: screening edges are the vertices of the upper convex hull of
  source, terrain and receiver. More than two edges are reduced to two
  equivalent edges. Single- and double-edge wedge diffraction uses Pierce's
  uniform asymptotic solution, with the reflection coefficients of the wedge
  faces. Fresnel-weighted ground reflections are added on the source side and
  the receiver side by image sources and image receivers.
* **Transition**: an edge just below the line of sight blends smoothly between
  the screened and unscreened solutions, so results stay continuous as a
  screen rises through the line of sight.
* **Refraction**: the log-lin profile is replaced by an equivalent linear
  gradient. Circular rays in that gradient are straightened by curving the
  terrain: downward refraction makes the ground concave, and upward
  refraction makes it convex, which creates a shadow zone that is treated by
  diffraction.

## Simplifications compared with the full Nord2000 specification

This package covers the core propagation model. Where it departs from the
full Nord2000 specification:

* **Scattering into shadow zones**: Nord2000 models turbulence scattering into
  shadow zones separately. Here an energetic floor (`shadow_floor`, default
  −30 dB re free field) stands in for it, and only screened paths use it.
* **Ground roughness and scattering zones**: not modelled.
* **Equivalent linear gradient**: fitted by least squares over the heights
  0 … max(h_S, h_R) + 0.1·d. For downward refraction, the ray apex height is
  added to that range.
* **Multiple ground reflections**: strong downward refraction at long range
  can produce several ground reflections. The model only approximates these
  through the curved-terrain transformation.
* **Fresnel zone size**: λ/8 by default. Change it with the `fresnel_fraction`
  parameter.
* **Source models**: the Nord2000 road and rail source models are not
  included. Supply a sound power spectrum per 1/3-octave band instead.

## Tests

```bash
pytest
```

The tests check air absorption against the ISO 9613-1 tables, the +6 dB
hard-ground limit, the −6 dB knife-edge shadow boundary, continuity as a
screen crosses the line of sight, reciprocity, and the qualitative effects
of refraction and mixed ground.
