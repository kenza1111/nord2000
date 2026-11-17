"""Example: road-side receiver behind a noise barrier, three weather cases."""

import numpy as np

from nord2000 import NOMINAL_FREQUENCIES, Nord2000, TerrainProfile, Weather

# 200 m profile: 20 m of asphalt (class G) then grass (class D),
# with a 3 m rigid noise barrier 25 m from the source.
profile = TerrainProfile([(0, 0), (20, 0), (200, 0)], ["G", "D"]).with_screen(25, 3.0)

# Flat source spectrum, 100 dB per 1/3 octave band.
lw = np.full(len(NOMINAL_FREQUENCIES), 100.0)

cases = {
    "neutral": Weather(),
    "downwind 5 m/s": Weather.from_meteo(wind_speed=5.0, wind_direction=0),
    "upwind 5 m/s": Weather.from_meteo(wind_speed=5.0, wind_direction=180),
}

np.set_printoptions(precision=1, suppress=True, linewidth=150)
for name, weather in cases.items():
    model = Nord2000(weather)
    res = model.propagate(profile, source_x=0.0, source_height=0.5,
                          receiver_x=200.0, receiver_height=4.0)
    print(f"{name:16s} gradient={res.gradient:+.3f} 1/s  edges={res.n_edges}  "
          f"L_A={res.receiver_level_a(lw):5.1f} dB(A)")
    print("  terrain effect dL_t [dB]:", res.terrain)
