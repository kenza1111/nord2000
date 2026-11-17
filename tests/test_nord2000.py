import math

import numpy as np
import pytest

from nord2000 import (
    Nord2000,
    TerrainProfile,
    Weather,
    a_weighting,
    air_absorption,
    spherical_reflection,
)
from nord2000.diffraction import aux_fresnel, diffraction_integral
from nord2000.geometry import fresnel_zone_weight


def test_air_absorption_matches_iso_9613_1_table():
    # ISO 9613-1 Table 1: 20 degC, 70 % RH, 101.325 kPa [dB/km]
    f = np.array([125, 250, 500, 1000, 2000])
    expected = np.array([0.34, 1.13, 2.80, 4.98, 9.02])
    got = air_absorption(f, 20.0, 70.0) * 1000
    np.testing.assert_allclose(got, expected, rtol=0.03)


def test_a_weighting_reference_values():
    np.testing.assert_allclose(a_weighting([100, 1000, 10000]), [-19.1, 0.0, -2.5], atol=0.1)


def test_rigid_reflection_coefficient_is_one():
    assert spherical_reflection(1000, math.inf, 10, 0.1) == 1
    q = spherical_reflection(100, 200000, 10, 0.1)
    assert abs(q - 1) < 0.05


def test_soft_ground_grazing_reflection_tends_to_minus_one():
    q = spherical_reflection(8000, 200, 1000, 1e-4)
    assert q.real < -0.9


def test_aux_fresnel_continuity_at_switch():
    f1, g1 = aux_fresnel(5.0)
    f2, g2 = aux_fresnel(5.0 + 1e-9)
    assert abs(f1 - f2) < 1e-4 and abs(g1 - g2) < 1e-4
    assert diffraction_integral(0.0) == complex(0.5, -0.5)


def test_fresnel_weights_partition_coplanar_segments():
    s, r = np.array([0.0, 1.0]), np.array([100.0, 1.5])
    w1, *_ = fresnel_zone_weight(s, r, [0, 0], [40, 0], 0.5)
    w2, *_ = fresnel_zone_weight(s, r, [40, 0], [100, 0], 0.5)
    assert w1 + w2 == pytest.approx(1.0)
    assert 0 < w1 < 1


def test_hard_ground_gives_plus_6_db_at_low_frequency():
    m = Nord2000(frequencies=[25, 50])
    r = m.propagate(TerrainProfile.flat(0, 100, "H"), 0, 1, 100, 1.5)
    np.testing.assert_allclose(r.terrain, 6.0, atol=0.1)


def test_soft_ground_dip_and_high_frequency_decorrelation():
    m = Nord2000()
    r = m.propagate(TerrainProfile.flat(0, 100, "D"), 0, 1, 100, 1.5)
    assert r.terrain.min() < -10  # ground dip
    # at high frequencies coherence is lost: about +3 dB or less
    assert abs(r.terrain[-1]) < 4


def test_divergence_and_total():
    m = Nord2000(frequencies=[1000])
    r = m.propagate(TerrainProfile.flat(0, 100, "G"), 0, 1, 100, 1)
    assert r.divergence[0] == pytest.approx(-10 * math.log10(4 * math.pi * 100**2))
    lp = r.receiver_levels([100.0])
    assert lp[0] == pytest.approx(100 + r.attenuation[0])


def test_screen_attenuates_and_is_continuous_across_line_of_sight():
    m = Nord2000(frequencies=[250, 1000, 4000])
    levels = []
    heights = np.linspace(0.6, 2.4, 37)
    for h in heights:
        p = TerrainProfile.flat(0, 100, "G").with_screen(50, h)
        levels.append(m.propagate(p, 0, 1, 100, 1.5).terrain)
    levels = np.array(levels)
    assert np.max(np.abs(np.diff(levels, axis=0))) < 2.5
    # a 6 m screen attenuates more than a low one
    high = m.propagate(TerrainProfile.flat(0, 100, "G").with_screen(50, 6), 0, 1, 100, 1.5)
    assert np.all(high.terrain < levels[0] - 5)


def test_thin_screen_on_rigid_ground_at_shadow_boundary():
    """Receiver on the shadow boundary of a rigid knife edge: ~ -6 dB per path."""
    m = Nord2000(frequencies=[1000], shadow_floor=None, weather=Weather(Cv2=0, CT2=0))
    # Source and receiver high above ground so that ground paths are negligible
    # compared with the screen path -> use very soft ground (class A) far below.
    prof = TerrainProfile.flat(0, 100, "A").with_screen(50, 50)
    r = m.propagate(prof, 0, 50, 100, 50)
    assert r.terrain[0] == pytest.approx(-6.0, abs=1.5)


def test_double_screen_more_attenuation_than_single():
    m = Nord2000(frequencies=[500, 2000])
    one = TerrainProfile.flat(0, 100, "G").with_screen(40, 4)
    two = one.with_screen(60, 4)
    r1 = m.propagate(one, 0, 1, 100, 1.5)
    r2 = m.propagate(two, 0, 1, 100, 1.5)
    assert r2.n_edges == 2
    assert np.all(r2.terrain < r1.terrain)


def test_thick_barrier_and_hill():
    m = Nord2000(frequencies=[500])
    barrier = TerrainProfile.flat(0, 100, "D").with_screen(45, 4, thickness=10)
    assert m.propagate(barrier, 0, 1, 100, 1.5).n_edges == 2
    hill = TerrainProfile([(0, 0), (50, 10), (100, 0)], "D")
    assert m.propagate(hill, 0, 1, 100, 1.5).terrain[0] < -10


def test_reverse_direction_is_reciprocal():
    m = Nord2000(frequencies=[125, 1000])
    p = TerrainProfile([(0, 0), (30, 0), (60, 3), (100, 3)], ["D", "G", "D"]).with_screen(70, 2)
    a = m.propagate(p, 0, 1, 100, 2).terrain
    b = m.propagate(p, 100, 2, 0, 1).terrain
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_refraction_upwind_shadow_and_downwind_enhancement():
    prof = TerrainProfile.flat(0, 500, "D")
    neutral = Nord2000().propagate(prof, 0, 1, 500, 2)
    up = Nord2000(Weather.from_meteo(wind_speed=5, wind_direction=180)).propagate(prof, 0, 1, 500, 2)
    down = Nord2000(Weather.from_meteo(wind_speed=5)).propagate(prof, 0, 1, 500, 2)
    assert up.gradient < 0 < down.gradient
    assert up.n_edges > 0
    hf = slice(15, None)  # >= 800 Hz
    assert np.mean(up.terrain[hf]) < np.mean(neutral.terrain[hf]) - 10
    assert np.mean(down.terrain[hf]) >= np.mean(neutral.terrain[hf]) - 1


def test_mixed_ground_between_hard_and_soft():
    m = Nord2000(frequencies=[500])
    soft = m.propagate(TerrainProfile.flat(0, 100, "D"), 0, 1, 100, 1.5).terrain[0]
    hard = m.propagate(TerrainProfile.flat(0, 100, "G"), 0, 1, 100, 1.5).terrain[0]
    mixed = m.propagate(TerrainProfile([(0, 0), (50, 0), (100, 0)], ["D", "G"]), 0, 1, 100, 1.5).terrain[0]
    assert min(soft, hard) <= mixed <= max(soft, hard)


def test_receiver_level_a():
    m = Nord2000()
    r = m.propagate(TerrainProfile.flat(0, 50, "D"), 0, 0.5, 50, 1.5)
    la = r.receiver_level_a(np.full(len(r.frequencies), 90.0))
    assert 40 < la < 70
