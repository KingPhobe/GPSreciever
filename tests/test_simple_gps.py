import numpy as np

from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.visibility import apply_elevation_mask


def test_simple_gps_states_are_finite() -> None:
    constellation = SimpleGpsConstellation(SimpleGpsConfig(num_sats=24, seed=42))
    states = constellation.get_sv_states(0.0)
    assert len(states) == 24
    speeds = []
    for sv in states:
        assert np.all(np.isfinite(sv.pos_ecef_m))
        assert np.all(np.isfinite(sv.vel_ecef_mps))
        speeds.append(float(np.linalg.norm(sv.vel_ecef_mps)))
    assert all(2_500.0 < speed < 3_500.0 for speed in speeds)


def test_simple_gps_visibility_mask() -> None:
    receiver_ecef = np.array([6_378_000.0, 0.0, 0.0])
    constellation = SimpleGpsConstellation(SimpleGpsConfig(num_sats=24, seed=7))
    states = constellation.get_sv_states(0.0)
    visible = apply_elevation_mask(receiver_ecef, states, elev_mask_deg=10.0)
    assert 8 <= len(visible) <= 12


def test_simple_gps_seed_repeatability() -> None:
    constellation_a = SimpleGpsConstellation(SimpleGpsConfig(num_sats=12, seed=123))
    constellation_b = SimpleGpsConstellation(SimpleGpsConfig(num_sats=12, seed=123))
    ids_a = [sv.sv_id for sv in constellation_a.get_sv_states(0.0)]
    ids_b = [sv.sv_id for sv in constellation_b.get_sv_states(0.0)]
    assert ids_a == ids_b
    assert len(ids_a) == len(ids_b)
