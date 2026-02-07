import numpy as np

from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation


def test_constellation_returns_expected_number_of_sats() -> None:
    config = SimpleGpsConfig(num_planes=5, sats_per_plane=2, seed=1)
    constellation = SimpleGpsConstellation(config)
    sv_states = constellation.get_sv_states(0.0)
    assert len(sv_states) == 10


def test_positions_and_velocities_are_finite_and_plausible() -> None:
    config = SimpleGpsConfig(seed=2)
    constellation = SimpleGpsConstellation(config)
    sv_states = constellation.get_sv_states(123.0)
    speeds = []
    for state in sv_states:
        assert np.all(np.isfinite(state.pos_ecef_m))
        assert np.all(np.isfinite(state.vel_ecef_mps))
        speeds.append(np.linalg.norm(state.vel_ecef_mps))
    min_speed = min(speeds)
    max_speed = max(speeds)
    assert 2_500.0 < min_speed < 5_500.0
    assert 2_500.0 < max_speed < 5_500.0


def test_seeded_constellation_is_repeatable() -> None:
    config = SimpleGpsConfig(seed=7)
    constellation_a = SimpleGpsConstellation(config)
    constellation_b = SimpleGpsConstellation(config)
    sv_states_a = constellation_a.get_sv_states(42.0)
    sv_states_b = constellation_b.get_sv_states(42.0)
    ids_a = [state.sv_id for state in sv_states_a]
    ids_b = [state.sv_id for state in sv_states_b]
    assert ids_a == ids_b
    assert len(sv_states_a) == len(sv_states_b)
