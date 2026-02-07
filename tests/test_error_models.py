import numpy as np

from gnss_twin.meas.clock_models import RandomWalkClock
from gnss_twin.meas.iono_klobuchar import klobuchar_delay_m
from gnss_twin.meas.noise import cn0_from_elevation, pseudorange_sigma_m
from gnss_twin.meas.tropo_saastamoinen import saastamoinen_delay_m


def test_noise_sigma_decreases_with_elevation_and_cn0() -> None:
    cn0_low = cn0_from_elevation(10.0)
    cn0_high = cn0_from_elevation(80.0)
    sigma_low = pseudorange_sigma_m(cn0_low, 10.0)
    sigma_high = pseudorange_sigma_m(cn0_high, 80.0)
    assert cn0_high > cn0_low
    assert sigma_high < sigma_low


def test_clock_deterministic_with_seed() -> None:
    clock_a = RandomWalkClock(seed=42)
    clock_b = RandomWalkClock(seed=42)
    steps = [1.0, 1.0, 0.5]
    states_a = [clock_a.step(dt) for dt in steps]
    states_b = [clock_b.step(dt) for dt in steps]
    for a, b in zip(states_a, states_b, strict=True):
        assert np.isclose(a.bias_s, b.bias_s)
        assert np.isclose(a.drift_sps, b.drift_sps)


def test_iono_tropo_non_negative() -> None:
    iono = klobuchar_delay_m(0.0, 37.0, -122.0, 30.0, 120.0)
    tropo = saastamoinen_delay_m(30.0, 37.0, 30.0)
    assert iono >= 0.0
    assert tropo >= 0.0
