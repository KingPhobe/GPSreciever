import numpy as np

from gnss_twin.meas import NoiseModel, compute_measurements
from gnss_twin.sat import SyntheticOrbitModel


def test_synthetic_orbit_and_measurements() -> None:
    rng = np.random.default_rng(0)
    orbit = SyntheticOrbitModel(num_sats=6)
    sats = orbit.get_state(0.0)
    assert len(sats) == 6
    receiver = np.array([1_000_000.0, 2_000_000.0, 3_000_000.0])
    noise = NoiseModel(sigma_m=0.0)
    measurements = compute_measurements(receiver, 0.0, sats, noise, rng)
    assert len(measurements) == 6
    assert all(meas.truth_range_m > 0.0 for meas in measurements)
