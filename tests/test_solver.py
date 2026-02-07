import numpy as np

from gnss_twin.meas import NoiseModel, compute_measurements
from gnss_twin.receiver import wls_solve
from gnss_twin.sat import SyntheticOrbitModel


def test_wls_solver_converges() -> None:
    rng = np.random.default_rng(1)
    orbit = SyntheticOrbitModel(num_sats=8)
    sats = orbit.get_state(0.0)
    receiver_truth = np.array([1_100_000.0, -4_800_000.0, 3_900_000.0])
    noise = NoiseModel(sigma_m=0.0)
    measurements = compute_measurements(receiver_truth, 0.0, sats, noise, rng)
    solution = wls_solve(measurements, receiver_truth + 50.0)
    assert np.linalg.norm(solution.position_ecef_m - receiver_truth) < 1e-2
    assert solution.dop["PDOP"] > 0.0
