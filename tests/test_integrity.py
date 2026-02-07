from dataclasses import replace

import numpy as np

from gnss_twin.integrity import raim_fde
from gnss_twin.meas import NoiseModel, compute_measurements
from gnss_twin.sat import SyntheticOrbitModel


def test_raim_fde_flags_fault() -> None:
    rng = np.random.default_rng(2)
    orbit = SyntheticOrbitModel(num_sats=7)
    sats = orbit.get_state(0.0)
    receiver_truth = np.array([1_500_000.0, -4_200_000.0, 3_200_000.0])
    noise = NoiseModel(sigma_m=0.0)
    measurements = compute_measurements(receiver_truth, 0.0, sats, noise, rng)
    measurements[0] = replace(measurements[0], pseudorange_m=measurements[0].pseudorange_m + 50.0)
    solution, report = raim_fde(measurements, receiver_truth + 20.0, threshold_m=10.0)
    assert report.fde_used
    assert report.excluded_prn == measurements[0].prn
    assert np.linalg.norm(solution.position_ecef_m - receiver_truth) < 5.0
