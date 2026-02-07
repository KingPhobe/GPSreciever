"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import numpy as np

from gnss_twin.integrity import raim_fde
from gnss_twin.meas import NoiseModel, compute_measurements
from gnss_twin.sat import SyntheticOrbitModel
from gnss_twin.utils import get_logger, plot_residuals, plot_solution_errors


def main() -> None:
    logger = get_logger()
    rng = np.random.default_rng(42)
    orbit = SyntheticOrbitModel(num_sats=8)
    receiver_truth = np.array([1_113_000.0, -4_842_000.0, 3_985_000.0])
    receiver_clock = 2.5e-6
    noise = NoiseModel(sigma_m=1.5)

    times = np.arange(0.0, 60.0, 5.0)
    errors = []
    residual_max = []
    for t in times:
        sats = orbit.get_state(float(t))
        measurements = compute_measurements(receiver_truth, receiver_clock, sats, noise, rng)
        solution, report = raim_fde(measurements, receiver_truth + 10.0)
        error = solution.position_ecef_m - receiver_truth
        errors.append(error)
        residual_max.append(report.max_residual_m)
        logger.info(
            "t=%.1fs valid=%s max_residual=%.2fm PDOP=%.2f",
            t,
            report.valid,
            report.max_residual_m,
            solution.dop["PDOP"],
        )

    errors_arr = np.vstack(errors)
    plot_solution_errors(times, errors_arr)
    plot_residuals(times, np.array(residual_max))


if __name__ == "__main__":
    main()
