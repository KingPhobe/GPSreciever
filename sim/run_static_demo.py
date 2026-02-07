"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import numpy as np

from gnss_twin.integrity import raim_fde
from gnss_twin.meas import NoiseModel, compute_measurements
from gnss_twin.models import (
    DopMetrics,
    EpochLog,
    FixFlags,
    GnssMeasurement,
    PvtSolution,
    ReceiverTruth,
    ResidualStats,
    SvState,
)
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

    dummy_meas = GnssMeasurement(
        sv_id="G01",
        t=0.0,
        pr_m=2.35e7,
        prr_mps=None,
        cn0_dbhz=45.0,
        elev_deg=30.0,
        az_deg=120.0,
        flags={"valid": True},
    )
    dummy_sv = SvState(
        sv_id="G01",
        t=0.0,
        pos_ecef_m=np.array([15_600_000.0, 0.0, 21_400_000.0]),
        vel_ecef_mps=np.array([0.0, 2_600.0, 1_200.0]),
        clk_bias_s=0.0,
        clk_drift_sps=0.0,
    )
    dummy_solution = PvtSolution(
        pos_ecef=receiver_truth,
        vel_ecef=None,
        clk_bias_s=receiver_clock,
        clk_drift_sps=0.0,
        dop=DopMetrics(gdop=2.0, pdop=1.5, hdop=1.0, vdop=1.2),
        residuals=ResidualStats(rms_m=1.0, mean_m=0.1, max_m=2.5, chi_square=0.8),
        fix_flags=FixFlags(fix_type="3D", valid=True, sv_used=[dummy_sv.sv_id], sv_rejected=[]),
    )
    dummy_truth = ReceiverTruth(
        pos_ecef_m=receiver_truth,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=receiver_clock,
        clk_drift_sps=0.0,
    )
    epoch_log = EpochLog(
        t=0.0,
        meas=[dummy_meas],
        solution=dummy_solution,
        truth=dummy_truth,
        per_sv_stats={dummy_sv.sv_id: {"residual_m": 1.0, "used": 1.0}},
    )
    print(epoch_log)


if __name__ == "__main__":
    main()
