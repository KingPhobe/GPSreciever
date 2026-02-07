"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import numpy as np
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
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import ecef_to_lla, lla_to_ecef


def main() -> None:
    receiver_lla = (37.4275, -122.1697, 30.0)
    receiver_truth = lla_to_ecef(*receiver_lla)
    receiver_clock = 4.2e-6
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
    rx_lat, rx_lon, rx_alt = ecef_to_lla(*receiver_truth)
    print(f"Receiver LLA (deg, deg, m): ({rx_lat:.6f}, {rx_lon:.6f}, {rx_alt:.2f})")
    print(f"Receiver ECEF (m): {receiver_truth}")

    sv_overhead = lla_to_ecef(rx_lat, rx_lon, 20_200_000.0)
    elev_deg, az_deg = elev_az_from_rx_sv(receiver_truth, sv_overhead)
    print(f"Sample elevation/azimuth (deg): ({elev_deg:.2f}, {az_deg:.2f})")


if __name__ == "__main__":
    main()
