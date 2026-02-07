"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
from gnss_twin.meas.pseudorange import LIGHT_SPEED_MPS, SyntheticMeasurementSource, geometric_range_m
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.sat.visibility import visible_sv_states
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
        sigma_pr_m=1.0,
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

    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=42))
    measurement_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=dummy_truth,
        cn0_zenith_dbhz=47.0,
    )
    first_epoch_meas = measurement_source.get_measurements(0.0)
    print("First-epoch pseudoranges (m):")
    for meas in first_epoch_meas:
        print(
            f"  {meas.sv_id}: {meas.pr_m:.3f} (elev {meas.elev_deg:.2f} deg, cn0 {meas.cn0_dbhz:.1f})"
        )
    for t in range(5):
        sv_states = constellation.get_sv_states(float(t))
        visible = visible_sv_states(receiver_truth, sv_states, elevation_mask_deg=10.0)
        print(f"{len(visible)} visible satellites at t={t}s")

    times = np.arange(0.0, 60.0, 1.0)
    rms_errors: list[float] = []
    pos_errors: list[float] = []
    pdop_series: list[float] = []
    last_pos = receiver_truth + 100.0
    last_clk = receiver_clock
    for t in times:
        sv_states = constellation.get_sv_states(float(t))
        sv_by_id = {state.sv_id: state for state in sv_states}
        meas = measurement_source.get_measurements(float(t))
        errors = []
        for m in meas:
            state = sv_by_id[m.sv_id]
            geometric = geometric_range_m(receiver_truth, state.pos_ecef_m)
            clock_term = LIGHT_SPEED_MPS * (measurement_source.receiver_clock_bias_s - state.clk_bias_s)
            errors.append(m.pr_m - geometric - clock_term)
        rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else 0.0
        rms_errors.append(rms)

        solution = wls_pvt(meas, sv_states, initial_pos_ecef_m=last_pos, initial_clk_bias_s=last_clk)
        if solution is None:
            pos_errors.append(float("nan"))
            pdop_series.append(float("nan"))
        else:
            pos_errors.append(float(np.linalg.norm(solution.pos_ecef_m - receiver_truth)))
            pdop_series.append(solution.dop.pdop)
            last_pos = solution.pos_ecef_m
            last_clk = solution.clk_bias_s

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(times, pos_errors, marker="o", markersize=3)
    axes[0].set_title("WLS Position Error over 60s")
    axes[0].set_ylabel("Position error (m)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, pdop_series, marker="o", markersize=3, color="tab:orange")
    axes[1].set_title("WLS PDOP over 60s")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("PDOP")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
