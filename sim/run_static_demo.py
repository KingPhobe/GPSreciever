"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import save_epochs_csv, save_epochs_npz
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
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.runtime import Engine
from gnss_twin.sat.visibility import visible_sv_states
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import ecef_to_lla, lla_to_ecef


def run_static_demo(cfg: SimConfig, run_dir: Path, save_figs: bool = True) -> Path:
    engine = Engine(cfg)
    receiver_lla = engine.receiver_lla
    receiver_truth = engine.receiver_truth_ecef
    receiver_clock = engine.receiver_clock
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
        fix_flags=FixFlags(
            fix_type="3D",
            valid=True,
            sv_used=[dummy_sv.sv_id],
            sv_rejected=[],
            sv_count=1,
            sv_in_view=1,
            mask_ok=True,
            pdop=1.5,
            gdop=2.0,
            chi_square=0.8,
            chi_square_threshold=3.8,
        ),
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
        per_sv_stats={dummy_sv.sv_id: {"residual_m": 1.0, "used": 1.0, "locked": 0.0}},
    )
    print(epoch_log)
    rx_lat, rx_lon, rx_alt = ecef_to_lla(*receiver_truth)
    print(f"Receiver LLA (deg, deg, m): ({rx_lat:.6f}, {rx_lon:.6f}, {rx_alt:.2f})")
    print(f"Receiver ECEF (m): {receiver_truth}")

    sv_overhead = lla_to_ecef(rx_lat, rx_lon, 20_200_000.0)
    elev_deg, az_deg = elev_az_from_rx_sv(receiver_truth, sv_overhead)
    print(f"Sample elevation/azimuth (deg): ({elev_deg:.2f}, {az_deg:.2f})")

    measurement_source: SyntheticMeasurementSource = engine.measurement_source
    first_epoch_meas = measurement_source.get_measurements(0.0)
    print("First-epoch pseudoranges (m):")
    for meas in first_epoch_meas:
        print(
            f"  {meas.sv_id}: {meas.pr_m:.3f} (elev {meas.elev_deg:.2f} deg, cn0 {meas.cn0_dbhz:.1f})"
        )
    for t in range(5):
        sv_states = engine.constellation.get_sv_states(float(t))
        visible = visible_sv_states(receiver_truth, sv_states, elevation_mask_deg=cfg.elev_mask_deg)
        print(f"{len(visible)} visible satellites at t={t}s")

    epochs = engine.run(0.0, cfg.duration, cfg.dt)

    run_dir.mkdir(parents=True, exist_ok=True)
    if save_figs:
        from gnss_twin.plots import save_run_plots

        output_dir = save_run_plots(epochs, out_dir=run_dir.parent, run_name=run_dir.name)
    else:
        output_dir = run_dir
    save_epochs_npz(output_dir / "epoch_logs.npz", epochs)
    save_epochs_csv(output_dir / "epoch_logs.csv", epochs)
    return output_dir / "epoch_logs.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the static GNSS twin demo.")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Duration in seconds.")
    parser.add_argument("--out-dir", type=str, default="out", help="Output directory root.")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for outputs.")
    parser.add_argument("--use-ekf", action="store_true", help="Enable EKF navigation filter.")
    parser.add_argument("--rng-seed", type=int, default=None, help="Random seed for simulation.")
    parser.add_argument(
        "--attack-name",
        type=str,
        default="none",
        help="Attack model name to apply (default: none).",
    )
    parser.add_argument(
        "--attack-param",
        action="append",
        default=[],
        help="Attack parameter in key=value form (repeatable).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving run plots.",
    )
    args = parser.parse_args()
    attack_params = _parse_attack_params(args.attack_param, args.attack_name)
    cfg = SimConfig(
        duration=args.duration_s,
        use_ekf=args.use_ekf,
        attack_name=args.attack_name,
        attack_params=attack_params,
        rng_seed=args.rng_seed if args.rng_seed is not None else 42,
    )
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / run_name
    epoch_log_path = run_static_demo(cfg, run_dir, save_figs=not args.no_plots)
    print(f"Saved outputs to {epoch_log_path.parent}")


def _parse_attack_params(raw_params: list[str], attack_name: str) -> dict[str, float | str]:
    params: dict[str, float | str] = {}
    for raw_param in raw_params:
        if "=" not in raw_param:
            raise ValueError(f"Invalid --attack-param '{raw_param}'; expected key=value.")
        key, value = raw_param.split("=", 1)
        if not key:
            raise ValueError("Attack parameter key cannot be empty.")
        params[key] = _coerce_param_value(value)
    if attack_name.lower() == "spoof_pr_ramp":
        target_sv = params.get("target_sv")
        if not target_sv or str(target_sv).strip() == "":
            raise ValueError("spoof_pr_ramp requires --attack-param target_sv=G##")
    return params


def _coerce_param_value(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
