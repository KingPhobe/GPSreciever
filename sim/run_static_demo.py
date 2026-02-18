"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import argparse
import csv
import random
import warnings
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import save_epochs_csv, save_epochs_npz
from gnss_twin.nmea.neo_m8n_output import NmeaEmit, NeoM8nNmeaOutput
from gnss_twin.runtime.factory import build_engine_with_truth, build_epoch_log
from gnss_twin.sat.visibility import visible_sv_states
from gnss_twin.timing import Authenticator, AuthenticatorConfig
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import ecef_to_lla, lla_to_ecef
from sim.run_table import write_run_table_from_epoch_logs


def build_engine(cfg: SimConfig):
    """Return a fully wired SimulationEngine identical to the static demo wiring."""
    engine, _ = build_engine_with_truth(cfg)
    return engine


def run_static_demo(
    cfg: SimConfig,
    run_dir: Path,
    save_figs: bool = True,
    *,
    verbose: bool = False,
) -> Path:
    engine, receiver_truth_state = build_engine_with_truth(cfg)
    seed = int(cfg.rng_seed)
    np.random.seed(seed)
    random.seed(seed)
    authenticator = Authenticator(replace(AuthenticatorConfig(), seed=seed))
    receiver_truth = receiver_truth_state.pos_ecef_m
    measurement_source = engine.meas_src
    integrity_checker = engine.integrity_checker
    constellation = measurement_source.constellation
    if verbose:
        rx_lat, rx_lon, rx_alt = ecef_to_lla(*receiver_truth)
        print(f"Receiver LLA (deg, deg, m): ({rx_lat:.6f}, {rx_lon:.6f}, {rx_alt:.2f})")
        print(f"Receiver ECEF (m): {receiver_truth}")

        sv_overhead = lla_to_ecef(rx_lat, rx_lon, 20_200_000.0)
        elev_deg, az_deg = elev_az_from_rx_sv(receiver_truth, sv_overhead)
        print(f"Sample elevation/azimuth (deg): ({elev_deg:.2f}, {az_deg:.2f})")

        first_epoch_meas = measurement_source.get_measurements(0.0)
        print(f"First epoch measurement count: {len(first_epoch_meas)}")
        print("First-epoch pseudoranges (m):")
        for meas in first_epoch_meas:
            print(
                f"  {meas.sv_id}: {meas.pr_m:.3f} (elev {meas.elev_deg:.2f} deg, cn0 {meas.cn0_dbhz:.1f})"
            )
        for t in range(5):
            sv_states = constellation.get_sv_states(float(t))
            visible = visible_sv_states(receiver_truth, sv_states, elevation_mask_deg=cfg.elev_mask_deg)
            print(f"{len(visible)} visible satellites at t={t}s")

    epochs = []
    nmea = NeoM8nNmeaOutput(rate_hz=1.0, talker="GN")
    nmea_emits: list[NmeaEmit] = []
    t0_utc = datetime.now(timezone.utc)
    times = np.arange(0.0, cfg.duration, cfg.dt)
    attack_name = cfg.attack_name or "none"
    for t in times:
        step = engine.step(float(t))
        epoch_log = build_epoch_log(
            t_s=float(t),
            step_out=step,
            integrity_checker=integrity_checker,
            attack_name=attack_name,
        )
        raim_pass = bool(epoch_log.raim_pass) if epoch_log.raim_pass is not None else False
        fix_valid = bool(epoch_log.fix_valid) if epoch_log.fix_valid is not None else False
        gnss_valid_for_disc = raim_pass and fix_valid
        sol = step.get("sol")
        if sol is not None:
            clk_bias_true_s = measurement_source.receiver_clock_bias_s
            clk_bias_est_s = sol.clk_bias_s
            pps_platform_edge_s = authenticator._pps_builder.platform_edge(
                t,
                clk_bias_est_s,
                clk_bias_true_s,
            )
        else:
            pps_platform_edge_s = None
        pps_ref_edge_s = authenticator._pps_builder.ref_edge(t)
        auth_tel, pps_tel = authenticator.step(
            t,
            pps_platform_edge_s=pps_platform_edge_s if pps_platform_edge_s is not None else pps_ref_edge_s,
            pps_ref_edge_s=pps_ref_edge_s,
            gnss_valid=gnss_valid_for_disc,
        )
        epoch_log = replace(
            epoch_log,
            pps_ref_edge_s=pps_tel.ref_edge_s,
            pps_platform_edge_s=pps_tel.platform_edge_s,
            pps_auth_edge_s=pps_tel.auth_edge_s,
            pps_platform_minus_ref_s=pps_tel.platform_minus_ref_s,
            pps_auth_minus_ref_s=pps_tel.auth_minus_ref_s,
            pps_platform_minus_auth_s=-pps_tel.auth_minus_platform_s,
            auth_bit=auth_tel.auth_bit,
            auth_locked=auth_tel.locked,
            auth_mode="holdover" if auth_tel.holdover_active else ("locked" if auth_tel.locked else "unlocked"),
            auth_sigma_t_s=auth_tel.rms_error_s,
            auth_reason_codes=[auth_tel.reason_code],
        )
        epochs.append(epoch_log)
        if sol is not None:
            lat_deg, lon_deg, alt_m = ecef_to_lla(*sol.pos_ecef)
            raim_valid = (
                (step.get("integrity") is not None)
                and (not step["integrity"].is_suspect)
                and (not step["integrity"].is_invalid)
            )
            num_sats = int(epoch_log.sats_used or 0)
            hdop = epoch_log.hdop
            if hdop is None and epoch_log.pdop is not None:
                hdop = float(epoch_log.pdop)
            elif hdop is None:
                hdop = float("nan")
            t_utc = t0_utc + timedelta(seconds=float(t))
            emits = nmea.step(
                float(t),
                t_utc=t_utc,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                alt_m=alt_m,
                raim_valid=raim_valid,
                num_sats=num_sats,
                hdop=float(hdop),
            )
            nmea_emits.extend(emits)

    run_dir.mkdir(parents=True, exist_ok=True)
    if save_figs:
        from gnss_twin.plots import save_run_plots
        from gnss_twin.plots.conops_plots import save_conops_plots

        output_dir = save_run_plots(epochs, out_dir=run_dir.parent, run_name=run_dir.name)
        save_conops_plots(epochs, output_dir)
    else:
        output_dir = run_dir
    save_epochs_npz(output_dir / "epoch_logs.npz", epochs)
    save_epochs_csv(output_dir / "epoch_logs.csv", epochs)
    write_run_table_from_epoch_logs(output_dir / "epoch_logs.csv", output_dir / "run_table.csv", cfg)
    (output_dir / "nmea_output.nmea").write_text(
        "".join(f"{emit.nmea_sentence}\r\n" for emit in nmea_emits),
        encoding="utf-8",
    )
    with (output_dir / "nmea_output.csv").open("w", encoding="utf-8", newline="") as nmea_csv:
        writer = csv.DictWriter(
            nmea_csv,
            fieldnames=["t_s", "t_utc_iso", "valid", "sentence_type", "talker", "nmea_sentence"],
        )
        writer.writeheader()
        for emit in nmea_emits:
            writer.writerow({
                "t_s": emit.t_s,
                "t_utc_iso": emit.t_utc_iso,
                "valid": emit.valid,
                "sentence_type": emit.sentence_type,
                "talker": emit.talker,
                "nmea_sentence": emit.nmea_sentence,
            })
    with (output_dir / "run_metadata.csv").open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=[
                "rx_lat_deg",
                "rx_lon_deg",
                "rx_alt_m",
                "nmea_profile",
                "nmea_rate_hz",
                "nmea_msgs",
                "nmea_talker",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rx_lat_deg": cfg.rx_lat_deg,
                "rx_lon_deg": cfg.rx_lon_deg,
                "rx_alt_m": cfg.rx_alt_m,
                "nmea_profile": "NEO-M8N",
                "nmea_rate_hz": 1.0,
                "nmea_msgs": "GGA,RMC",
                "nmea_talker": "GN",
            }
        )
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
    parser.add_argument("--verbose", action="store_true", help="Print debug info.")
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
    epoch_log_path = run_static_demo(
        cfg,
        run_dir,
        save_figs=not args.no_plots,
        verbose=args.verbose,
    )
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

    if "ramp_rate_mps" not in params:
        if "slope_mps" in params:
            warnings.warn("Deprecated attack param slope_mps; use ramp_rate_mps", stacklevel=2)
            params["ramp_rate_mps"] = float(params["slope_mps"])
        elif "slope" in params:
            warnings.warn("Deprecated attack param slope; use ramp_rate_mps", stacklevel=2)
            params["ramp_rate_mps"] = float(params["slope"])

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
