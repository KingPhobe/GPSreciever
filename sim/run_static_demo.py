"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import argparse
import csv
import random
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from gnss_twin.attacks import AttackPipeline, create_attack
from gnss_twin.config import SimConfig
from gnss_twin.logger import save_epochs_csv, save_epochs_npz
from gnss_twin.models import (
    EpochLog,
    GnssMeasurement,
    PvtSolution,
    ReceiverTruth,
    SvState,
    fix_type_from_label,
)
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.nmea.neo_m8n_output import NmeaEmit, NeoM8nNmeaOutput
from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.receiver.ekf_nav import EkfNav
from gnss_twin.receiver.gating import postfit_gate
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.runtime import (
    ConopsStateMachine,
    RaimIntegrityChecker,
    SimulationEngine,
    default_pnt_config,
)
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.sat.visibility import visible_sv_states
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import ecef_to_lla, lla_to_ecef
from sim.run_table import write_run_table_from_epoch_logs


class _DemoSolver:
    def __init__(self, cfg: SimConfig, initial_pos: np.ndarray, initial_clk: float) -> None:
        self.cfg = cfg
        self.integrity_cfg = IntegrityConfig()
        self.tracker = SvTracker(self.integrity_cfg)
        self.last_pos = initial_pos + 100.0
        self.last_clk = float(initial_clk)
        self.last_t_s: float | None = None
        self.ekf = EkfNav() if cfg.use_ekf else None

    def solve(
        self,
        measurements: list[GnssMeasurement],
        sv_states: list[SvState],
        *,
        t_s: float | None = None,
    ) -> PvtSolution | None:
        used_meas = list(measurements)
        wls_solution = None
        if len(used_meas) >= 4:
            wls_solution = wls_pvt(
                used_meas,
                sv_states,
                initial_pos_ecef_m=self.last_pos,
                initial_clk_bias_s=self.last_clk,
            )
            if wls_solution is not None:
                sigmas_by_sv = {m.sv_id: m.sigma_pr_m for m in used_meas}
                offender = postfit_gate(
                    wls_solution.residuals_m,
                    sigmas_by_sv,
                    gate=self.cfg.postfit_gate_sigma,
                )
                if offender:
                    used_meas = [m for m in used_meas if m.sv_id != offender]

        solution, _ = integrity_pvt(
            used_meas,
            sv_states,
            initial_pos_ecef_m=self.last_pos,
            initial_clk_bias_s=self.last_clk,
            config=self.integrity_cfg,
            tracker=self.tracker,
        )

        if self.ekf is not None:
            dt = self.cfg.dt if self.last_t_s is None or t_s is None else float(t_s - self.last_t_s)
            if not self.ekf.initialized and wls_solution is not None:
                self.ekf.initialize_from_wls(wls_solution)
            if self.ekf.initialized and self.last_t_s is not None and t_s is not None:
                self.ekf.predict(dt)
            if self.ekf.initialized:
                self.ekf.update_pseudorange(
                    used_meas,
                    sv_states,
                    initial_pos_ecef_m=self.last_pos,
                    initial_clk_bias_s=self.last_clk,
                )
                self.ekf.update_prr(used_meas, sv_states)
                solution = PvtSolution(
                    pos_ecef=self.ekf.pos_ecef_m.copy(),
                    vel_ecef=self.ekf.vel_ecef_mps.copy(),
                    clk_bias_s=self.ekf.clk_bias_s,
                    clk_drift_sps=self.ekf.clk_drift_sps,
                    dop=solution.dop,
                    residuals=solution.residuals,
                    fix_flags=solution.fix_flags,
                )

        self.last_t_s = t_s
        if solution.fix_flags.fix_type != "NO FIX" and np.isfinite(solution.pos_ecef).all():
            self.last_pos = solution.pos_ecef
            self.last_clk = solution.clk_bias_s
        return solution


def build_engine_with_truth(cfg: SimConfig) -> tuple[SimulationEngine, ReceiverTruth]:
    """Return a fully wired SimulationEngine and receiver truth state."""
    seed = int(cfg.rng_seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    receiver_truth = lla_to_ecef(cfg.rx_lat_deg, cfg.rx_lon_deg, cfg.rx_alt_m)
    receiver_clock = 4.2e-6
    receiver_truth_state = ReceiverTruth(
        pos_ecef_m=receiver_truth,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=receiver_clock,
        clk_drift_sps=0.0,
    )
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=seed))
    measurement_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth_state,
        cn0_zenith_dbhz=47.0,
        cn0_min_dbhz=cfg.cn0_model_min_dbhz,
        rng=rng,
    )
    attack_name = cfg.attack_name or "none"
    attacks = [] if attack_name.lower() == "none" else [create_attack(attack_name, cfg.attack_params)]
    attack_pipeline = AttackPipeline(attacks) if attacks else None
    if attack_pipeline is not None:
        attack_pipeline.reset(seed)
    integrity_checker = RaimIntegrityChecker(sim_cfg=cfg)
    conops_sm = ConopsStateMachine(default_pnt_config())
    solver = _DemoSolver(cfg, receiver_truth, receiver_clock)
    engine = SimulationEngine(
        measurement_source,
        solver,
        integrity_checker,
        attack_pipeline,
        conops_sm,
    )
    return engine, receiver_truth_state


def build_engine(cfg: SimConfig) -> SimulationEngine:
    """Return a fully wired SimulationEngine identical to the static demo wiring."""
    engine, _ = build_engine_with_truth(cfg)
    return engine


def build_epoch_log(
    *,
    t_s: float,
    step_out: dict,
    receiver_truth_state: ReceiverTruth,
    integrity_checker: RaimIntegrityChecker,
    attack_name: str,
) -> EpochLog:
    """Build an EpochLog entry from a SimulationEngine step output."""
    sol = step_out["sol"]
    integrity = step_out["integrity"]
    conops = step_out.get("conops")
    attack_report = step_out.get("attack_report")
    applied_count = getattr(attack_report, "applied_count", 0)
    if applied_count:
        attack_pr_bias_mean_m = attack_report.pr_bias_sum_m / applied_count
        attack_prr_bias_mean_mps = attack_report.prr_bias_sum_mps / applied_count
    else:
        attack_pr_bias_mean_m = 0.0
        attack_prr_bias_mean_mps = 0.0
    per_sv_stats = getattr(integrity_checker, "last_per_sv_stats", {})
    return EpochLog(
        t=float(t_s),
        meas=step_out["meas_attacked"],
        solution=sol,
        truth=receiver_truth_state,
        t_s=float(t_s),
        fix_valid=sol.fix_flags.valid if sol is not None else None,
        raim_pass=sol.fix_flags.raim_passed if sol is not None else None,
        fix_type=fix_type_from_label(sol.fix_flags.fix_type) if sol is not None else None,
        sats_used=sol.fix_flags.sv_count if sol is not None else None,
        pdop=sol.dop.pdop if sol is not None else None,
        hdop=sol.dop.hdop if sol is not None else None,
        vdop=sol.dop.vdop if sol is not None else None,
        residual_rms_m=sol.residuals.rms_m if sol is not None else None,
        pos_ecef=sol.pos_ecef.copy() if sol is not None else None,
        vel_ecef=sol.vel_ecef.copy() if sol is not None and sol.vel_ecef is not None else None,
        clk_bias_s=sol.clk_bias_s if sol is not None else None,
        clk_drift_sps=sol.clk_drift_sps if sol is not None else None,
        nis=None,
        nis_alarm=integrity.is_suspect or integrity.is_invalid,
        attack_name=attack_name,
        attack_active=applied_count > 0,
        attack_pr_bias_mean_m=attack_pr_bias_mean_m,
        attack_prr_bias_mean_mps=attack_prr_bias_mean_mps,
        innov_dim=None,
        conops_status=conops.status.value if conops is not None else None,
        conops_mode5=conops.mode5.value if conops is not None else None,
        conops_reason_codes=list(conops.reason_codes) if conops is not None else [],
        integrity_p_value=integrity.p_value,
        integrity_residual_rms=integrity.residual_rms,
        integrity_num_sats_used=integrity.num_sats_used,
        integrity_excluded_sv_ids_count=len(integrity.excluded_sv_ids),
        per_sv_stats=per_sv_stats,
    )


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
    receiver_truth = receiver_truth_state.pos_ecef_m
    receiver_clock = receiver_truth_state.clk_bias_s
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
            receiver_truth_state=receiver_truth_state,
            integrity_checker=integrity_checker,
            attack_name=attack_name,
        )
        epochs.append(epoch_log)
        sol = step.get("sol")
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
