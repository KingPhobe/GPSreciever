"""Interactive live simulation runner for GNSS twin demos."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gnss_twin.attacks import AttackPipeline, create_attack
from gnss_twin.config import SimConfig
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import ReceiverTruth
from gnss_twin.runtime import (
    ConopsStateMachine,
    RaimIntegrityChecker,
    SimulationEngine,
    default_pnt_config,
)
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef
from sim.run_static_demo import _DemoSolver


def parse_kv_list(raw_params: list[str]) -> dict[str, Any]:
    """Parse repeated key=value arguments into typed values."""
    params: dict[str, Any] = {}
    for raw_param in raw_params:
        if "=" not in raw_param:
            raise ValueError(f"Invalid --attack-param '{raw_param}'; expected key=value.")
        key, raw_value = raw_param.split("=", 1)
        if not key:
            raise ValueError("Attack parameter key cannot be empty.")
        params[key] = _coerce_kv_value(raw_value)
    return params


def _coerce_kv_value(raw_value: str) -> Any:
    value_lower = raw_value.lower()
    if value_lower in {"true", "false"}:
        return value_lower == "true"
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


class _LiveController:
    def __init__(self) -> None:
        self.paused = False
        self.quit_requested = False
        self.step_once = False

    def on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == " ":
            self.paused = not self.paused
            self.step_once = False
        elif key == "right":
            if self.paused:
                self.step_once = True
        elif key in {"q", "Q"}:
            self.quit_requested = True


class _LivePlotter:
    def __init__(self, max_points: int = 300) -> None:
        self.max_points = max_points
        self.t: list[float] = []
        self.residual_rms_m: list[float] = []
        self.pdop: list[float] = []
        self.clk_bias_s: list[float] = []
        self.attack_trace: list[float] = []

        plt.ion()
        self.fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        self.axes = axes
        (self.resid_line,) = axes[0].plot([], [], lw=1.8)
        (self.pdop_line,) = axes[1].plot([], [], lw=1.8)
        (self.clk_line,) = axes[2].plot([], [], lw=1.8)
        (self.attack_line,) = axes[3].plot([], [], lw=1.8)
        axes[0].set_ylabel("residual\nrms [m]")
        axes[1].set_ylabel("pdop")
        axes[2].set_ylabel("clk bias [s]")
        axes[3].set_ylabel("attack")
        axes[3].set_xlabel("time [s]")
        self.fig.suptitle("Live GNSS Twin Telemetry (Space pause/resume, Right step, q quit)")

    def update(
        self,
        t_s: float,
        residual_rms_m: float,
        pdop: float,
        clk_bias_s: float,
        attack_value: float,
    ) -> None:
        self.t.append(t_s)
        self.residual_rms_m.append(residual_rms_m)
        self.pdop.append(pdop)
        self.clk_bias_s.append(clk_bias_s)
        self.attack_trace.append(attack_value)
        if len(self.t) > self.max_points:
            self.t = self.t[-self.max_points :]
            self.residual_rms_m = self.residual_rms_m[-self.max_points :]
            self.pdop = self.pdop[-self.max_points :]
            self.clk_bias_s = self.clk_bias_s[-self.max_points :]
            self.attack_trace = self.attack_trace[-self.max_points :]

        self.resid_line.set_data(self.t, self.residual_rms_m)
        self.pdop_line.set_data(self.t, self.pdop)
        self.clk_line.set_data(self.t, self.clk_bias_s)
        self.attack_line.set_data(self.t, self.attack_trace)

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def _build_engine(cfg: SimConfig) -> SimulationEngine:
    seed = int(cfg.rng_seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    receiver_truth = lla_to_ecef(37.4275, -122.1697, 30.0)
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
        cn0_min_dbhz=cfg.cn0_min_dbhz,
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

    return SimulationEngine(
        measurement_source,
        solver,
        integrity_checker,
        attack_pipeline,
        conops_sm,
    )


def _json_safe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interactive live GNSS twin simulation.")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Duration in seconds.")
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Simulation step size in seconds (defaults to cfg.dt_s/cfg.dt/1.0).",
    )
    parser.add_argument("--use-ekf", action="store_true", help="Enable EKF navigation filter.")
    parser.add_argument("--rng-seed", type=int, default=42, help="Random seed for simulation.")
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
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (simulation-time / wall-time).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib UI and run loop in console-only mode.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default=None,
        help="Path to optional JSONL telemetry output for each epoch.",
    )
    args = parser.parse_args()

    attack_params = parse_kv_list(args.attack_param)
    cfg = SimConfig(
        duration=args.duration_s,
        dt=args.dt if args.dt is not None else 1.0,
        use_ekf=args.use_ekf,
        attack_name=args.attack_name,
        attack_params=attack_params,
        rng_seed=args.rng_seed,
    )
    dt = args.dt if args.dt is not None else float(getattr(cfg, "dt_s", getattr(cfg, "dt", 1.0)))

    engine = _build_engine(cfg)
    controller = _LiveController()
    plotter = None

    if not args.no_plots:
        plotter = _LivePlotter(max_points=300)
        plotter.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    sim_times = np.arange(0.0, cfg.duration + 1e-9, dt)
    last_tick = time.perf_counter()
    jsonl_fp = None

    if args.out_jsonl:
        out_jsonl_path = Path(args.out_jsonl)
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = out_jsonl_path.open("w", encoding="utf-8")

    try:
        for t_s in sim_times:
            if controller.quit_requested:
                break

            while controller.paused and not controller.step_once and not controller.quit_requested:
                if plotter is not None:
                    plt.pause(0.05)
                else:
                    time.sleep(0.05)

            if controller.quit_requested:
                break

            controller.step_once = False
            step = engine.step(float(t_s))
            sol = step.get("sol")
            integrity = step.get("integrity")
            conops = step.get("conops")
            attack_report = step.get("attack_report")

            fix_flags = getattr(sol, "fix_flags", None)
            dop = getattr(sol, "dop", None)
            residuals = getattr(sol, "residuals", None)

            fix_valid = bool(getattr(fix_flags, "valid", False))
            fix_type = getattr(fix_flags, "fix_type", "NO FIX")
            sats_used = int(getattr(fix_flags, "sv_count", 0) or 0)
            pdop = float(getattr(dop, "pdop", np.nan))
            residual_rms_m = float(getattr(residuals, "rms_m", np.nan))
            clk_bias_s = float(getattr(sol, "clk_bias_s", np.nan))
            clk_drift_sps = float(getattr(sol, "clk_drift_sps", np.nan))
            nis = float(getattr(integrity, "nis", np.nan))
            nis_alarm = bool(
                getattr(integrity, "is_suspect", False) or getattr(integrity, "is_invalid", False)
            )
            integrity_p_value = float(getattr(integrity, "p_value", np.nan))
            integrity_excluded_sv_ids_count = len(getattr(integrity, "excluded_sv_ids", []))

            applied_count = getattr(attack_report, "applied_count", 0)
            attack_active = applied_count > 0
            if applied_count:
                attack_pr_bias_mean_m = attack_report.pr_bias_sum_m / applied_count
                attack_prr_bias_mean_mps = attack_report.prr_bias_sum_mps / applied_count
            else:
                attack_pr_bias_mean_m = 0.0
                attack_prr_bias_mean_mps = 0.0

            conops_status = getattr(getattr(conops, "status", None), "value", "-")
            conops_mode5 = getattr(getattr(conops, "mode5", None), "value", "-")

            print(
                f"t={t_s:6.1f}s fix_type={fix_type:6s} sats_used={sats_used:2d} pdop={pdop:6.2f} "
                f"residual_rms_m={residual_rms_m:7.3f} nis_alarm={int(nis_alarm)} "
                f"conops_status={conops_status} conops_mode5={conops_mode5} "
                f"attack_active={int(attack_active)}"
            )

            if jsonl_fp is not None:
                json_record = {
                    "t_s": _json_safe_number(float(t_s)),
                    "fix_valid": fix_valid,
                    "fix_type": fix_type,
                    "sats_used": sats_used,
                    "pdop": _json_safe_number(pdop),
                    "residual_rms_m": _json_safe_number(residual_rms_m),
                    "clk_bias_s": _json_safe_number(clk_bias_s),
                    "clk_drift_sps": _json_safe_number(clk_drift_sps),
                    "nis": _json_safe_number(nis),
                    "nis_alarm": nis_alarm,
                    "conops_status": conops_status,
                    "conops_mode5": conops_mode5,
                    "integrity_p_value": _json_safe_number(integrity_p_value),
                    "integrity_excluded_sv_ids_count": integrity_excluded_sv_ids_count,
                    "attack_name": cfg.attack_name,
                    "attack_active": attack_active,
                    "attack_pr_bias_mean_m": _json_safe_number(float(attack_pr_bias_mean_m)),
                    "attack_prr_bias_mean_mps": _json_safe_number(float(attack_prr_bias_mean_mps)),
                }
                jsonl_fp.write(json.dumps(json_record) + "\n")
                jsonl_fp.flush()

            if plotter is not None:
                plotter.update(
                    float(t_s),
                    float(residual_rms_m),
                    float(pdop),
                    float(clk_bias_s),
                    float(attack_pr_bias_mean_m if attack_active else 0.0),
                )

            target_wall_dt = dt / max(args.speed, 1e-9)
            elapsed = time.perf_counter() - last_tick
            if elapsed < target_wall_dt:
                time.sleep(target_wall_dt - elapsed)
            last_tick = time.perf_counter()
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    if plotter is not None:
        plt.ioff()
        plt.close(plotter.fig)


if __name__ == "__main__":
    main()
