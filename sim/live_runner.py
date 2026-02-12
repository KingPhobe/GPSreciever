"""Interactive live simulation runner for GNSS twin demos."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gnss_twin.config import SimConfig
from sim.run_static_demo import build_engine


def parse_kv_list(raw_params: list[str]) -> dict[str, Any]:
    """Parse repeated key=value items into a dictionary with typed values."""
    params: dict[str, Any] = {}
    for raw_param in raw_params:
        if "=" not in raw_param:
            raise ValueError(f"Invalid --attack-param '{raw_param}'; expected key=value.")
        key, raw_value = raw_param.split("=", 1)
        if not key:
            raise ValueError("Attack parameter key cannot be empty.")
        params[key] = _parse_kv_value(raw_value)
    return params


def _parse_kv_value(raw_value: str) -> Any:
    value_lower = raw_value.lower()
    if value_lower in {"true", "false"}:
        return value_lower == "true"
    if raw_value.isdigit() or (
        len(raw_value) > 1 and raw_value[0] in {"+", "-"} and raw_value[1:].isdigit()
    ):
        return int(raw_value)
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


class _LiveController:
    def __init__(self) -> None:
        self.paused = False
        self.quit_requested = False
        self.single_step_requested = False

    def on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == " ":
            self.paused = not self.paused
            self.single_step_requested = False
        elif key == "right" and self.paused:
            self.single_step_requested = True
        elif key in {"q", "Q", "escape", "esc"}:
            self.quit_requested = True


class _LivePlotter:
    def __init__(self, max_points: int = 300) -> None:
        self.max_points = max_points
        self.t_s: list[float] = []
        self.residual_rms_m: list[float] = []
        self.pdop: list[float] = []
        self.clk_bias_s: list[float] = []
        self.attack_pr_bias_mean_m: list[float] = []

        plt.ion()
        self.fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
        self.axes = axes
        (self.residual_line,) = axes[0].plot([], [], lw=1.8)
        (self.pdop_line,) = axes[1].plot([], [], lw=1.8)
        (self.clk_bias_line,) = axes[2].plot([], [], lw=1.8)
        (self.attack_line,) = axes[3].plot([], [], lw=1.8)
        axes[0].set_ylabel("residual\nrms [m]")
        axes[1].set_ylabel("pdop")
        axes[2].set_ylabel("clk bias [s]")
        axes[3].set_ylabel("attack pr\nbias [m]")
        axes[3].set_xlabel("time [s]")
        self.fig.suptitle("Live GNSS Twin (space pause, right step, q/esc quit)")

    def _append(self, t_s: float, residual_rms_m: float, pdop: float, clk_bias_s: float, attack_pr_bias: float) -> None:
        self.t_s.append(t_s)
        self.residual_rms_m.append(residual_rms_m)
        self.pdop.append(pdop)
        self.clk_bias_s.append(clk_bias_s)
        self.attack_pr_bias_mean_m.append(attack_pr_bias)

        if len(self.t_s) > self.max_points:
            self.t_s = self.t_s[-self.max_points :]
            self.residual_rms_m = self.residual_rms_m[-self.max_points :]
            self.pdop = self.pdop[-self.max_points :]
            self.clk_bias_s = self.clk_bias_s[-self.max_points :]
            self.attack_pr_bias_mean_m = self.attack_pr_bias_mean_m[-self.max_points :]

    def update(self, t_s: float, residual_rms_m: float, pdop: float, clk_bias_s: float, attack_pr_bias: float) -> None:
        self._append(t_s, residual_rms_m, pdop, clk_bias_s, attack_pr_bias)
        self.residual_line.set_data(self.t_s, self.residual_rms_m)
        self.pdop_line.set_data(self.t_s, self.pdop)
        self.clk_bias_line.set_data(self.t_s, self.clk_bias_s)
        self.attack_line.set_data(self.t_s, self.attack_pr_bias_mean_m)

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return value


def _as_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interactive live GNSS twin simulation.")
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--use-ekf", action="store_true")
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--attack-name", type=str, default="none")
    parser.add_argument("--attack-param", action="append", default=[])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--out-jsonl", type=str, default=None)
    args = parser.parse_args()

    cfg = SimConfig(
        duration=args.duration_s,
        dt=args.dt if args.dt is not None else 1.0,
        use_ekf=args.use_ekf,
        rng_seed=args.rng_seed,
        attack_name=args.attack_name,
        attack_params=parse_kv_list(args.attack_param),
    )
    dt = args.dt if args.dt is not None else float(getattr(cfg, "dt_s", getattr(cfg, "dt", 1.0)))

    engine = build_engine(cfg)
    controller = _LiveController()
    plotter: _LivePlotter | None = None
    if not args.no_plots:
        plotter = _LivePlotter(max_points=300)
        plotter.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    out_fp = None
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fp = out_path.open("w", encoding="utf-8")

    sim_times = np.arange(0.0, args.duration_s + 1e-9, dt)
    target_wall_dt = dt / max(args.speed, 1e-9)

    try:
        for t_s in sim_times:
            epoch_start = time.perf_counter()
            if controller.quit_requested:
                break

            while controller.paused and not controller.single_step_requested and not controller.quit_requested:
                if plotter is not None:
                    plt.pause(0.05)
                else:
                    time.sleep(0.05)

            if controller.quit_requested:
                break

            controller.single_step_requested = False
            step = engine.step(float(t_s))
            sol = step.get("sol")
            integrity = step.get("integrity")
            conops = step.get("conops")
            attack_report = step.get("attack_report")

            fix_flags = getattr(sol, "fix_flags", None)
            dop = getattr(sol, "dop", None)
            residuals = getattr(sol, "residuals", None)

            fix_valid = getattr(fix_flags, "valid", None)
            fix_type = getattr(fix_flags, "fix_type", None)
            sats_used = _as_int(getattr(fix_flags, "sv_count", None))
            pdop = _as_float(getattr(dop, "pdop", None))
            residual_rms_m = _as_float(getattr(residuals, "rms_m", None))
            clk_bias_s = _as_float(getattr(sol, "clk_bias_s", None))
            clk_drift_sps = _as_float(getattr(sol, "clk_drift_sps", None))

            nis = _as_float(getattr(integrity, "nis", None))
            is_suspect = bool(getattr(integrity, "is_suspect", False))
            is_invalid = bool(getattr(integrity, "is_invalid", False))
            nis_alarm = is_suspect or is_invalid
            integrity_p_value = _as_float(getattr(integrity, "p_value", None))
            excluded_sv_ids = getattr(integrity, "excluded_sv_ids", []) or []
            integrity_excluded_sv_ids_count = len(excluded_sv_ids)

            applied_count = int(getattr(attack_report, "applied_count", 0) or 0)
            attack_active = applied_count > 0
            if attack_active:
                attack_pr_bias_mean_m = _as_float(getattr(attack_report, "pr_bias_sum_m", 0.0)) / applied_count
                attack_prr_bias_mean_mps = _as_float(getattr(attack_report, "prr_bias_sum_mps", 0.0)) / applied_count
            else:
                attack_pr_bias_mean_m = 0.0
                attack_prr_bias_mean_mps = 0.0

            conops_status = getattr(getattr(conops, "status", None), "value", None)
            conops_mode5 = getattr(getattr(conops, "mode5", None), "value", None)

            print(
                " ".join(
                    [
                        f"t_s={t_s:.1f}",
                        f"fix_type={fix_type}",
                        f"sats_used={sats_used}",
                        f"pdop={pdop:.3f}" if np.isfinite(pdop) else "pdop=None",
                        f"residual_rms_m={residual_rms_m:.3f}" if np.isfinite(residual_rms_m) else "residual_rms_m=None",
                        f"nis_alarm={int(nis_alarm)}",
                        f"conops_status={conops_status}",
                        f"conops_mode5={conops_mode5}",
                        f"attack_active={int(attack_active)}",
                    ]
                )
            )

            if out_fp is not None:
                record = {
                    "t_s": _json_safe(float(t_s)),
                    "fix_valid": _json_safe(fix_valid),
                    "fix_type": _json_safe(fix_type),
                    "sats_used": _json_safe(sats_used),
                    "pdop": _json_safe(pdop),
                    "residual_rms_m": _json_safe(residual_rms_m),
                    "clk_bias_s": _json_safe(clk_bias_s),
                    "clk_drift_sps": _json_safe(clk_drift_sps),
                    "nis": _json_safe(nis),
                    "nis_alarm": _json_safe(nis_alarm),
                    "conops_status": _json_safe(conops_status),
                    "conops_mode5": _json_safe(conops_mode5),
                    "integrity_p_value": _json_safe(integrity_p_value),
                    "integrity_excluded_sv_ids_count": _json_safe(integrity_excluded_sv_ids_count),
                    "attack_name": _json_safe(args.attack_name),
                    "attack_active": _json_safe(attack_active),
                    "attack_pr_bias_mean_m": _json_safe(attack_pr_bias_mean_m),
                    "attack_prr_bias_mean_mps": _json_safe(attack_prr_bias_mean_mps),
                }
                out_fp.write(json.dumps(record) + "\n")
                out_fp.flush()

            if plotter is not None:
                plotter.update(
                    t_s=float(t_s),
                    residual_rms_m=float(residual_rms_m),
                    pdop=float(pdop),
                    clk_bias_s=float(clk_bias_s),
                    attack_pr_bias=float(attack_pr_bias_mean_m if attack_active else 0.0),
                )

            elapsed = time.perf_counter() - epoch_start
            if elapsed < target_wall_dt:
                time.sleep(target_wall_dt - elapsed)
    finally:
        if out_fp is not None:
            out_fp.close()
        if plotter is not None:
            plt.ioff()
            plt.close(plotter.fig)


if __name__ == "__main__":
    main()
