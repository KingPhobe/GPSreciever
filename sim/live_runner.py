"""Interactive live simulation runner for GNSS twin demos."""

from __future__ import annotations

import argparse
import random
import time

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
from sim.run_static_demo import _DemoSolver, _parse_attack_params


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interactive live GNSS twin simulation.")
    dt_default = getattr(SimConfig, "dt", 1.0)
    parser.add_argument("--duration-s", type=float, default=60.0, help="Duration in seconds.")
    parser.add_argument("--dt", type=float, default=dt_default, help="Simulation step size in seconds.")
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
    args = parser.parse_args()

    attack_params = _parse_attack_params(args.attack_param, args.attack_name)
    cfg = SimConfig(
        duration=args.duration_s,
        dt=args.dt,
        use_ekf=args.use_ekf,
        attack_name=args.attack_name,
        attack_params=attack_params,
        rng_seed=args.rng_seed,
    )

    engine = _build_engine(cfg)
    controller = _LiveController()
    plotter = None

    if not args.no_plots:
        plotter = _LivePlotter(max_points=300)
        plotter.fig.canvas.mpl_connect("key_press_event", controller.on_key)

    sim_times = np.arange(0.0, cfg.duration + 1e-9, cfg.dt)
    last_tick = time.perf_counter()

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
        sol = step["sol"]
        integrity = step["integrity"]
        conops = step.get("conops")
        attack_report = step.get("attack_report")

        fix_type = sol.fix_flags.fix_type if sol is not None else "NO FIX"
        sats_used = sol.fix_flags.sv_count if sol is not None else 0
        pdop = sol.dop.pdop if sol is not None else float("nan")
        residual_rms_m = sol.residuals.rms_m if sol is not None else float("nan")
        clk_bias_s = sol.clk_bias_s if sol is not None else float("nan")

        applied_count = getattr(attack_report, "applied_count", 0)
        attack_active = applied_count > 0
        if applied_count:
            attack_pr_bias_mean_m = attack_report.pr_bias_sum_m / applied_count
        else:
            attack_pr_bias_mean_m = 0.0

        conops_status = conops.status.value if conops is not None else "-"
        conops_mode5 = conops.mode5.value if conops is not None else "-"

        print(
            f"t={t_s:6.1f}s fix={fix_type:6s} sats={sats_used:2d} pdop={pdop:5.2f} "
            f"resid={residual_rms_m:6.2f}m int=(sus={int(integrity.is_suspect)} inv={int(integrity.is_invalid)}) "
            f"conops={conops_status}/{conops_mode5} attack={int(attack_active)}"
        )

        if plotter is not None:
            plotter.update(
                float(t_s),
                float(residual_rms_m),
                float(pdop),
                float(clk_bias_s),
                float(attack_pr_bias_mean_m if attack_active else 0.0),
            )

        target_wall_dt = cfg.dt / max(args.speed, 1e-6)
        elapsed = time.perf_counter() - last_tick
        if elapsed < target_wall_dt:
            time.sleep(target_wall_dt - elapsed)
        last_tick = time.perf_counter()

    if plotter is not None:
        plt.ioff()
        plt.close(plotter.fig)


if __name__ == "__main__":
    main()
