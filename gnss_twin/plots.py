"""Plotting utilities for GNSS twin run outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

from gnss_twin.models import EpochLog

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def save_run_plots(
    epochs: list[EpochLog],
    *,
    out_dir: str | Path = "out",
    run_name: str | None = None,
) -> Path:
    """Save standard run plots to an output directory."""

    output_dir = _prepare_output_dir(out_dir, run_name)
    times = np.array([epoch.t for epoch in epochs], dtype=float)
    pos_error = np.array([_position_error(epoch) for epoch in epochs], dtype=float)
    clk_bias = np.array([_clock_bias(epoch) for epoch in epochs], dtype=float)
    residual_rms = np.array([_residual_rms(epoch) for epoch in epochs], dtype=float)
    dop = np.array([_dop_vector(epoch) for epoch in epochs], dtype=float)
    sv_used = np.array([_sv_used(epoch) for epoch in epochs], dtype=float)
    fix_type = np.array([_fix_type_value(epoch) for epoch in epochs], dtype=float)
    fix_valid = np.array([_fix_valid_value(epoch) for epoch in epochs], dtype=float)

    _plot_position_error(times, pos_error, output_dir / "position_error.png")
    _plot_clock_bias(times, clk_bias, output_dir / "clock_bias.png")
    _plot_residual_rms(times, residual_rms, output_dir / "residual_rms.png")
    _plot_dop(times, dop, output_dir / "dop.png")
    _plot_sv_used(times, sv_used, output_dir / "satellites_used.png")
    _plot_fix_status(times, fix_type, fix_valid, output_dir / "fix_status.png")
    return output_dir


def _prepare_output_dir(out_dir: str | Path, run_name: str | None) -> Path:
    root = Path(out_dir)
    label = run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = root / label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _position_error(epoch: EpochLog) -> float:
    if epoch.solution is None or epoch.truth is None:
        return float("nan")
    if not np.isfinite(epoch.solution.pos_ecef).all():
        return float("nan")
    return float(np.linalg.norm(epoch.solution.pos_ecef - epoch.truth.pos_ecef_m))


def _clock_bias(epoch: EpochLog) -> float:
    if epoch.solution is None:
        return float("nan")
    return float(epoch.solution.clk_bias_s)


def _residual_rms(epoch: EpochLog) -> float:
    if epoch.solution is None:
        return float("nan")
    return float(epoch.solution.residuals.rms_m)


def _dop_vector(epoch: EpochLog) -> np.ndarray:
    if epoch.solution is None:
        return np.array([float("nan")] * 4)
    dop = epoch.solution.dop
    return np.array([dop.gdop, dop.pdop, dop.hdop, dop.vdop], dtype=float)


def _sv_used(epoch: EpochLog) -> float:
    if epoch.solution is None:
        return float("nan")
    return float(epoch.solution.fix_flags.sv_count)


def _fix_type_value(epoch: EpochLog) -> float:
    if epoch.solution is None:
        return float("nan")
    mapping = {"NO FIX": 0.0, "2D": 1.0, "3D": 2.0}
    return float(mapping.get(epoch.solution.fix_flags.fix_type, 0.0))


def _fix_valid_value(epoch: EpochLog) -> float:
    if epoch.solution is None:
        return float("nan")
    return 1.0 if epoch.solution.fix_flags.valid else 0.0


def _plot_position_error(times: np.ndarray, errors: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, errors, marker="o", markersize=3)
    ax.set_title("Position Error vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_clock_bias(times: np.ndarray, clk_bias: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, clk_bias, marker="o", markersize=3, color="tab:orange")
    ax.set_title("Clock Bias vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Clock bias (s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residual_rms(times: np.ndarray, residuals: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, residuals, marker="o", markersize=3, color="tab:green")
    ax.set_title("Residual RMS vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual RMS (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dop(times: np.ndarray, dop: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["GDOP", "PDOP", "HDOP", "VDOP"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for idx, label in enumerate(labels):
        ax.plot(times, dop[:, idx], marker="o", markersize=3, label=label, color=colors[idx])
    ax.set_title("DOP vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DOP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_sv_used(times: np.ndarray, sv_used: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(times, sv_used, where="post", color="tab:purple")
    ax.set_title("Satellites Used vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Satellites used")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_fix_status(times: np.ndarray, fix_type: np.ndarray, fix_valid: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(times, fix_type, where="post", label="Fix type (0/1/2)", color="tab:gray")
    ax.plot(times, fix_valid, marker="o", markersize=3, label="Valid", color="tab:blue")
    ax.set_title("Fix Type & Validity vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fix type / Valid")
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["NO FIX", "2D", "3D"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
