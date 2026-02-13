"""Plotting utilities for GNSS twin run outputs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

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

    frame = epochs_to_frame(epochs)
    return plot_update(frame, out_dir=out_dir, run_name=run_name)


def plot_update(
    data: Any,
    *,
    out_dir: str | Path = "out",
    run_name: str | None = None,
) -> Path:
    """Save standard run plots using a GUI-friendly DataFrame input."""

    output_dir = _prepare_output_dir(out_dir, run_name)
    plot_position_error(data, output_dir / "position_error.png")
    plot_clock_bias(data, output_dir / "clock_bias.png")
    plot_residual_rms(data, output_dir / "residual_rms.png")
    plot_dop(data, output_dir / "dop.png")
    plot_sv_used(data, output_dir / "satellites_used.png")
    plot_fix_status(data, output_dir / "fix_status.png")
    plot_attack_telemetry(data, output_dir / "attack_telemetry.png")
    return output_dir


def plot_position_error(data: Any, path: str | Path) -> None:
    times = _series_or_nan(data, "t_s")
    if "pos_error_m" in data.columns:
        errors = data["pos_error_m"].to_numpy(dtype=float)
    else:
        errors = np.full(len(times), float("nan"))
    _plot_position_error(times, errors, Path(path))


def plot_clock_bias(data: Any, path: str | Path) -> None:
    times = _series_or_nan(data, "t_s")
    clk_bias = _series_or_nan(data, "clk_bias_s")
    _plot_clock_bias(times, clk_bias, Path(path))


def plot_residual_rms(data: Any, path: str | Path) -> None:
    times = _series_or_nan(data, "t_s")
    residual_rms = _series_or_nan(data, "residual_rms_m")
    _plot_residual_rms(times, residual_rms, Path(path))


def plot_dop(data: Any, path: str | Path) -> None:
    times = _series_or_nan(data, "t_s")
    dop = np.vstack(
        [
            _series_or_nan(data, "gdop"),
            _series_or_nan(data, "pdop"),
            _series_or_nan(data, "hdop"),
            _series_or_nan(data, "vdop"),
        ]
    ).T
    _plot_dop(times, dop, Path(path))


def plot_sv_used(data: Any, path: str | Path) -> None:
    times = _series_or_nan(data, "t_s")
    sv_used = _series_or_nan(data, "sats_used")
    _plot_sv_used(times, sv_used, Path(path))


def plot_fix_status(data: Any, path: str | Path) -> None:
    times = _series_or_nan(data, "t_s")
    fix_type = _series_or_nan(data, "fix_type")
    fix_valid = _series_or_nan(data, "fix_valid")
    _plot_fix_status(times, fix_type, fix_valid, Path(path))




def plot_attack_telemetry(data: Any, path: str | Path) -> None:
    times = _series_or_nan(data, "t_s")
    attack_active = _series_or_nan(data, "attack_active")
    pr_bias = _series_or_nan(data, "attack_pr_bias_mean_m")
    prr_bias = _series_or_nan(data, "attack_prr_bias_mean_mps")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(times, attack_active, where="post", label="Attack active (0/1)", color="tab:red")
    ax.plot(times, pr_bias, label="PR bias mean (m)", color="tab:blue")
    ax.plot(times, prr_bias, label="PRR bias mean (m/s)", color="tab:green")
    ax.set_title("Attack Telemetry vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Telemetry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def epochs_to_frame(epochs: list[EpochLog]) -> Any:
    """Build a DataFrame from EpochLog entries for GUI plotting."""

    import pandas as pd

    payload = []
    for epoch in epochs:
        solution = epoch.solution
        pos_error = _position_error(epoch)
        dop = _dop_vector(epoch)
        payload.append(
            {
                "t_s": epoch.t_s if epoch.t_s is not None else epoch.t,
                "pos_error_m": pos_error,
                "clk_bias_s": _clock_bias(epoch),
                "residual_rms_m": _residual_rms(epoch),
                "gdop": dop[0],
                "pdop": dop[1],
                "hdop": dop[2],
                "vdop": dop[3],
                "sats_used": _sv_used(epoch),
                "fix_type": _fix_type_value(epoch),
                "fix_valid": _fix_valid_value(epoch),
                "attack_name": epoch.attack_name or "",
                "attack_active": bool(epoch.attack_active),
                "attack_pr_bias_mean_m": float(epoch.attack_pr_bias_mean_m),
                "attack_prr_bias_mean_mps": float(epoch.attack_prr_bias_mean_mps),
                "pos_ecef_x": solution.pos_ecef[0] if solution is not None else float("nan"),
                "pos_ecef_y": solution.pos_ecef[1] if solution is not None else float("nan"),
                "pos_ecef_z": solution.pos_ecef[2] if solution is not None else float("nan"),
                "vel_ecef_x": (
                    solution.vel_ecef[0]
                    if solution is not None and solution.vel_ecef is not None
                    else float("nan")
                ),
                "vel_ecef_y": (
                    solution.vel_ecef[1]
                    if solution is not None and solution.vel_ecef is not None
                    else float("nan")
                ),
                "vel_ecef_z": (
                    solution.vel_ecef[2]
                    if solution is not None and solution.vel_ecef is not None
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(payload)


def _series_or_nan(data: Any, column: str) -> np.ndarray:
    if column in data.columns:
        return data[column].to_numpy(dtype=float)
    return np.full(len(data), float("nan"))


def _prepare_output_dir(out_dir: str | Path, run_name: str | None) -> Path:
    root = Path(out_dir)
    output_dir = root if run_name is None else root / run_name
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
