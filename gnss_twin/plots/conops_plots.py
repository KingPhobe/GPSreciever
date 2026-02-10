"""ConOps/integrity timeline plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from gnss_twin.models import EpochLog

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

_STATUS_TO_VALUE = {"invalid": 0.0, "suspect": 1.0, "valid": 2.0}


def save_conops_plots(epochs: list[EpochLog], output_dir: str | Path) -> None:
    """Save ConOps/integrity timeline plots into an output directory."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    times = np.array([epoch.t_s if epoch.t_s is not None else epoch.t for epoch in epochs], dtype=float)
    statuses = np.array([_status_to_float(epoch.conops_status) for epoch in epochs], dtype=float)
    residual_rms = np.array([_coalesce_residual_rms(epoch) for epoch in epochs], dtype=float)
    p_values = np.array([epoch.integrity_p_value if epoch.integrity_p_value is not None else np.nan for epoch in epochs])

    _plot_status_timeline(times, statuses, out / "conops_status_timeline.png")
    _plot_series(times, residual_rms, out / "integrity_residual_rms_timeline.png", "Integrity Residual RMS vs Time", "Residual RMS (m)", "tab:green")
    if np.isfinite(p_values).any():
        _plot_series(times, p_values, out / "integrity_p_value_timeline.png", "Integrity p-value vs Time", "p-value", "tab:blue")


def _status_to_float(status: str | None) -> float:
    if status is None:
        return np.nan
    return _STATUS_TO_VALUE.get(status.lower(), np.nan)


def _coalesce_residual_rms(epoch: EpochLog) -> float:
    if epoch.integrity_residual_rms is not None:
        return float(epoch.integrity_residual_rms)
    if epoch.residual_rms_m is not None:
        return float(epoch.residual_rms_m)
    if epoch.solution is not None:
        return float(epoch.solution.residuals.rms_m)
    return np.nan


def _plot_status_timeline(times: np.ndarray, statuses: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(times, statuses, where="post", color="tab:red")
    ax.set_title("ConOps PNT Status vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Status")
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["invalid", "suspect", "valid"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_series(
    times: np.ndarray,
    series: np.ndarray,
    path: Path,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, series, marker="o", markersize=3, color=color)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
