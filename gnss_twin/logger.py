"""Simple logging utilities for GNSS twin outputs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from gnss_twin.models import EpochLog


def append_epoch_csv(path: str | Path, epoch: EpochLog) -> None:
    """Append a single epoch summary to a CSV file."""

    target = Path(path)
    header = (
        "t,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,clk_bias_s,clk_drift_sps,"
        "gdop,pdop,hdop,vdop,residual_rms_m,residual_mean_m,residual_max_m,chi_square,"
        "fix_type,valid\n"
    )
    line = _epoch_to_csv_line(epoch)
    if not target.exists():
        target.write_text(header)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)


def save_epochs_npz(path: str | Path, epochs: list[EpochLog]) -> None:
    """Save epoch logs to a compressed NPZ file."""

    payload = [asdict(epoch) for epoch in epochs]
    np.savez_compressed(path, epochs=np.array(payload, dtype=object))


def _epoch_to_csv_line(epoch: EpochLog) -> str:
    solution = epoch.solution
    if solution is None:
        return (
            f"{epoch.t}," + ",".join(["" for _ in range(18)]) + "\n"
        )

    pos = solution.pos_ecef
    vel = solution.vel_ecef if solution.vel_ecef is not None else np.array([np.nan, np.nan, np.nan])
    dop = solution.dop
    residuals = solution.residuals
    flags = solution.fix_flags
    return (
        f"{epoch.t},"
        f"{pos[0]},{pos[1]},{pos[2]},"
        f"{vel[0]},{vel[1]},{vel[2]},"
        f"{solution.clk_bias_s},{solution.clk_drift_sps},"
        f"{dop.gdop},{dop.pdop},{dop.hdop},{dop.vdop},"
        f"{residuals.rms_m},{residuals.mean_m},{residuals.max_m},{residuals.chi_square},"
        f"{flags.fix_type},{int(flags.valid)}\n"
    )
