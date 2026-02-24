"""Simple logging utilities for GNSS twin outputs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from gnss_twin.models import EpochLog, FixType, fix_type_from_label

EPOCH_CSV_COLUMNS = [
    "t",
    "t_s",
    "fix_valid",
    "raim_pass",
    "fix_type",
    "sats_used",
    "pdop",
    "hdop",
    "vdop",
    "residual_rms_m",
    "pos_ecef_x",
    "pos_ecef_y",
    "pos_ecef_z",
    "vel_ecef_x",
    "vel_ecef_y",
    "vel_ecef_z",
    "clk_bias_s",
    "clk_drift_sps",
    "truth_pos_ecef_x",
    "truth_pos_ecef_y",
    "truth_pos_ecef_z",
    "truth_vel_ecef_x",
    "truth_vel_ecef_y",
    "truth_vel_ecef_z",
    "truth_clk_bias_s",
    "truth_clk_drift_sps",
    "pos_error_m",
    "clk_bias_error_s",
    "clk_drift_error_sps",
    "nis",
    "nis_alarm",
    "nis_stat_alarm",
    "integrity_alarm",
    "clock_drift_alarm",
    "composite_alarm",
    "attack_name",
    "gdop",
    "residual_mean_m",
    "residual_max_m",
    "chi_square",
    "innov_dim",
    "attack_active",
    "attack_pr_bias_mean_m",
    "attack_prr_bias_mean_mps",
    "conops_status",
    "conops_mode5",
    "conops_reason_codes",
    "integrity_p_value",
    "integrity_residual_rms",
    "integrity_num_sats_used",
    "integrity_excluded_sv_ids_count",
    "pps_ref_edge_s",
    "pps_platform_edge_s",
    "pps_auth_edge_s",
    "pps_platform_minus_ref_s",
    "pps_auth_minus_ref_s",
    "pps_platform_minus_auth_s",
    "auth_bit",
    "auth_locked",
    "auth_mode",
    "auth_sigma_t_s",
    "auth_reason_codes",
    "pps_err_s",
    "holdover_ok",
    "time_since_ground_pps_s",
    "mode5_auth_bit",
]
_CSV_HEADER = ",".join(EPOCH_CSV_COLUMNS) + "\n"


def append_epoch_csv(path: str | Path, epoch: EpochLog) -> None:
    """Append a single epoch summary to a CSV file."""

    target = Path(path)
    line = _epoch_to_csv_line(epoch)
    if not target.exists():
        target.write_text(_CSV_HEADER)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)


def save_epochs_csv(path: str | Path, epochs: list[EpochLog]) -> None:
    """Save all epoch summaries to a CSV file."""

    target = Path(path)
    target.write_text(_CSV_HEADER)
    with target.open("a", encoding="utf-8") as handle:
        for epoch in epochs:
            handle.write(_epoch_to_csv_line(epoch))


def save_epochs_npz(path: str | Path, epochs: list[EpochLog]) -> None:
    """Save epoch logs to a compressed NPZ file."""

    payload = [asdict(epoch) for epoch in epochs]
    np.savez_compressed(path, epochs=np.array(payload, dtype=object))


def load_epochs_npz(path: str | Path) -> list[dict]:
    """Load epoch logs from a compressed NPZ file."""

    data = np.load(path, allow_pickle=True)
    epochs = data["epochs"].tolist()
    return list(epochs)


def _epoch_to_csv_line(epoch: EpochLog) -> str:
    solution = epoch.solution
    truth = epoch.truth
    t_s = epoch.t_s if epoch.t_s is not None else epoch.t
    pos = _resolve_vector(epoch.pos_ecef, solution.pos_ecef if solution is not None else None)
    vel = _resolve_vector(epoch.vel_ecef, solution.vel_ecef if solution is not None else None)
    truth_pos = _resolve_vector(None, truth.pos_ecef_m if truth is not None else None)
    truth_vel = _resolve_vector(None, truth.vel_ecef_mps if truth is not None else None)

    pos_error_m = None
    clk_bias_error_s = None
    clk_drift_error_sps = None
    if solution is not None and truth is not None:
        if np.isfinite(solution.pos_ecef).all() and np.isfinite(truth.pos_ecef_m).all():
            pos_error_m = float(np.linalg.norm(solution.pos_ecef - truth.pos_ecef_m))
        clk_bias_error_s = float(solution.clk_bias_s - truth.clk_bias_s)
        clk_drift_error_sps = float(solution.clk_drift_sps - truth.clk_drift_sps)

    dop = solution.dop if solution is not None else None
    residuals = solution.residuals if solution is not None else None
    flags = solution.fix_flags if solution is not None else None
    fix_valid = epoch.fix_valid if epoch.fix_valid is not None else (flags.valid if flags else None)
    raim_pass = epoch.raim_pass if epoch.raim_pass is not None else (flags.raim_passed if flags else None)
    fix_type = epoch.fix_type if epoch.fix_type is not None else (fix_type_from_label(flags.fix_type) if flags else None)
    pps_ref_edge_s = getattr(epoch, "pps_ref_edge_s", None)
    pps_platform_edge_s = getattr(epoch, "pps_platform_edge_s", None)
    pps_auth_edge_s = getattr(epoch, "pps_auth_edge_s", None)
    pps_platform_minus_ref_s = getattr(epoch, "pps_platform_minus_ref_s", None)
    pps_auth_minus_ref_s = getattr(epoch, "pps_auth_minus_ref_s", None)
    pps_platform_minus_auth_s = getattr(epoch, "pps_platform_minus_auth_s", None)
    auth_bit = getattr(epoch, "auth_bit", None)
    auth_locked = getattr(epoch, "auth_locked", None)
    auth_mode = getattr(epoch, "auth_mode", None)
    auth_sigma_t_s = getattr(epoch, "auth_sigma_t_s", None)
    auth_reason_codes = getattr(epoch, "auth_reason_codes", None) or []
    pps_err_s = getattr(epoch, "pps_err_s", None)
    holdover_ok = getattr(epoch, "holdover_ok", None)
    time_since_ground_pps_s = getattr(epoch, "time_since_ground_pps_s", None)
    mode5_auth_bit = getattr(epoch, "mode5_auth_bit", None)
    nis_stat_alarm = getattr(epoch, "nis_stat_alarm", None)
    integrity_alarm = getattr(epoch, "integrity_alarm", None)
    clock_drift_alarm = getattr(epoch, "clock_drift_alarm", None)
    composite_alarm = getattr(epoch, "composite_alarm", None)
    if composite_alarm is None:
        composite_alarm = getattr(epoch, "nis_alarm", None)
    if isinstance(fix_type, FixType):
        fix_type_value = int(fix_type)
    elif fix_type is None:
        fix_type_value = None
    else:
        fix_type_value = int(fix_type)

    row = [
        epoch.t,
        t_s,
        _format_value(fix_valid),
        _format_value(raim_pass),
        _format_value(fix_type_value),
        _format_value(epoch.sats_used if epoch.sats_used is not None else (flags.sv_count if flags else None)),
        _format_value(epoch.pdop if epoch.pdop is not None else (dop.pdop if dop else None)),
        _format_value(epoch.hdop if epoch.hdop is not None else (dop.hdop if dop else None)),
        _format_value(epoch.vdop if epoch.vdop is not None else (dop.vdop if dop else None)),
        _format_value(
            epoch.residual_rms_m if epoch.residual_rms_m is not None else (residuals.rms_m if residuals else None)
        ),
        _format_value(pos[0]),
        _format_value(pos[1]),
        _format_value(pos[2]),
        _format_value(vel[0]),
        _format_value(vel[1]),
        _format_value(vel[2]),
        _format_value(epoch.clk_bias_s if epoch.clk_bias_s is not None else (solution.clk_bias_s if solution else None)),
        _format_value(
            epoch.clk_drift_sps if epoch.clk_drift_sps is not None else (solution.clk_drift_sps if solution else None)
        ),
        _format_value(truth_pos[0]),
        _format_value(truth_pos[1]),
        _format_value(truth_pos[2]),
        _format_value(truth_vel[0]),
        _format_value(truth_vel[1]),
        _format_value(truth_vel[2]),
        _format_value(truth.clk_bias_s if truth is not None else None),
        _format_value(truth.clk_drift_sps if truth is not None else None),
        _format_value(pos_error_m),
        _format_value(clk_bias_error_s),
        _format_value(clk_drift_error_sps),
        _format_value(epoch.nis),
        _format_value(int(epoch.nis_alarm)),
        _format_value(nis_stat_alarm),
        _format_value(integrity_alarm),
        _format_value(clock_drift_alarm),
        _format_value(composite_alarm),
        epoch.attack_name or "",
        _format_value(dop.gdop if dop else None),
        _format_value(residuals.mean_m if residuals else None),
        _format_value(residuals.max_m if residuals else None),
        _format_value(residuals.chi_square if residuals else None),
        _format_value(epoch.innov_dim),
        _format_value(int(epoch.attack_active)),
        _format_value(epoch.attack_pr_bias_mean_m),
        _format_value(epoch.attack_prr_bias_mean_mps),
        epoch.conops_status or "",
        epoch.conops_mode5 or "",
        "|".join(epoch.conops_reason_codes),
        _format_value(epoch.integrity_p_value),
        _format_value(epoch.integrity_residual_rms),
        _format_value(epoch.integrity_num_sats_used),
        _format_value(epoch.integrity_excluded_sv_ids_count),
        _format_value(pps_ref_edge_s),
        _format_value(pps_platform_edge_s),
        _format_value(pps_auth_edge_s),
        _format_value(pps_platform_minus_ref_s),
        _format_value(pps_auth_minus_ref_s),
        _format_value(pps_platform_minus_auth_s),
        _format_value(auth_bit),
        _format_value(auth_locked),
        auth_mode or "",
        _format_value(auth_sigma_t_s),
        "|".join(auth_reason_codes),
        _format_value(pps_err_s),
        _format_value(holdover_ok),
        _format_value(time_since_ground_pps_s),
        _format_value(mode5_auth_bit),
    ]
    return ",".join(str(value) for value in row) + "\n"


def _resolve_vector(primary: np.ndarray | None, fallback: np.ndarray | None) -> np.ndarray:
    vector = primary if primary is not None else fallback
    if vector is None:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.array(vector, dtype=float)


def _format_value(value: float | int | bool | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)
