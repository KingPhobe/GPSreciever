"""Shared metric extraction helpers for validation/verification.

These helpers operate on `epoch_logs.npz` (preferred) because it contains both
solution and truth, allowing truth-based error metrics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from gnss_twin.logger import load_epochs_npz


def metrics_from_epoch_npz(npz_path: Path, *, attack_start_t: float | None = None) -> dict[str, float]:
    """Compute standard run metrics from an epoch_logs.npz file."""
    epochs = load_epochs_npz(npz_path)
    return _metrics_from_epoch_dicts(epochs, attack_start_t=attack_start_t)


def sanitize_json(obj: Any) -> Any:
    """Convert NaN/Inf to None so json.dumps(..., allow_nan=False) succeeds."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj


def _metrics_from_epoch_dicts(epochs: list[dict[str, Any]], *, attack_start_t: float | None) -> dict[str, float]:
    t_s = np.array([_epoch_time_s(e) for e in epochs], dtype=float)

    pos_err = np.array([_pos_error_m(e) for e in epochs], dtype=float)
    clk_err = np.array([_clk_bias_error_s(e) for e in epochs], dtype=float)
    residual = np.array([_residual_rms_m(e) for e in epochs], dtype=float)
    sats_used = np.array([_sats_used(e) for e in epochs], dtype=float)
    raim_pass = np.array([_raim_pass(e) for e in epochs], dtype=float)
    fix_valid = np.array([_fix_valid(e) for e in epochs], dtype=float)
    nis_alarm = np.array([_nis_alarm(e) for e in epochs], dtype=float)
    attack_active = np.array([1.0 if bool(e.get("attack_active")) else 0.0 for e in epochs], dtype=float)

    after_mask = np.isfinite(t_s) & (t_s >= float(attack_start_t)) if attack_start_t is not None else None

    out = {
        "pos_err_rms_m": _rms(pos_err),
        "pos_err_p95_m": _pctl(pos_err, 95),
        "clk_bias_err_rms_s": _rms(clk_err),
        "clk_bias_err_p95_s": _pctl(clk_err, 95),
        "residual_rms_mean_m": _mean(residual),
        "residual_rms_max_m": _max(residual),
        "sats_used_mean": _mean(sats_used),
        "sats_used_min": _min(sats_used),
        "raim_pass_rate": _mean(raim_pass),
        "fix_valid_rate": _mean(fix_valid),
        "nis_alarm_rate": _mean(nis_alarm),
        "attack_active_rate": _mean(attack_active),
    }

    if after_mask is not None and np.any(after_mask):
        out.update(
            {
                "pos_err_rms_after_attack_m": _rms(pos_err[after_mask]),
                "nis_alarm_rate_after_attack": _mean(nis_alarm[after_mask]),
                "raim_pass_rate_after_attack": _mean(raim_pass[after_mask]),
                "fix_valid_rate_after_attack": _mean(fix_valid[after_mask]),
                "attack_active_rate_after_attack": _mean(attack_active[after_mask]),
            }
        )
    else:
        out.update(
            {
                "pos_err_rms_after_attack_m": float("nan"),
                "nis_alarm_rate_after_attack": float("nan"),
                "raim_pass_rate_after_attack": float("nan"),
                "fix_valid_rate_after_attack": float("nan"),
                "attack_active_rate_after_attack": float("nan"),
            }
        )

    return out


def _epoch_time_s(epoch: dict[str, Any]) -> float:
    t_s = epoch.get("t_s", epoch.get("t"))
    try:
        return float(t_s)
    except Exception:
        return float("nan")


def _pos_error_m(epoch: dict[str, Any]) -> float:
    sol = epoch.get("solution")
    truth = epoch.get("truth")
    if not isinstance(sol, dict) or not isinstance(truth, dict):
        return float("nan")
    pos = sol.get("pos_ecef")
    truth_pos = truth.get("pos_ecef_m")
    if pos is None or truth_pos is None:
        return float("nan")
    try:
        p = np.array(pos, dtype=float).reshape(3)
        tp = np.array(truth_pos, dtype=float).reshape(3)
    except Exception:
        return float("nan")
    if not np.isfinite(p).all() or not np.isfinite(tp).all():
        return float("nan")
    return float(np.linalg.norm(p - tp))


def _clk_bias_error_s(epoch: dict[str, Any]) -> float:
    sol = epoch.get("solution")
    truth = epoch.get("truth")
    if not isinstance(sol, dict) or not isinstance(truth, dict):
        return float("nan")
    try:
        return float(sol.get("clk_bias_s")) - float(truth.get("clk_bias_s"))
    except Exception:
        return float("nan")


def _residual_rms_m(epoch: dict[str, Any]) -> float:
    v = epoch.get("residual_rms_m")
    if v is not None:
        try:
            return float(v)
        except Exception:
            return float("nan")
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("residuals"), dict):
        try:
            return float(sol["residuals"].get("rms_m"))
        except Exception:
            return float("nan")
    return float("nan")


def _sats_used(epoch: dict[str, Any]) -> float:
    v = epoch.get("sats_used")
    if v is not None:
        try:
            return float(v)
        except Exception:
            return float("nan")
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("fix_flags"), dict):
        try:
            return float(sol["fix_flags"].get("sv_count"))
        except Exception:
            return float("nan")
    return float("nan")


def _raim_pass(epoch: dict[str, Any]) -> float:
    v = epoch.get("raim_pass")
    if v is not None:
        return 1.0 if bool(v) else 0.0
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("fix_flags"), dict):
        return 1.0 if bool(sol["fix_flags"].get("raim_passed")) else 0.0
    return float("nan")


def _fix_valid(epoch: dict[str, Any]) -> float:
    v = epoch.get("fix_valid")
    if v is not None:
        return 1.0 if bool(v) else 0.0
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("fix_flags"), dict):
        return 1.0 if bool(sol["fix_flags"].get("valid")) else 0.0
    return float("nan")


def _nis_alarm(epoch: dict[str, Any]) -> float:
    v = epoch.get("nis_alarm")
    if v is None:
        return float("nan")
    return 1.0 if bool(v) else 0.0


def _finite(v: np.ndarray) -> np.ndarray:
    return v[np.isfinite(v)]


def _mean(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.mean(v)) if v.size else float("nan")


def _max(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.max(v)) if v.size else float("nan")


def _min(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.min(v)) if v.size else float("nan")


def _rms(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.sqrt(np.mean(v**2))) if v.size else float("nan")


def _pctl(v: np.ndarray, p: float) -> float:
    v = _finite(v)
    return float(np.percentile(v, p)) if v.size else float("nan")
