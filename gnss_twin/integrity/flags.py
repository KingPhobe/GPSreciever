"""Integrity flags and lightweight fix validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import chi2

from gnss_twin.integrity.raim import compute_raim
from gnss_twin.models import DopMetrics, FixFlags, GnssMeasurement, PvtSolution, ResidualStats, SvState
from gnss_twin.receiver.wls_pvt import WlsPvtResult, wls_pvt


@dataclass(frozen=True)
class IntegrityConfig:
    """Configuration for fix validity checks and tracking placeholders."""

    elevation_mask_deg: float = 10.0
    pdop_max: float = 6.0
    gdop_max: float = 8.0
    vdop_max: float = 10.0
    chi_square_alpha: float = 0.01
    max_residual_m: float = 12.0
    max_fde_iterations: int = 2
    cn0_track_dbhz: float = 30.0
    cn0_continuity_dbhz: float = 28.0
    tracking_gap_s: float = 2.0


@dataclass
class _TrackingState:
    last_seen_t: float
    tracked: bool


class SvTracker:
    """Stateful CN0/continuity tracker for per-satellite metadata."""

    def __init__(self, config: IntegrityConfig | None = None) -> None:
        self._config = config or IntegrityConfig()
        self._state: dict[str, _TrackingState] = {}

    def update(self, measurements: list[GnssMeasurement]) -> dict[str, dict[str, float]]:
        """Update tracking state and return per-SV tracking metadata."""

        info: dict[str, dict[str, float]] = {}
        cfg = self._config
        for meas in measurements:
            prev = self._state.get(meas.sv_id)
            seen_recent = False
            if prev is not None:
                seen_recent = (meas.t - prev.last_seen_t) <= cfg.tracking_gap_s
            continuity = bool(prev and prev.tracked and seen_recent)
            tracked = bool(
                meas.cn0_dbhz >= cfg.cn0_track_dbhz
                or (continuity and meas.cn0_dbhz >= cfg.cn0_continuity_dbhz)
            )
            self._state[meas.sv_id] = _TrackingState(last_seen_t=meas.t, tracked=tracked)
            info[meas.sv_id] = {
                "cn0_dbhz": float(meas.cn0_dbhz),
                "tracked": 1.0 if tracked else 0.0,
                "continuity": 1.0 if continuity else 0.0,
            }
        return info


def integrity_pvt(
    measurements: list[GnssMeasurement],
    sv_states: list[SvState],
    *,
    initial_pos_ecef_m: np.ndarray | None = None,
    initial_clk_bias_s: float = 0.0,
    config: IntegrityConfig | None = None,
    tracker: SvTracker | None = None,
) -> tuple[PvtSolution, Mapping[str, Mapping[str, float]]]:
    """Solve for PVT and attach integrity flags and per-SV metadata."""

    cfg = config or IntegrityConfig()
    tracker = tracker or SvTracker(cfg)
    tracking_info = tracker.update(measurements)
    per_sv_stats: dict[str, dict[str, float]] = {
        sv_id: {
            **info,
            "used": 0.0,
            "rejected": 0.0,
            "residual_m": float("nan"),
        }
        for sv_id, info in tracking_info.items()
    }

    masked = [meas for meas in measurements if meas.elev_deg >= cfg.elevation_mask_deg]
    mask_ok = len(masked) >= 4
    if len(masked) < 4:
        return _no_fix_solution(len(measurements), mask_ok, "insufficient_masked_sv"), per_sv_stats

    used = masked
    rejected: list[str] = []
    solution: WlsPvtResult | None = None
    residuals_by_sv: dict[str, float] = {}
    raim_stat = float("nan")
    raim_dof = 0
    raim_threshold = float("inf")
    raim_passed = False
    for iteration in range(cfg.max_fde_iterations + 1):
        solution = wls_pvt(
            used,
            sv_states,
            initial_pos_ecef_m=initial_pos_ecef_m,
            initial_clk_bias_s=initial_clk_bias_s,
        )
        if solution is None:
            return _no_fix_solution(len(measurements), mask_ok, "solver_failed"), per_sv_stats
        residuals_by_sv = solution.residuals_m
        if not residuals_by_sv:
            raim_passed = True
            break
        sigmas_by_sv = {meas.sv_id: meas.sigma_pr_m for meas in used}
        raim_stat, raim_dof, raim_threshold, raim_passed = compute_raim(
            residuals_by_sv,
            sigmas_by_sv,
            num_states=4,
            alpha=cfg.chi_square_alpha,
        )
        if raim_passed or iteration >= cfg.max_fde_iterations:
            break
        worst_sv, _ = max(
            residuals_by_sv.items(),
            key=lambda item: abs(item[1])
            / max(float(sigmas_by_sv.get(item[0], 0.0)), 1e-3),
        )
        rejected.append(worst_sv)
        used = [meas for meas in used if meas.sv_id != worst_sv]
        if len(used) < 4:
            return _no_fix_solution(len(measurements), mask_ok, "insufficient_sv_after_fde"), per_sv_stats

    residual_stats = _compute_residual_stats(used, residuals_by_sv)
    chi_square_threshold = raim_threshold if np.isfinite(raim_threshold) else _chi_square_threshold(
        len(used),
        cfg.chi_square_alpha,
    )
    chi_square_ok = raim_passed
    residual_ok = bool(np.isfinite(residual_stats.max_m) and residual_stats.max_m <= cfg.max_residual_m)
    dop = solution.dop
    dop_ok = (
        np.isfinite(dop.pdop)
        and np.isfinite(dop.gdop)
        and dop.pdop <= cfg.pdop_max
        and dop.gdop <= cfg.gdop_max
    )
    valid = mask_ok and dop_ok and chi_square_ok and residual_ok
    validity_reason = "ok"
    if not mask_ok:
        validity_reason = "insufficient_masked_sv"
    elif not dop_ok:
        validity_reason = "dop_limit"
    elif not chi_square_ok:
        validity_reason = "raim_failed"
    elif not residual_ok:
        validity_reason = "max_residual_exceeded"
    fix_type = _fix_type(len(used), dop, cfg)

    for meas in used:
        sv_id = meas.sv_id
        per_sv_stats.setdefault(sv_id, {
            "cn0_dbhz": float(meas.cn0_dbhz),
            "tracked": 0.0,
            "continuity": 0.0,
        })
        per_sv_stats[sv_id]["used"] = 1.0
        per_sv_stats[sv_id]["residual_m"] = float(residuals_by_sv.get(sv_id, float("nan")))
    for sv_id in rejected:
        per_sv_stats.setdefault(sv_id, {
            "cn0_dbhz": float("nan"),
            "tracked": 0.0,
            "continuity": 0.0,
        })
        per_sv_stats[sv_id]["rejected"] = 1.0

    fix_flags = FixFlags(
        fix_type=fix_type,
        valid=valid,
        sv_used=[meas.sv_id for meas in used],
        sv_rejected=rejected,
        sv_count=len(used),
        sv_in_view=len(measurements),
        mask_ok=mask_ok,
        pdop=dop.pdop,
        gdop=dop.gdop,
        chi_square=residual_stats.chi_square,
        chi_square_threshold=chi_square_threshold,
        raim_passed=raim_passed,
        validity_reason=validity_reason,
    )
    return _solution_from_wls(solution, residual_stats, fix_flags), per_sv_stats


def _solution_from_wls(
    solution: WlsPvtResult,
    residuals: ResidualStats,
    fix_flags: FixFlags,
) -> PvtSolution:
    return PvtSolution(
        pos_ecef=solution.pos_ecef_m,
        vel_ecef=solution.vel_ecef_mps,
        clk_bias_s=solution.clk_bias_s,
        clk_drift_sps=solution.clk_drift_sps or 0.0,
        dop=solution.dop,
        residuals=residuals,
        fix_flags=fix_flags,
    )


def _no_fix_solution(sv_in_view: int, mask_ok: bool, validity_reason: str) -> PvtSolution:
    nan_vec = np.full(3, np.nan)
    dop = DopMetrics(gdop=float("nan"), pdop=float("nan"), hdop=float("nan"), vdop=float("nan"))
    residuals = ResidualStats(rms_m=float("nan"), mean_m=float("nan"), max_m=float("nan"), chi_square=float("nan"))
    fix_flags = FixFlags(
        fix_type="NO FIX",
        valid=False,
        sv_used=[],
        sv_rejected=[],
        sv_count=0,
        sv_in_view=sv_in_view,
        mask_ok=mask_ok,
        pdop=float("nan"),
        gdop=float("nan"),
        chi_square=float("nan"),
        chi_square_threshold=float("inf"),
        raim_passed=False,
        validity_reason=validity_reason,
    )
    return PvtSolution(
        pos_ecef=nan_vec,
        vel_ecef=None,
        clk_bias_s=float("nan"),
        clk_drift_sps=float("nan"),
        dop=dop,
        residuals=residuals,
        fix_flags=fix_flags,
    )


def _compute_residual_stats(
    used: list[GnssMeasurement],
    residuals_by_sv: Mapping[str, float],
) -> ResidualStats:
    residuals = np.array([float(residuals_by_sv[meas.sv_id]) for meas in used], dtype=float)
    rms = float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else float("nan")
    mean = float(np.mean(residuals)) if residuals.size else float("nan")
    max_m = float(np.max(np.abs(residuals))) if residuals.size else float("nan")
    chi_square = _chi_square(residuals_by_sv, used)
    return ResidualStats(rms_m=rms, mean_m=mean, max_m=max_m, chi_square=chi_square)


def _chi_square(residuals_by_sv: Mapping[str, float], used: list[GnssMeasurement]) -> float:
    if not used:
        return float("nan")
    terms = []
    for meas in used:
        sigma = max(float(meas.sigma_pr_m), 1e-3)
        terms.append((float(residuals_by_sv[meas.sv_id]) / sigma) ** 2)
    return float(np.sum(terms))


def _chi_square_threshold(num_used: int, alpha: float) -> float:
    dof = num_used - 4
    if dof <= 0:
        return float("inf")
    return float(chi2.ppf(1.0 - alpha, dof))


def _fix_type(num_used: int, dop: DopMetrics, cfg: IntegrityConfig) -> str:
    if num_used < 4:
        return "NO FIX"
    if not np.isfinite(dop.vdop) or dop.vdop > cfg.vdop_max:
        return "2D"
    return "3D"
