"""RAIM-style integrity checks and FDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import chi2

from gnss_twin.integrity.report import IntegrityReport
from gnss_twin.meas.models import Measurement
from gnss_twin.receiver.solver import Solution, wls_solve


@dataclass(frozen=True)
class RaimFdeReport:
    """Integrity assessment output for RAIM FDE."""

    valid: bool
    max_residual_m: float
    threshold_m: float
    excluded_prn: int | None
    fde_used: bool


def chi2_threshold(dof: int, p: float) -> float:
    """Return chi-square threshold for the given probability and dof."""

    if dof <= 0:
        return float("inf")
    return float(chi2.ppf(p, dof))


def compute_raim(
    residuals_by_sv: Mapping[str, float],
    sigmas_by_sv: Mapping[str, float],
    num_states: int = 4,
    alpha: float = 0.01,
) -> tuple[float, int, float, bool]:
    """Compute a global RAIM consistency test statistic."""

    terms: list[float] = []
    for sv_id, residual in residuals_by_sv.items():
        sigma = max(float(sigmas_by_sv.get(sv_id, 0.0)), 1e-3)
        terms.append((float(residual) / sigma) ** 2)
    t_stat = float(np.sum(terms)) if terms else float("nan")
    dof = len(terms) - num_states
    threshold = chi2_threshold(dof, 1.0 - alpha)
    pass_bool = bool(dof > 0 and np.isfinite(t_stat) and t_stat <= threshold)
    return t_stat, dof, threshold, pass_bool


def raim_fde(
    measurements: list[Measurement],
    initial_position_ecef_m: np.ndarray,
    threshold_m: float = 10.0,
    return_report: bool = False,
) -> tuple[Solution, RaimFdeReport | IntegrityReport]:
    """Perform residual-based RAIM with a single-satellite exclusion."""

    solution = wls_solve(measurements, initial_position_ecef_m)
    residuals = solution.residuals_m
    max_residual = float(np.max(np.abs(residuals)))
    if max_residual <= threshold_m or len(measurements) < 5:
        report = RaimFdeReport(
            valid=max_residual <= threshold_m,
            max_residual_m=max_residual,
            threshold_m=threshold_m,
            excluded_prn=None,
            fde_used=False,
        )
        if return_report:
            return solution, _to_integrity_report(report, residuals, len(measurements))
        return solution, report

    idx = int(np.argmax(np.abs(residuals)))
    excluded = measurements[idx]
    filtered = [m for i, m in enumerate(measurements) if i != idx]
    if len(filtered) < 4:
        report = RaimFdeReport(
            valid=False,
            max_residual_m=max_residual,
            threshold_m=threshold_m,
            excluded_prn=excluded.prn,
            fde_used=True,
        )
        if return_report:
            return solution, _to_integrity_report(report, residuals, len(filtered))
        return solution, report
    fde_solution = wls_solve(filtered, initial_position_ecef_m)
    fde_max_residual = float(np.max(np.abs(fde_solution.residuals_m)))
    report = RaimFdeReport(
        valid=fde_max_residual <= threshold_m,
        max_residual_m=fde_max_residual,
        threshold_m=threshold_m,
        excluded_prn=excluded.prn,
        fde_used=True,
    )
    if return_report:
        return fde_solution, _to_integrity_report(
            report,
            fde_solution.residuals_m,
            len(filtered),
        )
    return fde_solution, report


def _to_integrity_report(
    report: RaimFdeReport,
    residuals: Sequence[float],
    num_sats_used: int,
) -> IntegrityReport:
    residuals_arr = np.array(residuals, dtype=float)
    residual_rms = float(np.sqrt(np.mean(residuals_arr**2))) if residuals_arr.size else None
    excluded_sv_ids = [report.excluded_prn] if report.excluded_prn is not None else []
    num_rejected = len(excluded_sv_ids)
    is_invalid = num_sats_used < 4
    reason_codes: list[str] = []
    if report.fde_used:
        reason_codes.append("fde_exclusion")
    if report.max_residual_m > report.threshold_m:
        reason_codes.append("residual_threshold")
    if is_invalid:
        reason_codes.append("insufficient_sats")
    return IntegrityReport(
        chi2=None,
        p_value=None,
        residual_rms=residual_rms,
        num_sats_used=num_sats_used,
        num_rejected=num_rejected,
        excluded_sv_ids=excluded_sv_ids,
        is_suspect=not report.valid,
        is_invalid=is_invalid,
        reason_codes=reason_codes,
    )
