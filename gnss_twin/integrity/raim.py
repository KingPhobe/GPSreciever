"""RAIM-style integrity checks and FDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import chi2

from gnss_twin.meas.models import Measurement
from gnss_twin.receiver.solver import Solution, wls_solve


@dataclass(frozen=True)
class IntegrityReport:
    """Integrity assessment output."""

    valid: bool
    max_residual_m: float
    threshold_m: float
    excluded_prn: int | None
    fde_used: bool


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
    threshold = float(chi2.ppf(1.0 - alpha, dof)) if dof > 0 else float("inf")
    pass_bool = bool(dof > 0 and np.isfinite(t_stat) and t_stat <= threshold)
    return t_stat, dof, threshold, pass_bool


def raim_fde(
    measurements: list[Measurement],
    initial_position_ecef_m: np.ndarray,
    threshold_m: float = 10.0,
) -> tuple[Solution, IntegrityReport]:
    """Perform residual-based RAIM with a single-satellite exclusion."""

    solution = wls_solve(measurements, initial_position_ecef_m)
    residuals = solution.residuals_m
    max_residual = float(np.max(np.abs(residuals)))
    if max_residual <= threshold_m or len(measurements) < 5:
        report = IntegrityReport(
            valid=max_residual <= threshold_m,
            max_residual_m=max_residual,
            threshold_m=threshold_m,
            excluded_prn=None,
            fde_used=False,
        )
        return solution, report

    idx = int(np.argmax(np.abs(residuals)))
    excluded = measurements[idx]
    filtered = [m for i, m in enumerate(measurements) if i != idx]
    if len(filtered) < 4:
        report = IntegrityReport(
            valid=False,
            max_residual_m=max_residual,
            threshold_m=threshold_m,
            excluded_prn=excluded.prn,
            fde_used=True,
        )
        return solution, report
    fde_solution = wls_solve(filtered, initial_position_ecef_m)
    fde_max_residual = float(np.max(np.abs(fde_solution.residuals_m)))
    report = IntegrityReport(
        valid=fde_max_residual <= threshold_m,
        max_residual_m=fde_max_residual,
        threshold_m=threshold_m,
        excluded_prn=excluded.prn,
        fde_used=True,
    )
    return fde_solution, report
