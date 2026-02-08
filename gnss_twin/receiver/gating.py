"""Measurement gating utilities."""

from __future__ import annotations

from typing import Mapping

from gnss_twin.config import SimConfig
from gnss_twin.models import GnssMeasurement


def prefit_filter(
    measurements: list[GnssMeasurement],
    cfg: SimConfig,
) -> tuple[list[GnssMeasurement], list[dict[str, object]]]:
    """Reject measurements that fail CN0 or sigma thresholds."""

    kept: list[GnssMeasurement] = []
    rejected_meta: list[dict[str, object]] = []
    for meas in measurements:
        reasons: list[str] = []
        if meas.cn0_dbhz < cfg.cn0_min_dbhz:
            reasons.append("cn0")
        if meas.sigma_pr_m > cfg.sigma_pr_max_m:
            reasons.append("sigma_pr")
        if reasons:
            rejected_meta.append(
                {
                    "sv_id": meas.sv_id,
                    "reasons": reasons,
                    "cn0_dbhz": float(meas.cn0_dbhz),
                    "sigma_pr_m": float(meas.sigma_pr_m),
                }
            )
        else:
            kept.append(meas)
    return kept, rejected_meta


def postfit_gate(
    residuals_by_sv: Mapping[str, float],
    sigmas_by_sv: Mapping[str, float],
    gate: float = 4.0,
) -> str | None:
    """Return the SV with the worst standardized residual above the gate."""

    worst_sv: str | None = None
    worst_z = 0.0
    for sv_id, residual in residuals_by_sv.items():
        sigma = max(float(sigmas_by_sv.get(sv_id, 0.0)), 1e-3)
        z = abs(float(residual)) / sigma
        if z > worst_z:
            worst_z = z
            worst_sv = sv_id
    if worst_sv is None or worst_z <= gate:
        return None
    return worst_sv
