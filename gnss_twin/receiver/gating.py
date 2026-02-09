"""Measurement gating utilities."""

from __future__ import annotations

from typing import Iterable, Mapping

from gnss_twin.config import SimConfig
from gnss_twin.integrity.report import IntegrityReport
from gnss_twin.models import GnssMeasurement


def prefit_filter(
    measurements: list[GnssMeasurement],
    cfg: SimConfig,
    return_report: bool = False,
) -> tuple[list[GnssMeasurement], list[dict[str, object]]] | tuple[
    list[GnssMeasurement],
    list[dict[str, object]],
    IntegrityReport,
]:
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
    if return_report:
        return kept, rejected_meta, _to_integrity_report(kept, rejected_meta)
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


def _to_integrity_report(
    kept: Iterable[GnssMeasurement],
    rejected_meta: Iterable[dict[str, object]],
) -> IntegrityReport:
    kept_list = list(kept)
    rejected_list = list(rejected_meta)
    excluded_sv_ids: list[int] = []
    reason_codes: list[str] = []
    for meta in rejected_list:
        sv_id = meta.get("sv_id")
        if isinstance(sv_id, str):
            parsed = _parse_sv_id(sv_id)
            if parsed is not None:
                excluded_sv_ids.append(parsed)
        reasons = meta.get("reasons")
        if isinstance(reasons, list):
            reason_codes.extend(str(reason) for reason in reasons)
    num_sats_used = len(kept_list)
    num_rejected = len(rejected_list)
    return IntegrityReport(
        chi2=None,
        p_value=None,
        residual_rms=None,
        num_sats_used=num_sats_used,
        num_rejected=num_rejected,
        excluded_sv_ids=excluded_sv_ids,
        is_suspect=num_rejected > 0,
        is_invalid=num_sats_used == 0,
        reason_codes=sorted(set(reason_codes)),
    )


def _parse_sv_id(sv_id: str) -> int | None:
    digits = "".join(ch for ch in sv_id if ch.isdigit())
    if not digits:
        return None
    return int(digits)
