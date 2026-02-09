"""Unified integrity report structure."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntegrityReport:
    """Unified integrity assessment output."""

    chi2: float | None = None
    p_value: float | None = None
    residual_rms: float | None = None
    num_sats_used: int = 0
    num_rejected: int = 0
    excluded_sv_ids: list[int] = field(default_factory=list)
    is_suspect: bool = False
    is_invalid: bool = False
    reason_codes: list[str] = field(default_factory=list)
