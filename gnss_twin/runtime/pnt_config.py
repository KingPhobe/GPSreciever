"""Runtime configuration for PNT integrity heuristics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PntConfig:
    """Editable configuration for PNT integrity thresholds."""

    tta_s: float
    suspect_hold_s: float
    reacq_confirm_s: float
    min_sats_valid: int
    max_pdop_valid: float
    residual_rms_suspect: float
    residual_rms_invalid: float
    chi2_p_suspect: float
    chi2_p_invalid: float
    clock_innov_suspect: float
    clock_innov_invalid: float


def default_pnt_config() -> PntConfig:
    """Return conservative defaults (PLACEHOLDER — tune with Rcubed)."""

    return PntConfig(
        tta_s=10.0,  # PLACEHOLDER — tune with Rcubed
        suspect_hold_s=5.0,  # PLACEHOLDER — tune with Rcubed
        reacq_confirm_s=5.0,  # PLACEHOLDER — tune with Rcubed
        min_sats_valid=5,  # PLACEHOLDER — tune with Rcubed
        max_pdop_valid=6.0,  # PLACEHOLDER — tune with Rcubed
        # NOTE: In the current simulation, post-fit pseudorange residual RMS is typically
        # ~4–13 m in nominal runs (depending on the sigma model). Use thresholds that keep
        # baseline CONOPS in VALID and reserve INVALID for clearly-bad conditions.
        residual_rms_suspect=10.0,  # PLACEHOLDER — tune with Rcubed
        residual_rms_invalid=15.0,  # PLACEHOLDER — tune with Rcubed
        chi2_p_suspect=0.05,  # PLACEHOLDER — tune with Rcubed
        chi2_p_invalid=0.01,  # PLACEHOLDER — tune with Rcubed
        clock_innov_suspect=1e3,  # PLACEHOLDER — tune with Rcubed
        clock_innov_invalid=2e3,  # PLACEHOLDER — tune with Rcubed
    )
