"""Mission CONOPS state machine for PNT/Mode-5 gating."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from gnss_twin.integrity.report import IntegrityReport
from gnss_twin.runtime.conops_types import ConopsOutput, Mode5Gate, PntStatus
from gnss_twin.runtime.pnt_config import PntConfig


class ConopsState(Enum):
    NOMINAL = "nominal"
    SUSPECT = "suspect"
    INVALID = "invalid"
    RECOVERY = "recovery"
    REACQUIRED = "reacquired"


@dataclass
class _StateTracker:
    state: ConopsState
    state_entry_t_s: float
    last_transition_t_s: float
    suspect_start_t_s: float | None = None
    reacq_start_t_s: float | None = None


class ConopsStateMachine:
    """Deterministic state machine for CONOPS gating."""

    def __init__(self, cfg: PntConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self, t0_s: float = 0.0) -> None:
        self._tracker = _StateTracker(
            state=ConopsState.NOMINAL,
            state_entry_t_s=float(t0_s),
            last_transition_t_s=float(t0_s),
        )

    def step(
        self,
        t_s: float,
        integrity: IntegrityReport,
        sol: Any | dict,
        rx_obs: dict | None,
    ) -> ConopsOutput:
        del rx_obs
        t_s = float(t_s)

        residual_rms = _coalesce(
            integrity.residual_rms, _get_sol_metric(sol, "residual_rms")
        )
        p_value = integrity.p_value
        num_sats = _prefer_positive_int(
            integrity.num_sats_used, _get_sol_metric(sol, "num_sats")
        )
        pdop = _get_sol_metric(sol, "pdop")

        suspect_reasons = []
        invalid_reasons = []

        suspect_trigger = False
        if integrity.is_suspect:
            suspect_trigger = True
            suspect_reasons.append("integrity_suspect")
        if residual_rms is not None and residual_rms > self.cfg.residual_rms_suspect:
            suspect_trigger = True
            suspect_reasons.append("residual_rms_suspect")
        if p_value is not None and p_value < self.cfg.chi2_p_suspect:
            suspect_trigger = True
            suspect_reasons.append("chi2_p_suspect")

        invalid_trigger = False
        if integrity.is_invalid:
            invalid_trigger = True
            invalid_reasons.append("integrity_invalid")
        if residual_rms is not None and residual_rms > self.cfg.residual_rms_invalid:
            invalid_trigger = True
            invalid_reasons.append("residual_rms_invalid")
        if p_value is not None and p_value < self.cfg.chi2_p_invalid:
            invalid_trigger = True
            invalid_reasons.append("chi2_p_invalid")

        reasons: list[str] = []
        new_state = self._tracker.state

        if self._tracker.state == ConopsState.NOMINAL:
            if suspect_trigger:
                new_state = ConopsState.SUSPECT
                self._tracker.suspect_start_t_s = t_s
                reasons.extend(suspect_reasons)
                reasons.append("transition_nominal_to_suspect")
        elif self._tracker.state == ConopsState.SUSPECT:
            if invalid_trigger:
                new_state = ConopsState.INVALID
                reasons.extend(invalid_reasons)
                reasons.append("transition_suspect_to_invalid")
            elif suspect_trigger:
                if self._tracker.suspect_start_t_s is None:
                    self._tracker.suspect_start_t_s = self._tracker.state_entry_t_s
                suspect_duration = t_s - self._tracker.suspect_start_t_s
                if suspect_duration >= self.cfg.suspect_hold_s:
                    new_state = ConopsState.INVALID
                    reasons.extend(suspect_reasons)
                    reasons.append("suspect_hold_exceeded")
            else:
                new_state = ConopsState.NOMINAL
                reasons.append("suspect_cleared")
        elif self._tracker.state == ConopsState.INVALID:
            recovery_ready = (
                num_sats is not None
                and num_sats >= self.cfg.min_sats_valid
                and residual_rms is not None
                and residual_rms <= self.cfg.residual_rms_suspect
            )
            if recovery_ready:
                new_state = ConopsState.RECOVERY
                self._tracker.reacq_start_t_s = t_s
                reasons.append("recovery_conditions_met")
        elif self._tracker.state == ConopsState.RECOVERY:
            clean_integrity = not suspect_trigger and not invalid_trigger
            if invalid_trigger:
                new_state = ConopsState.INVALID
                reasons.extend(invalid_reasons)
                reasons.append("recovery_invalid")
            elif clean_integrity:
                if self._tracker.reacq_start_t_s is None:
                    self._tracker.reacq_start_t_s = t_s
                clean_duration = t_s - self._tracker.reacq_start_t_s
                if clean_duration >= self.cfg.reacq_confirm_s:
                    new_state = ConopsState.REACQUIRED
                    reasons.append("reacq_confirmed")
            else:
                self._tracker.reacq_start_t_s = None
                reasons.append("recovery_not_clean")
        elif self._tracker.state == ConopsState.REACQUIRED:
            if invalid_trigger:
                new_state = ConopsState.INVALID
                reasons.extend(invalid_reasons)
                reasons.append("reacquired_invalid")
            elif suspect_trigger:
                new_state = ConopsState.SUSPECT
                self._tracker.suspect_start_t_s = t_s
                reasons.extend(suspect_reasons)
                reasons.append("reacquired_suspect")

        if new_state != self._tracker.state:
            self._tracker.state = new_state
            self._tracker.state_entry_t_s = t_s
            self._tracker.last_transition_t_s = t_s

        time_in_state_s = t_s - self._tracker.state_entry_t_s
        pnt_status = _map_state_to_pnt_status(self._tracker.state)
        mode5 = _map_state_to_mode5(self._tracker.state)
        tta_triggered = (
            self._tracker.state in {ConopsState.SUSPECT, ConopsState.INVALID}
            and time_in_state_s >= self.cfg.tta_s
        )
        integrity_summary = {
            "p_value": p_value if p_value is not None else float("nan"),
            "residual_rms": residual_rms if residual_rms is not None else float("nan"),
            "num_sats": num_sats if num_sats is not None else 0,
            "pdop": pdop if pdop is not None else float("nan"),
            "integrity_suspect": integrity.is_suspect,
            "integrity_invalid": integrity.is_invalid,
        }
        if not reasons:
            reasons.append("state_stable")

        return ConopsOutput(
            status=pnt_status,
            mode5=mode5,
            reason_codes=reasons,
            tta_triggered=tta_triggered,
            time_in_state_s=time_in_state_s,
            last_transition_t_s=self._tracker.last_transition_t_s,
            integrity_summary=integrity_summary,
        )


def _coalesce(primary: Any | None, fallback: Any | None) -> Any | None:
    if primary is None:
        return fallback
    return primary


def _prefer_positive_int(primary: int | None, fallback: Any | None) -> int | None:
    if primary is not None and primary > 0:
        return primary
    if fallback is None:
        return None
    return int(fallback)


def _get_sol_metric(sol: Any | dict, key: str) -> float | int | None:
    if sol is None:
        return None
    if isinstance(sol, dict):
        return sol.get(key)
    return getattr(sol, key, None)


def _map_state_to_pnt_status(state: ConopsState) -> PntStatus:
    if state == ConopsState.SUSPECT:
        return PntStatus.SUSPECT
    if state in {ConopsState.INVALID, ConopsState.RECOVERY}:
        return PntStatus.INVALID
    return PntStatus.VALID


def _map_state_to_mode5(state: ConopsState) -> Mode5Gate:
    if state == ConopsState.SUSPECT:
        return Mode5Gate.HOLD_LAST
    if state in {ConopsState.INVALID, ConopsState.RECOVERY}:
        return Mode5Gate.DENY
    return Mode5Gate.ALLOW
