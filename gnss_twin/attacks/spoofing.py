"""Spoofing attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import warnings

from gnss_twin.attacks.base import AttackDelta, AttackModel

if TYPE_CHECKING:
    from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass
class SpoofClockRampAttack:
    """Apply a coherent clock pull-off ramp to all pseudoranges."""

    start_t: float = 20.0
    end_t: float | None = None
    ramp_rate_mps: float = 1.0

    def reset(self, seed: int | None = None) -> None:
        return None

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> tuple["GnssMeasurement", AttackDelta]:
        if meas.t < self.start_t:
            return meas, AttackDelta(applied=False)
        if self.end_t is not None and meas.t > self.end_t:
            return meas, AttackDelta(applied=False)
        bias_m = (meas.t - self.start_t) * self.ramp_rate_mps
        prr_mps = (
            None if meas.prr_mps is None else meas.prr_mps + self.ramp_rate_mps
        )
        return (
            replace(meas, pr_m=meas.pr_m + bias_m, prr_mps=prr_mps),
            AttackDelta(applied=True, pr_bias_m=bias_m, prr_bias_mps=self.ramp_rate_mps),
        )


@dataclass
class SpoofPrRampAttack:
    """Apply a ramped pseudorange bias to a specific satellite."""

    start_t: float = 20.0
    end_t: float | None = None
    ramp_rate_mps: float = 1.0
    target_sv: str = ""
    auto_select_visible_sv: bool = False
    strict_target_sv: bool = True

    def __post_init__(self) -> None:
        self._warned_missing_target = False
        self._resolved_target_sv: str | None = self.target_sv
        self._visible_sv_ids: tuple[str, ...] = tuple()
        self._has_visibility_context = False

    def reset(self, seed: int | None = None) -> None:
        self._warned_missing_target = False
        self._resolved_target_sv = self.target_sv
        self._visible_sv_ids = tuple()
        self._has_visibility_context = False
        return None

    def set_visible_sv_ids(self, sv_ids: list[str]) -> None:
        self._visible_sv_ids = tuple(sorted(set(sv_ids)))
        self._has_visibility_context = True

    def _warn_missing_target(self) -> None:
        if self._warned_missing_target:
            return
        warnings.warn(
            (
                f"spoof_pr_ramp target_sv '{self.target_sv}' is not visible; "
                "attack will not be applied"
            ),
            stacklevel=3,
        )
        self._warned_missing_target = True

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> tuple["GnssMeasurement", AttackDelta]:
        if meas.t < self.start_t:
            return meas, AttackDelta(applied=False)
        if self.end_t is not None and meas.t > self.end_t:
            return meas, AttackDelta(applied=False)

        target_sv = self._resolved_target_sv
        if self._has_visibility_context and self.target_sv not in self._visible_sv_ids:
            if self.auto_select_visible_sv and self._visible_sv_ids:
                target_sv = self._visible_sv_ids[0]
                self._resolved_target_sv = target_sv
                if not self._warned_missing_target:
                    warnings.warn(
                        (
                            f"spoof_pr_ramp target_sv '{self.target_sv}' is not visible; "
                            f"auto-selecting '{target_sv}'"
                        ),
                        stacklevel=3,
                    )
                    self._warned_missing_target = True
            else:
                if self.strict_target_sv:
                    self._warn_missing_target()
                elif not self._warned_missing_target:
                    warnings.warn(
                        (
                            f"spoof_pr_ramp target_sv '{self.target_sv}' is not visible; "
                            "continuing with no-op for this epoch"
                        ),
                        stacklevel=3,
                    )
                    self._warned_missing_target = True
                return meas, AttackDelta(applied=False)
        elif target_sv is None:
            target_sv = self.target_sv
            self._resolved_target_sv = target_sv

        if target_sv is None or meas.sv_id != target_sv:
            return meas, AttackDelta(applied=False)

        bias_m = (meas.t - self.start_t) * self.ramp_rate_mps
        prr_mps = (
            None if meas.prr_mps is None else meas.prr_mps + self.ramp_rate_mps
        )
        return (
            replace(meas, pr_m=meas.pr_m + bias_m, prr_mps=prr_mps),
            AttackDelta(applied=True, pr_bias_m=bias_m, prr_bias_mps=self.ramp_rate_mps),
        )


__all__ = ["SpoofClockRampAttack", "SpoofPrRampAttack", "AttackModel"]
