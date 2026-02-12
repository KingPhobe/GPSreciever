"""Spoofing attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

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

    def reset(self, seed: int | None = None) -> None:
        return None

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> tuple["GnssMeasurement", AttackDelta]:
        if meas.sv_id != self.target_sv:
            return meas, AttackDelta(applied=False)
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


__all__ = ["SpoofClockRampAttack", "SpoofPrRampAttack", "AttackModel"]
