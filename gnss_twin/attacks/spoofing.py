"""Spoofing attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gnss_twin.attacks.base import AttackModel

if TYPE_CHECKING:
    from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass
class SpoofClockRampAttack:
    """Apply a coherent clock pull-off ramp to all pseudoranges."""

    start_t: float = 20.0
    ramp_rate_mps: float = 1.0

    def reset(self, seed: int | None = None) -> None:
        return None

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> "GnssMeasurement":
        if meas.t < self.start_t:
            return meas
        bias_m = (meas.t - self.start_t) * self.ramp_rate_mps
        prr_mps = (
            None if meas.prr_mps is None else meas.prr_mps + self.ramp_rate_mps
        )
        return replace(meas, pr_m=meas.pr_m + bias_m, prr_mps=prr_mps)


@dataclass
class SpoofPrRampAttack:
    """Apply a ramped pseudorange bias to a specific satellite."""

    start_t: float = 20.0
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
    ) -> "GnssMeasurement":
        if meas.sv_id != self.target_sv:
            return meas
        if meas.t < self.start_t:
            return meas
        bias_m = (meas.t - self.start_t) * self.ramp_rate_mps
        prr_mps = (
            None if meas.prr_mps is None else meas.prr_mps + self.ramp_rate_mps
        )
        return replace(meas, pr_m=meas.pr_m + bias_m, prr_mps=prr_mps)


__all__ = ["SpoofClockRampAttack", "SpoofPrRampAttack", "AttackModel"]
