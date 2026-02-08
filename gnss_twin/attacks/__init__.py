"""Attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gnss_twin.attacks.base import AttackModel
from gnss_twin.attacks.spoofing import SpoofClockRampAttack, SpoofPrRampAttack

if TYPE_CHECKING:
    from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass
class NoOpAttack:
    """Attack model that leaves measurements unchanged."""

    def reset(self, seed: int | None = None) -> None:
        return None

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> "GnssMeasurement":
        return meas


def create_attack(name: str, params: dict) -> AttackModel:
    """Create an attack model by name."""

    lowered = name.lower()
    if lowered == "none":
        return NoOpAttack()
    if lowered == "spoof_clock_ramp":
        return SpoofClockRampAttack(
            start_t=float(params.get("start_t", 20.0)),
            ramp_rate_mps=float(params.get("ramp_rate_mps", 1.0)),
        )
    if lowered == "spoof_pr_ramp":
        target_sv = str(params.get("target_sv", "")).strip()
        if not target_sv:
            raise ValueError("spoof_pr_ramp requires target_sv to be provided")
        return SpoofPrRampAttack(
            start_t=float(params.get("start_t", 20.0)),
            ramp_rate_mps=float(params.get("ramp_rate_mps", 1.0)),
            target_sv=target_sv,
        )
    raise ValueError(f"Unknown attack model: {name}")


__all__ = [
    "AttackModel",
    "NoOpAttack",
    "SpoofClockRampAttack",
    "SpoofPrRampAttack",
    "create_attack",
]
