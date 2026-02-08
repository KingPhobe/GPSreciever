"""Attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gnss_twin.attacks.base import AttackModel

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

    if name.lower() == "none":
        return NoOpAttack()
    raise ValueError(f"Unknown attack model: {name}")


__all__ = ["AttackModel", "NoOpAttack", "create_attack"]
