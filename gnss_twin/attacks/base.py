"""Attack model interfaces for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass(frozen=True)
class AttackDelta:
    """Per-measurement attack delta for telemetry logging."""

    applied: bool
    pr_bias_m: float = 0.0
    prr_bias_mps: float = 0.0


@dataclass
class AttackReport:
    """Aggregated attack telemetry for an epoch."""

    applied_count: int = 0
    pr_bias_sum_m: float = 0.0
    prr_bias_sum_mps: float = 0.0


class AttackModel(Protocol):
    """Interface for applying adversarial modifications to measurements."""

    def reset(self, seed: int | None = None) -> None:
        """Reset any internal state, optionally with a seed."""

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> tuple["GnssMeasurement", AttackDelta]:
        """Apply attack to a single measurement and return a telemetry delta."""
