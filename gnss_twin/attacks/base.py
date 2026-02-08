"""Attack model interfaces for GNSS twin simulations."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


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
    ) -> "GnssMeasurement":
        """Apply attack to a single measurement."""
