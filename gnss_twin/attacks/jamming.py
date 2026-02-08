"""Jamming attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gnss_twin.attacks.base import AttackModel

if TYPE_CHECKING:
    from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass
class JamCn0DropAttack:
    """Lower CN0 and inflate measurement sigmas after a start time."""

    start_t: float = 20.0
    cn0_drop_db: float = 15.0
    sigma_pr_scale: float = 5.0
    sigma_prr_scale: float = 5.0

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
        updates: dict[str, float] = {
            "cn0_dbhz": max(0.0, meas.cn0_dbhz - self.cn0_drop_db),
            "sigma_pr_m": meas.sigma_pr_m * self.sigma_pr_scale,
        }
        if hasattr(meas, "sigma_prr_mps"):
            sigma_prr_mps = getattr(meas, "sigma_prr_mps")
            if sigma_prr_mps is not None:
                updates["sigma_prr_mps"] = sigma_prr_mps * self.sigma_prr_scale
        return replace(meas, **updates)


__all__ = ["JamCn0DropAttack", "AttackModel"]
