"""Attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import warnings

from gnss_twin.attacks.base import AttackDelta, AttackModel
from gnss_twin.attacks.jamming import JamCn0DropAttack
from gnss_twin.attacks.pipeline import AttackPipeline
from gnss_twin.attacks.spoofing import (
    SpoofClockRampAttack,
    SpoofPositionOffsetAttack,
    SpoofPrRampAttack,
)

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
    ) -> tuple["GnssMeasurement", AttackDelta]:
        return meas, AttackDelta(applied=False)


def create_attack(name: str, params: dict) -> AttackModel:
    """Create an attack model by name."""

    lowered = name.lower()
    if lowered == "none":
        return NoOpAttack()
    if lowered == "spoof_clock_ramp":
        unknown = set(params) - {"start_t", "end_t", "ramp_rate_mps", "slope_mps", "slope", "t_end"}
        if unknown:
            warnings.warn(
                f"Unknown spoof_clock_ramp attack params: {sorted(unknown)}",
                stacklevel=2,
            )
        if "ramp_rate_mps" in params:
            ramp_rate_mps = float(params["ramp_rate_mps"])
        elif "slope_mps" in params:
            ramp_rate_mps = float(params["slope_mps"])
        elif "slope" in params:
            ramp_rate_mps = float(params["slope"])
        else:
            ramp_rate_mps = 1.0
        end_t = params.get("end_t", params.get("t_end"))
        return SpoofClockRampAttack(
            start_t=float(params.get("start_t", 20.0)),
            end_t=(float(end_t) if end_t is not None else None),
            ramp_rate_mps=ramp_rate_mps,
        )
    if lowered == "spoof_pr_ramp":
        unknown = set(params) - {
            "start_t",
            "end_t",
            "ramp_rate_mps",
            "target_sv",
            "slope_mps",
            "slope",
            "t_end",
            "auto_select_visible_sv",
            "strict_target_sv",
        }
        if unknown:
            warnings.warn(
                f"Unknown spoof_pr_ramp attack params: {sorted(unknown)}",
                stacklevel=2,
            )
        target_sv = str(params.get("target_sv", "")).strip()
        auto_select_visible_sv = bool(params.get("auto_select_visible_sv", False))
        strict_target_sv = bool(params.get("strict_target_sv", True))
        if not target_sv and not auto_select_visible_sv:
            raise ValueError("spoof_pr_ramp requires target_sv unless auto_select_visible_sv is true")
        if "ramp_rate_mps" in params:
            ramp_rate_mps = float(params["ramp_rate_mps"])
        elif "slope_mps" in params:
            ramp_rate_mps = float(params["slope_mps"])
        elif "slope" in params:
            ramp_rate_mps = float(params["slope"])
        else:
            ramp_rate_mps = 1.0
        end_t = params.get("end_t", params.get("t_end"))
        return SpoofPrRampAttack(
            start_t=float(params.get("start_t", 20.0)),
            end_t=(float(end_t) if end_t is not None else None),
            ramp_rate_mps=ramp_rate_mps,
            target_sv=target_sv,
            auto_select_visible_sv=auto_select_visible_sv,
            strict_target_sv=strict_target_sv,
        )
    if lowered in {"spoof_pos_offset", "spoof_position_offset"}:
        unknown = set(params) - {
            "start_t",
            "end_t",
            "north_m",
            "east_m",
            "up_m",
            "ramp_time_s",
            "t_end",
        }
        if unknown:
            warnings.warn(
                f"Unknown spoof_pos_offset attack params: {sorted(unknown)}",
                stacklevel=2,
            )
        end_t = params.get("end_t", params.get("t_end"))
        return SpoofPositionOffsetAttack(
            start_t=float(params.get("start_t", 20.0)),
            end_t=(float(end_t) if end_t is not None else None),
            north_m=float(params.get("north_m", 0.0)),
            east_m=float(params.get("east_m", 0.0)),
            up_m=float(params.get("up_m", 0.0)),
            ramp_time_s=float(params.get("ramp_time_s", 0.0)),
        )
    if lowered == "jam_cn0_drop":
        unknown = set(params) - {"start_t", "cn0_drop_db", "sigma_pr_scale", "sigma_prr_scale"}
        if unknown:
            warnings.warn(
                f"Unknown jam_cn0_drop attack params: {sorted(unknown)}",
                stacklevel=2,
            )
        return JamCn0DropAttack(
            start_t=float(params.get("start_t", 20.0)),
            cn0_drop_db=float(params.get("cn0_drop_db", 15.0)),
            sigma_pr_scale=float(params.get("sigma_pr_scale", 5.0)),
            sigma_prr_scale=float(params.get("sigma_prr_scale", 5.0)),
        )
    raise ValueError(f"Unknown attack model: {name}")


__all__ = [
    "AttackModel",
    "AttackDelta",
    "AttackPipeline",
    "NoOpAttack",
    "JamCn0DropAttack",
    "SpoofClockRampAttack",
    "SpoofPrRampAttack",
    "SpoofPositionOffsetAttack",
    "create_attack",
]
