"""ConOps output types for PNT/Mode-5 status reporting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PntStatus(Enum):
    VALID = "valid"
    SUSPECT = "suspect"
    INVALID = "invalid"


class Mode5Gate(Enum):
    ALLOW = "allow"
    DENY = "deny"
    HOLD_LAST = "hold_last"


@dataclass
class ConopsOutput:
    status: PntStatus
    mode5: Mode5Gate
    reason_codes: list[str]
    tta_triggered: bool
    time_in_state_s: float
    last_transition_t_s: float
    integrity_summary: dict[str, float | int | bool]
