"""Configuration objects for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Simulation configuration defaults."""

    seed: int | None = None
    dt: float = 1.0
    duration: float = 60.0
    elev_mask_deg: float = 10.0
