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
    enable_tracking_placeholder: bool = True
    cn0_lock_on_dbhz: float = 30.0
    cn0_lock_off_dbhz: float = 25.0
    n_good_to_lock: int = 3
    n_bad_to_unlock: int = 2
    cn0_min_dbhz: float = 28.0
    sigma_pr_max_m: float = 20.0
    postfit_gate_sigma: float = 4.0
