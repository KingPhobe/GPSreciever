"""Signal strength and pseudorange noise models."""

from __future__ import annotations

import numpy as np


def cn0_from_elevation(
    elev_deg: float,
    cn0_zenith_dbhz: float = 45.0,
    cn0_min_dbhz: float = 25.0,
) -> float:
    """Return a simple CN0 model based on elevation angle."""

    elev_deg = float(elev_deg)
    elev_deg = np.clip(elev_deg, 0.0, 90.0)
    weight = np.sin(np.deg2rad(elev_deg))
    cn0 = cn0_min_dbhz + (cn0_zenith_dbhz - cn0_min_dbhz) * weight
    return float(np.clip(cn0, cn0_min_dbhz, cn0_zenith_dbhz))


def pseudorange_sigma_m(
    cn0_dbhz: float,
    elev_deg: float,
    base_sigma_m: float = 0.6,
    elevation_weight: float = 1.0,
    cn0_ref_dbhz: float = 45.0,
) -> float:
    """Return pseudorange sigma based on CN0 and elevation weighting."""

    cn0_dbhz = float(cn0_dbhz)
    elev_deg = float(elev_deg)
    cn0_factor = 10.0 ** ((cn0_ref_dbhz - cn0_dbhz) / 20.0)
    elev_weight = max(np.sin(np.deg2rad(max(elev_deg, 0.1))), 0.1)
    elev_factor = elev_weight ** (-elevation_weight)
    return float(base_sigma_m * cn0_factor * elev_factor)
