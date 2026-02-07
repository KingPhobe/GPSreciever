"""Visibility helpers for simplified satellite filtering."""

from __future__ import annotations

import numpy as np

from gnss_twin.models import SvState


def elevation_deg(receiver_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> float:
    """Compute elevation angle in degrees."""

    los = sat_ecef_m - receiver_ecef_m
    los_unit = los / np.linalg.norm(los)
    up_unit = receiver_ecef_m / np.linalg.norm(receiver_ecef_m)
    elev_rad = np.arcsin(np.clip(np.dot(los_unit, up_unit), -1.0, 1.0))
    return float(np.rad2deg(elev_rad))


def apply_elevation_mask(
    receiver_ecef_m: np.ndarray,
    sv_states: list[SvState],
    elev_mask_deg: float,
) -> list[SvState]:
    """Return satellites above the elevation mask."""

    visible: list[SvState] = []
    for sv in sv_states:
        elev = elevation_deg(receiver_ecef_m, sv.pos_ecef_m)
        if elev >= elev_mask_deg:
            visible.append(sv)
    return visible
