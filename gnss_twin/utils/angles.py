"""Angle utilities for GNSS geometry."""

from __future__ import annotations

import numpy as np

from gnss_twin.utils.wgs84 import ecef_to_lla, enu_from_ecef_delta


def elev_az_from_rx_sv(pos_rx: np.ndarray, pos_sv: np.ndarray) -> tuple[float, float]:
    """Compute elevation and azimuth (deg) from receiver to satellite using ENU."""

    lat_deg, lon_deg, _ = ecef_to_lla(*pos_rx)
    delta = pos_sv - pos_rx
    enu = enu_from_ecef_delta(delta, lat_deg, lon_deg)
    east, north, up = enu
    horiz = np.hypot(east, north)
    elev = np.rad2deg(np.arctan2(up, horiz))
    az = np.rad2deg(np.arctan2(east, north))
    if az < 0.0:
        az += 360.0
    return elev, az
