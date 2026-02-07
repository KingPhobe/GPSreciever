"""WGS-84 coordinate utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WGS84:
    """WGS-84 ellipsoid constants."""

    a: float = 6_378_137.0
    f: float = 1.0 / 298.257223563

    @property
    def b(self) -> float:
        return self.a * (1.0 - self.f)

    @property
    def e2(self) -> float:
        return self.f * (2.0 - self.f)

    @property
    def ep2(self) -> float:
        b = self.b
        return (self.a**2 - b**2) / b**2


ELLIPSOID = WGS84()


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert geodetic latitude/longitude/altitude to ECEF.

    Args:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
        alt_m: Altitude above the WGS-84 ellipsoid in meters.

    Returns:
        ECEF position (x, y, z) in meters.
    """

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    n = ELLIPSOID.a / np.sqrt(1.0 - ELLIPSOID.e2 * sin_lat**2)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - ELLIPSOID.e2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_lla(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    """Convert ECEF to geodetic latitude/longitude/altitude.

    Uses the Bowring method for initial latitude and iterates to convergence.

    Returns:
        (lat_deg, lon_deg, alt_m)
    """

    lon = np.arctan2(y_m, x_m)
    p = np.hypot(x_m, y_m)

    if p == 0.0:
        lat = np.pi / 2.0 if z_m >= 0.0 else -np.pi / 2.0
        alt = abs(z_m) - ELLIPSOID.b
        return (np.rad2deg(lat), np.rad2deg(lon), alt)

    theta = np.arctan2(z_m * ELLIPSOID.a, p * ELLIPSOID.b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat = np.arctan2(
        z_m + ELLIPSOID.ep2 * ELLIPSOID.b * sin_theta**3,
        p - ELLIPSOID.e2 * ELLIPSOID.a * cos_theta**3,
    )

    for _ in range(5):
        sin_lat = np.sin(lat)
        n = ELLIPSOID.a / np.sqrt(1.0 - ELLIPSOID.e2 * sin_lat**2)
        alt = p / np.cos(lat) - n
        lat_next = np.arctan2(z_m, p * (1.0 - ELLIPSOID.e2 * n / (n + alt)))
        if abs(lat_next - lat) < 1e-12:
            lat = lat_next
            break
        lat = lat_next

    sin_lat = np.sin(lat)
    n = ELLIPSOID.a / np.sqrt(1.0 - ELLIPSOID.e2 * sin_lat**2)
    alt = p / np.cos(lat) - n

    return (np.rad2deg(lat), np.rad2deg(lon), alt)


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Return rotation matrix from ECEF to ENU at given geodetic coordinates."""

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )


def enu_from_ecef_delta(
    delta_ecef_m: np.ndarray, lat_deg: float, lon_deg: float
) -> np.ndarray:
    """Convert delta ECEF to ENU at given geodetic coordinates."""

    rot = ecef_to_enu_matrix(lat_deg, lon_deg)
    return rot @ delta_ecef_m
