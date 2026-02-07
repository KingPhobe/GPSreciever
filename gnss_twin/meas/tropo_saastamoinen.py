"""Saastamoinen tropospheric delay model."""

from __future__ import annotations

import numpy as np


def _water_vapor_pressure_hpa(temp_k: float, rel_humidity: float) -> float:
    temp_c = temp_k - 273.15
    sat_pressure = 6.11 * np.exp((17.15 * temp_c) / (234.7 + temp_c))
    return float(rel_humidity * sat_pressure)


def saastamoinen_delay_m(
    elev_deg: float,
    lat_deg: float,
    alt_m: float,
    pressure_hpa: float = 1013.25,
    temp_k: float = 293.15,
    rel_humidity: float = 0.5,
) -> float:
    """Return zenith-mapped tropospheric delay in meters."""

    elev_rad = np.deg2rad(max(elev_deg, 0.1))
    lat_rad = np.deg2rad(lat_deg)
    pressure_hpa = max(0.0, pressure_hpa)
    temp_k = max(200.0, temp_k)
    rel_humidity = np.clip(rel_humidity, 0.0, 1.0)

    e_hpa = _water_vapor_pressure_hpa(temp_k, rel_humidity)
    hydrostatic = 0.0022768 * pressure_hpa / (
        1.0 - 0.00266 * np.cos(2.0 * lat_rad) - 0.00028 * alt_m / 1000.0
    )
    wet = 0.002277 * (1255.0 / temp_k + 0.05) * e_hpa
    mapping = 1.0 / np.sin(elev_rad)
    delay_m = (hydrostatic + wet) * mapping
    return float(max(delay_m, 0.0))
