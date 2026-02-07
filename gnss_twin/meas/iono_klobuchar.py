"""Klobuchar ionospheric delay model."""

from __future__ import annotations

import numpy as np

DEFAULT_ALPHA = (2.5e-8, 1.5e-8, -1.2e-7, 0.0)
DEFAULT_BETA = (90_000.0, 0.0, -110_000.0, 0.0)


def klobuchar_delay_m(
    t_s: float,
    lat_deg: float,
    lon_deg: float,
    elev_deg: float,
    az_deg: float,
    alpha: tuple[float, float, float, float] | None = None,
    beta: tuple[float, float, float, float] | None = None,
) -> float:
    """Return L1 ionospheric group delay in meters using the Klobuchar model."""

    alpha = DEFAULT_ALPHA if alpha is None else alpha
    beta = DEFAULT_BETA if beta is None else beta

    lat_sc = np.deg2rad(lat_deg) / np.pi
    lon_sc = np.deg2rad(lon_deg) / np.pi
    elev_sc = np.deg2rad(elev_deg) / np.pi
    az_sc = np.deg2rad(az_deg)

    elev_sc = max(elev_sc, 1e-3)
    psi = 0.0137 / (elev_sc + 0.11) - 0.022
    phi_i = lat_sc + psi * np.cos(az_sc)
    phi_i = np.clip(phi_i, -0.416, 0.416)
    lam_i = lon_sc + psi * np.sin(az_sc) / np.cos(phi_i * np.pi)
    phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * np.pi)

    t_local = 43_200.0 * lam_i + t_s
    t_local = np.mod(t_local, 86_400.0)

    amp = alpha[0] + alpha[1] * phi_m + alpha[2] * phi_m**2 + alpha[3] * phi_m**3
    amp = max(0.0, amp)
    per = beta[0] + beta[1] * phi_m + beta[2] * phi_m**2 + beta[3] * phi_m**3
    per = max(72_000.0, per)

    x = 2.0 * np.pi * (t_local - 50_400.0) / per
    if abs(x) < 1.57:
        delay_s = 5e-9 + amp * (1.0 - x**2 / 2.0 + x**4 / 24.0)
    else:
        delay_s = 5e-9

    f = 1.0 + 16.0 * (0.53 - elev_sc) ** 3
    delay_m = 299_792_458.0 * f * delay_s
    return float(max(delay_m, 0.0))
