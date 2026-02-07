"""Weighted least squares positioning solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_twin.meas.models import LIGHT_SPEED, Measurement


@dataclass(frozen=True)
class Solution:
    """Position solution with diagnostics."""

    position_ecef_m: np.ndarray
    clock_bias_s: float
    residuals_m: np.ndarray
    covariance: np.ndarray
    dop: dict[str, float]


def _compute_dop(covariance: np.ndarray) -> dict[str, float]:
    gdop = float(np.sqrt(np.trace(covariance)))
    pdop = float(np.sqrt(np.sum(np.diag(covariance)[:3])))
    hdop = float(np.sqrt(np.sum(np.diag(covariance)[:2])))
    vdop = float(np.sqrt(covariance[2, 2]))
    tdop = float(np.sqrt(covariance[3, 3]))
    return {"GDOP": gdop, "PDOP": pdop, "HDOP": hdop, "VDOP": vdop, "TDOP": tdop}


def wls_solve(
    measurements: list[Measurement],
    initial_position_ecef_m: np.ndarray,
    initial_clock_bias_s: float = 0.0,
    max_iter: int = 6,
    sigma_m: float = 1.0,
) -> Solution:
    """Estimate receiver position and clock bias from pseudoranges."""

    if len(measurements) < 4:
        raise ValueError("At least four measurements are required for a solution.")

    position = initial_position_ecef_m.astype(float).copy()
    clock_bias = float(initial_clock_bias_s)
    weights = np.eye(len(measurements)) / (sigma_m**2)
    for _ in range(max_iter):
        h_rows = []
        preds = []
        for meas in measurements:
            sat_pos = meas.sat_position_ecef_m
            los = sat_pos - position
            rho = float(np.linalg.norm(los))
            preds.append(rho + LIGHT_SPEED * (clock_bias - meas.sat_clock_bias_s))
            h_rows.append((-(los / rho)).tolist() + [LIGHT_SPEED])
        observed = np.array(
            [
                m.pseudorange_m - m.iono_delay_m - m.tropo_delay_m - m.noise_m
                for m in measurements
            ]
        )
        residuals = observed - np.array(preds)
        h_matrix = np.array(h_rows, dtype=float)
        normal = h_matrix.T @ weights @ h_matrix
        delta = np.linalg.solve(normal, h_matrix.T @ weights @ residuals)
        position += delta[:3]
        clock_bias += delta[3]
        if np.linalg.norm(delta[:3]) < 1e-4:
            break
    covariance = np.linalg.inv(normal)
    dop = _compute_dop(covariance)
    final_preds = []
    for meas in measurements:
        sat_pos = meas.sat_position_ecef_m
        los = sat_pos - position
        rho = float(np.linalg.norm(los))
        final_preds.append(rho + LIGHT_SPEED * (clock_bias - meas.sat_clock_bias_s))
    observed = np.array(
        [
            m.pseudorange_m - m.iono_delay_m - m.tropo_delay_m - m.noise_m
            for m in measurements
        ]
    )
    residuals = observed - np.array(final_preds)
    return Solution(position_ecef_m=position, clock_bias_s=clock_bias, residuals_m=residuals, covariance=covariance, dop=dop)
