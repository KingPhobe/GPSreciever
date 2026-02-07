"""Iterative weighted least squares PVT solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_twin.meas.pseudorange import LIGHT_SPEED_MPS
from gnss_twin.models import DopMetrics, GnssMeasurement, SvState


@dataclass(frozen=True)
class WlsPvtResult:
    """Weighted least squares position/clock bias estimate."""

    pos_ecef_m: np.ndarray
    vel_ecef_mps: np.ndarray | None
    clk_bias_s: float
    clk_drift_sps: float | None
    residuals_m: dict[str, float]
    covariance: np.ndarray
    dop: DopMetrics


def _compute_dop_from_geometry(geometry: np.ndarray) -> DopMetrics:
    normal = geometry.T @ geometry
    try:
        q = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return DopMetrics(gdop=float("inf"), pdop=float("inf"), hdop=float("inf"), vdop=float("inf"))
    gdop = float(np.sqrt(np.trace(q)))
    pdop = float(np.sqrt(np.sum(np.diag(q)[:3])))
    hdop = float(np.sqrt(np.sum(np.diag(q)[:2])))
    vdop = float(np.sqrt(q[2, 2]))
    return DopMetrics(gdop=gdop, pdop=pdop, hdop=hdop, vdop=vdop)


def _build_matrices(
    pos_ecef_m: np.ndarray,
    clk_bias_s: float,
    measurements: list[GnssMeasurement],
    sv_by_id: dict[str, SvState],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    h_rows: list[list[float]] = []
    residuals: list[float] = []
    weights: list[float] = []
    sv_ids: list[str] = []
    for meas in measurements:
        state = sv_by_id.get(meas.sv_id)
        if state is None:
            continue
        los = state.pos_ecef_m - pos_ecef_m
        rho = float(np.linalg.norm(los))
        if rho <= 0.0:
            continue
        predicted = rho + LIGHT_SPEED_MPS * (clk_bias_s - state.clk_bias_s)
        residuals.append(meas.pr_m - predicted)
        h_rows.append((-(los / rho)).tolist() + [LIGHT_SPEED_MPS])
        sigma = max(float(meas.sigma_pr_m), 1e-3)
        weights.append(1.0 / (sigma * sigma))
        sv_ids.append(meas.sv_id)
    return np.array(h_rows, dtype=float), np.array(residuals, dtype=float), np.diag(weights), sv_ids


def _build_velocity_matrices(
    pos_ecef_m: np.ndarray,
    measurements: list[GnssMeasurement],
    sv_by_id: dict[str, SvState],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    h_rows: list[list[float]] = []
    residuals: list[float] = []
    weights: list[float] = []
    sv_ids: list[str] = []
    for meas in measurements:
        if meas.prr_mps is None:
            continue
        state = sv_by_id.get(meas.sv_id)
        if state is None:
            continue
        los = state.pos_ecef_m - pos_ecef_m
        rho = float(np.linalg.norm(los))
        if rho <= 0.0:
            continue
        los_unit = los / rho
        predicted_sv_term = float(np.dot(state.vel_ecef_mps, los_unit)) - LIGHT_SPEED_MPS * state.clk_drift_sps
        residuals.append(meas.prr_mps - predicted_sv_term)
        h_rows.append((-(los_unit)).tolist() + [LIGHT_SPEED_MPS])
        weights.append(1.0)
        sv_ids.append(meas.sv_id)
    return np.array(h_rows, dtype=float), np.array(residuals, dtype=float), np.diag(weights), sv_ids


def wls_pvt(
    measurements: list[GnssMeasurement],
    sv_states: list[SvState],
    initial_pos_ecef_m: np.ndarray | None = None,
    initial_clk_bias_s: float = 0.0,
    max_iter: int = 8,
    tol_m: float = 1e-4,
) -> WlsPvtResult | None:
    """Solve for receiver position and clock bias from pseudoranges."""

    if len(measurements) < 4:
        return None

    sv_by_id = {state.sv_id: state for state in sv_states}
    pos = (
        initial_pos_ecef_m.astype(float).copy()
        if initial_pos_ecef_m is not None
        else np.zeros(3, dtype=float)
    )
    clk_bias = float(initial_clk_bias_s)
    covariance = np.full((4, 4), np.nan)
    geometry = np.zeros((0, 4))
    for _ in range(max_iter):
        h_matrix, residuals, weights, sv_ids = _build_matrices(pos, clk_bias, measurements, sv_by_id)
        if h_matrix.shape[0] < 4:
            return None
        geometry = h_matrix
        normal = h_matrix.T @ weights @ h_matrix
        try:
            delta = np.linalg.solve(normal, h_matrix.T @ weights @ residuals)
        except np.linalg.LinAlgError:
            return None
        pos += delta[:3]
        clk_bias += delta[3]
        if np.linalg.norm(delta[:3]) < tol_m:
            break
    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        covariance = np.full((4, 4), np.nan)
    dop = _compute_dop_from_geometry(geometry)

    h_matrix, residuals, weights, sv_ids = _build_matrices(pos, clk_bias, measurements, sv_by_id)
    residuals_by_sv = {sv_id: float(resid) for sv_id, resid in zip(sv_ids, residuals)}
    vel_ecef_mps: np.ndarray | None = None
    clk_drift_sps: float | None = None
    vel_h_matrix, vel_residuals, vel_weights, _ = _build_velocity_matrices(pos, measurements, sv_by_id)
    if vel_h_matrix.shape[0] >= 4:
        vel_normal = vel_h_matrix.T @ vel_weights @ vel_h_matrix
        try:
            vel_solution = np.linalg.solve(vel_normal, vel_h_matrix.T @ vel_weights @ vel_residuals)
            vel_ecef_mps = vel_solution[:3]
            clk_drift_sps = float(vel_solution[3])
        except np.linalg.LinAlgError:
            vel_ecef_mps = None
            clk_drift_sps = None
    return WlsPvtResult(
        pos_ecef_m=pos,
        vel_ecef_mps=vel_ecef_mps,
        clk_bias_s=clk_bias,
        clk_drift_sps=clk_drift_sps,
        residuals_m=residuals_by_sv,
        covariance=covariance,
        dop=dop,
    )
