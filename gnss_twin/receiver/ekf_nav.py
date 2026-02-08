"""Extended Kalman filter navigation solution with WLS initialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_twin.meas.pseudorange import LIGHT_SPEED_MPS
from gnss_twin.models import GnssMeasurement, SvState
from gnss_twin.receiver.wls_pvt import WlsPvtResult, wls_pvt


@dataclass
class EkfConfig:
    """Tunable parameters for the EKF navigation filter."""

    accel_sigma_mps2: float = 0.5
    clock_bias_sigma_s: float = 5e-4
    clock_drift_sigma_sps: float = 5e-6
    prr_sigma_mps: float = 0.5


class EkfNav:
    """Simple EKF for receiver position, velocity, and clock states."""

    def __init__(self, config: EkfConfig | None = None) -> None:
        self.config = config or EkfConfig()
        self.x = np.zeros(8, dtype=float)
        self.P = np.diag(
            [
                1e6,
                1e6,
                1e6,
                1e2,
                1e2,
                1e2,
                (1e-3) ** 2,
                (1e-4) ** 2,
            ]
        )
        self.initialized = False
        self.last_nis: float | None = None
        self.last_innov_dim: int | None = None

    @property
    def pos_ecef_m(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel_ecef_mps(self) -> np.ndarray:
        return self.x[3:6]

    @property
    def clk_bias_s(self) -> float:
        return float(self.x[6])

    @property
    def clk_drift_sps(self) -> float:
        return float(self.x[7])

    def initialize_from_wls(self, solution: WlsPvtResult) -> None:
        vel = solution.vel_ecef_mps if solution.vel_ecef_mps is not None else np.zeros(3)
        clk_drift = solution.clk_drift_sps if solution.clk_drift_sps is not None else 0.0
        self.x = np.hstack([solution.pos_ecef_m, vel, solution.clk_bias_s, clk_drift]).astype(float)
        self.P = np.diag(
            [
                solution.covariance[0, 0],
                solution.covariance[1, 1],
                solution.covariance[2, 2],
                25.0,
                25.0,
                25.0,
                solution.covariance[3, 3],
                (5e-5) ** 2,
            ]
        )
        self.initialized = True

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        f = np.eye(8)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        f[6, 7] = dt
        self.x = f @ self.x

        q_acc = self.config.accel_sigma_mps2**2
        q_clock_bias = self.config.clock_bias_sigma_s**2
        q_clock_drift = self.config.clock_drift_sigma_sps**2
        q = np.zeros((8, 8))
        pos_var = (dt**4) / 4.0 * q_acc
        pos_vel_cov = (dt**3) / 2.0 * q_acc
        vel_var = (dt**2) * q_acc
        for idx in range(3):
            q[idx, idx] = pos_var
            q[idx, idx + 3] = pos_vel_cov
            q[idx + 3, idx] = pos_vel_cov
            q[idx + 3, idx + 3] = vel_var
        q[6, 6] = (dt**2) * q_clock_bias
        q[7, 7] = (dt**2) * q_clock_drift
        self.P = f @ self.P @ f.T + q

    def update_pseudorange(
        self,
        measurements: list[GnssMeasurement],
        sv_states: list[SvState],
        *,
        initial_pos_ecef_m: np.ndarray | None = None,
        initial_clk_bias_s: float = 0.0,
    ) -> bool:
        if not self.initialized:
            if len(measurements) < 4:
                return False
            wls_solution = wls_pvt(
                measurements,
                sv_states,
                initial_pos_ecef_m=initial_pos_ecef_m,
                initial_clk_bias_s=initial_clk_bias_s,
            )
            if wls_solution is None:
                return False
            self.initialize_from_wls(wls_solution)
            return True
        return self._update_range(measurements, sv_states)

    def update_prr(self, measurements: list[GnssMeasurement], sv_states: list[SvState]) -> bool:
        if not self.initialized:
            return False
        return self._update_range_rate(measurements, sv_states)

    def _update_range(self, measurements: list[GnssMeasurement], sv_states: list[SvState]) -> bool:
        sv_by_id = {state.sv_id: state for state in sv_states}
        h_rows: list[list[float]] = []
        residuals: list[float] = []
        weights: list[float] = []
        pos = self.pos_ecef_m
        cb = self.clk_bias_s
        for meas in measurements:
            state = sv_by_id.get(meas.sv_id)
            if state is None:
                continue
            los = state.pos_ecef_m - pos
            rho = float(np.linalg.norm(los))
            if rho <= 0.0:
                continue
            los_unit = los / rho
            predicted = rho + LIGHT_SPEED_MPS * (cb - state.clk_bias_s)
            pr_corr = meas.pr_m - getattr(meas, "pr_model_corr_m", 0.0)
            residuals.append(pr_corr - predicted)
            h = np.zeros(8, dtype=float)
            h[:3] = -los_unit
            h[6] = LIGHT_SPEED_MPS
            h_rows.append(h.tolist())
            sigma = max(float(meas.sigma_pr_m), 1e-3)
            weights.append(sigma * sigma)
        if not h_rows:
            return False
        h_matrix = np.array(h_rows, dtype=float)
        y = np.array(residuals, dtype=float)
        r = np.diag(weights)
        return self._kalman_update(h_matrix, y, r)

    def _update_range_rate(self, measurements: list[GnssMeasurement], sv_states: list[SvState]) -> bool:
        sv_by_id = {state.sv_id: state for state in sv_states}
        h_rows: list[list[float]] = []
        residuals: list[float] = []
        weights: list[float] = []
        pos = self.pos_ecef_m
        vel = self.vel_ecef_mps
        cd = self.clk_drift_sps
        for meas in measurements:
            if meas.prr_mps is None:
                continue
            state = sv_by_id.get(meas.sv_id)
            if state is None:
                continue
            los = state.pos_ecef_m - pos
            rho = float(np.linalg.norm(los))
            if rho <= 0.0:
                continue
            los_unit = los / rho
            predicted = (
                float(np.dot(state.vel_ecef_mps - vel, los_unit))
                + LIGHT_SPEED_MPS * (cd - state.clk_drift_sps)
            )
            residuals.append(meas.prr_mps - predicted)
            h = np.zeros(8, dtype=float)
            h[3:6] = -los_unit
            h[7] = LIGHT_SPEED_MPS
            h_rows.append(h.tolist())
            weights.append(self.config.prr_sigma_mps**2)
        if not h_rows:
            return False
        h_matrix = np.array(h_rows, dtype=float)
        y = np.array(residuals, dtype=float)
        r = np.diag(weights)
        return self._kalman_update(h_matrix, y, r)

    def _kalman_update(self, h_matrix: np.ndarray, residuals: np.ndarray, r: np.ndarray) -> bool:
        s = h_matrix @ self.P @ h_matrix.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return False
        self.last_nis = float(residuals.T @ s_inv @ residuals)
        self.last_innov_dim = int(residuals.shape[0])
        k = self.P @ h_matrix.T @ s_inv
        self.x = self.x + k @ residuals
        i = np.eye(self.P.shape[0])
        self.P = (i - k @ h_matrix) @ self.P @ (i - k @ h_matrix).T + k @ r @ k.T
        return True
