"""Simple precision-clock authenticator with lock/holdover state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from gnss_twin.timing.pps import PpsTelemetry, PpsTelemetryBuilder, PpsTelemetryConfig


@dataclass(frozen=True)
class AuthTelemetry:
    """Authentication timing telemetry/state."""

    t_s: float
    auth_bit: int
    reason_code: str
    locked: bool
    holdover_active: bool
    rms_error_s: float
    residual_s: float
    bias_s: float
    drift_sps: float
    samples_in_window: int


@dataclass(frozen=True)
class AuthenticatorConfig:
    """Configuration for the simple precision-clock authenticator."""

    seed: int | None = 1
    sigma_process_bias_s: float = 1.0e-9
    sigma_process_drift_sps: float = 2.0e-10
    gain_bias: float = 0.25
    gain_drift: float = 0.05
    rms_window: int = 10
    min_samples_to_lock: int = 6
    rms_lock_threshold_s: float = 1.5e-7
    rms_holdover_threshold_s: float = 3.0e-7
    holdover_max_s: float = 20.0
    pps_config: PpsTelemetryConfig = PpsTelemetryConfig()


class Authenticator:
    """Bias+drift precision clock model with lock/holdover logic."""

    REASON_CODES = {
        "INIT": "insufficient_history",
        "LOCKED": "rms_within_lock_threshold",
        "HOLDOVER": "gnss_lost_holdover_active",
        "GNSS_INVALID": "gnss_invalid_not_locked",
        "RMS_HIGH": "rms_above_threshold",
    }

    def __init__(self, config: AuthenticatorConfig | None = None) -> None:
        self._config = config or AuthenticatorConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._pps_builder = PpsTelemetryBuilder(self._config.pps_config)
        self.reset()

    def reset(self) -> None:
        """Reset filter and lock state."""

        self._bias_s = 0.0
        self._drift_sps = 0.0
        self._last_t_s: float | None = None
        self._was_locked = False
        self._holdover_elapsed_s = 0.0
        self._residual_window: deque[float] = deque(maxlen=self._config.rms_window)

    def step(
        self,
        t_s: float,
        pps_platform_edge_s: float,
        pps_ref_edge_s: float,
        gnss_valid: bool,
    ) -> tuple[AuthTelemetry, PpsTelemetry]:
        """Advance one time step and return authenticator + PPS telemetry."""

        t_s = float(t_s)
        if self._last_t_s is None:
            dt_s = 1.0
        else:
            dt_s = max(1.0e-6, t_s - self._last_t_s)
        self._last_t_s = t_s

        process_bias = float(self._rng.normal(0.0, self._config.sigma_process_bias_s * np.sqrt(dt_s)))
        process_drift = float(self._rng.normal(0.0, self._config.sigma_process_drift_sps * np.sqrt(dt_s)))

        self._bias_s += self._drift_sps * dt_s + process_bias
        self._drift_sps += process_drift

        measured_bias_s = float(pps_platform_edge_s) - float(pps_ref_edge_s)
        residual_s = measured_bias_s - self._bias_s

        if gnss_valid:
            self._bias_s += self._config.gain_bias * residual_s
            self._drift_sps += self._config.gain_drift * residual_s / dt_s
            self._residual_window.append(residual_s)
            self._holdover_elapsed_s = 0.0
        elif self._was_locked:
            self._holdover_elapsed_s += dt_s

        rms_error_s = self._compute_rms_error()
        enough_samples = len(self._residual_window) >= self._config.min_samples_to_lock

        locked = bool(
            gnss_valid and enough_samples and rms_error_s <= self._config.rms_lock_threshold_s
        )
        holdover_active = bool(
            (not gnss_valid)
            and self._was_locked
            and self._holdover_elapsed_s <= self._config.holdover_max_s
            and rms_error_s <= self._config.rms_holdover_threshold_s
        )

        auth_bit = int(locked)
        if locked:
            reason_code = self.REASON_CODES["LOCKED"]
        elif holdover_active:
            reason_code = self.REASON_CODES["HOLDOVER"]
        elif not gnss_valid:
            reason_code = self.REASON_CODES["GNSS_INVALID"]
        elif enough_samples:
            reason_code = self.REASON_CODES["RMS_HIGH"]
        else:
            reason_code = self.REASON_CODES["INIT"]

        self._was_locked = locked or holdover_active

        self._pps_builder.ref_edge(t_s)
        self._pps_builder.platform_edge(
            t_s=t_s,
            clk_bias_est_s=measured_bias_s,
            clk_bias_true_s=measured_bias_s,
        )
        self._pps_builder.auth_edge(t_s=t_s, auth_clock_bias_s=self._bias_s)
        pps_telemetry = self._pps_builder.telemetry(t_s=t_s)

        auth_telemetry = AuthTelemetry(
            t_s=t_s,
            auth_bit=auth_bit,
            reason_code=reason_code,
            locked=locked,
            holdover_active=holdover_active,
            rms_error_s=rms_error_s,
            residual_s=residual_s,
            bias_s=self._bias_s,
            drift_sps=self._drift_sps,
            samples_in_window=len(self._residual_window),
        )
        return auth_telemetry, pps_telemetry

    def _compute_rms_error(self) -> float:
        if not self._residual_window:
            return float("inf")
        values = np.asarray(self._residual_window, dtype=float)
        return float(np.sqrt(np.mean(values * values)))
