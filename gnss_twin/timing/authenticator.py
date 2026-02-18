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
    holdover_max_auth_minus_ref_s: float = 2e-6
    drift_rw_rms_sps_sqrt: float = 0.0
    require_ref_for_holdover_auth: bool = True
    pps_config: PpsTelemetryConfig = PpsTelemetryConfig()


class Authenticator:
    """Bias+drift precision clock model with lock/holdover logic."""

    REASON_CODES = {
        "INIT": "insufficient_history",
        "LOCKED": "auth_ok",
        "HOLDOVER": "auth_ok_holdover",
        "GNSS_INVALID": "gnss_invalid_not_locked",
        "RMS_HIGH": "rms_above_threshold",
        "HOLDOVER_NO_REF": "auth_fail_holdover_no_ref",
        "HOLDOVER_DRIFT": "auth_fail_holdover_drift",
        "HOLDOVER_DRIFT_EXCEEDED": "holdover_drift_exceeded",
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
        self._locked = False
        self._auth_latched = False
        self._holdover_elapsed_s = 0.0
        self._residual_window: deque[float] = deque(maxlen=self._config.rms_window)

    def step(
        self,
        t_s: float,
        pps_platform_edge_s: float,
        pps_ref_edge_s: float | None,
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

        if dt_s > 0.0 and self._config.drift_rw_rms_sps_sqrt > 0.0:
            self._drift_sps += float(self._rng.normal(0.0, self._config.drift_rw_rms_sps_sqrt * np.sqrt(dt_s)))

        pps_auth_edge_s = self._bias_s
        phase_err_s = None
        ref_err_s = None
        if pps_ref_edge_s is not None:
            phase_err_s = float(pps_platform_edge_s) - float(pps_ref_edge_s)
            ref_err_s = float(pps_auth_edge_s) - float(pps_ref_edge_s)

        disciplining = bool(gnss_valid and phase_err_s is not None)
        mode = "unlocked"
        residual_s = 0.0

        if disciplining:
            residual_s = phase_err_s - self._bias_s
            self._bias_s += self._config.gain_bias * residual_s
            self._drift_sps += self._config.gain_drift * residual_s / dt_s
            self._residual_window.append(residual_s)
            self._holdover_elapsed_s = 0.0
            mode = "disciplined"
        elif self._locked:
            mode = "holdover"
            self._holdover_elapsed_s += dt_s
            if ref_err_s is not None:
                residual_s = ref_err_s
                self._residual_window.append(ref_err_s)

        rms_error_s = self._compute_rms_error()
        sigma_s = None if not np.isfinite(rms_error_s) else rms_error_s
        enough_samples = len(self._residual_window) >= self._config.min_samples_to_lock

        unlock_reason = None
        if disciplining:
            self._locked = bool(enough_samples and rms_error_s <= self._config.rms_lock_threshold_s)
        elif mode == "holdover":
            if ref_err_s is not None and abs(ref_err_s) > self._config.holdover_max_auth_minus_ref_s:
                self._locked = False
                self._auth_latched = False
                mode = "unlocked"
                unlock_reason = self.REASON_CODES["HOLDOVER_DRIFT_EXCEEDED"]
            elif ref_err_s is None and self._holdover_elapsed_s > self._config.holdover_max_s:
                self._locked = False
                self._auth_latched = False
                mode = "unlocked"
                unlock_reason = self.REASON_CODES["GNSS_INVALID"]
        else:
            self._locked = False

        max_platform_minus_auth_s = self._config.rms_lock_threshold_s
        max_sigma_t_s = self._config.rms_holdover_threshold_s

        if disciplining:
            auth_ok = bool(
                self._locked
                and abs(phase_err_s) <= max_platform_minus_auth_s
                and sigma_s is not None
                and sigma_s <= max_sigma_t_s
            )
            auth_bit = int(auth_ok)
            if auth_ok:
                self._auth_latched = True
                reason_code = self.REASON_CODES["LOCKED"]
            else:
                self._auth_latched = False
                reason_code = self.REASON_CODES["RMS_HIGH"] if enough_samples else self.REASON_CODES["INIT"]
        elif mode == "holdover":
            if ref_err_s is None and self._config.require_ref_for_holdover_auth:
                auth_bit = 0
                reason_code = self.REASON_CODES["HOLDOVER_NO_REF"]
            else:
                holdover_ok = bool(
                    self._auth_latched
                    and ref_err_s is not None
                    and abs(ref_err_s) <= self._config.holdover_max_auth_minus_ref_s
                    and (sigma_s is None or sigma_s <= max_sigma_t_s)
                )
                auth_bit = int(holdover_ok)
                reason_code = self.REASON_CODES["HOLDOVER"] if holdover_ok else self.REASON_CODES["HOLDOVER_DRIFT"]
        else:
            auth_bit = 0
            if unlock_reason is not None:
                reason_code = unlock_reason
            elif not gnss_valid:
                reason_code = self.REASON_CODES["GNSS_INVALID"]
            elif enough_samples:
                reason_code = self.REASON_CODES["RMS_HIGH"]
            else:
                reason_code = self.REASON_CODES["INIT"]

        holdover_active = bool(mode == "holdover" and self._locked)

        measured_bias_s = phase_err_s if phase_err_s is not None else float(pps_platform_edge_s) - t_s
        ref_for_telemetry = t_s if pps_ref_edge_s is None else float(pps_ref_edge_s)

        self._pps_builder.ref_edge(ref_for_telemetry)
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
            locked=self._locked,
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


@dataclass
class Mode5Authenticator:
    """Mode-5 authenticator latch driven by ConOps gate output."""

    auth_bit: bool = False

    def step(self, mode5_gate: str, gnss_valid: bool, holdover_ok: bool) -> bool:
        """Advance authenticator latch state for a single epoch."""

        if mode5_gate == "deny":
            self.auth_bit = False
        elif mode5_gate == "allow":
            self.auth_bit = True
        elif mode5_gate == "hold_last":
            if gnss_valid:
                return self.auth_bit
            self.auth_bit = bool(self.auth_bit and holdover_ok)

        return self.auth_bit
