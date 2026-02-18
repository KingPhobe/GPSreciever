"""PPS telemetry helpers for reference/platform/auth timing edges."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PpsSource(Enum):
    """PPS pulse source classification."""

    GROUND = "ground"
    RECEIVER = "receiver"


@dataclass(frozen=True)
class PpsPulse:
    """Single PPS pulse stamped in true and local clock domains."""

    t_true_s: float
    t_local_s: float
    source: PpsSource


def pps_error_s(rx_pulse: PpsPulse, ground_pulse: PpsPulse) -> float:
    """Return PPS error in seconds using local-clock timestamps."""

    return float(rx_pulse.t_local_s) - float(ground_pulse.t_local_s)


@dataclass(frozen=True)
class PpsTelemetry:
    """PPS edge telemetry snapshot."""

    t_s: float
    ref_edge_s: float
    platform_edge_s: float
    auth_edge_s: float
    platform_minus_ref_s: float
    auth_minus_ref_s: float
    auth_minus_platform_s: float
    clk_bias_est_s: float
    clk_bias_true_s: float
    auth_clock_bias_s: float


@dataclass(frozen=True)
class PpsTelemetryConfig:
    """Noise and reproducibility configuration for PPS telemetry synthesis."""

    sigma_ref_jitter_s: float = 2.0e-9
    sigma_platform_jitter_s: float = 4.0e-9
    sigma_auth_jitter_s: float = 3.0e-9
    seed: int | None = 0


class PpsTelemetryBuilder:
    """Incrementally build PPS telemetry from three edge sources."""

    def __init__(self, config: PpsTelemetryConfig | None = None) -> None:
        self._config = config or PpsTelemetryConfig()
        self._rng = np.random.default_rng(self._config.seed)
        self._ref_edge_s = 0.0
        self._platform_edge_s = 0.0
        self._auth_edge_s = 0.0
        self._clk_bias_est_s = 0.0
        self._clk_bias_true_s = 0.0
        self._auth_clock_bias_s = 0.0

    def ref_edge(self, t_s: float) -> float:
        """Set/reference the reference PPS edge."""

        jitter = float(self._rng.normal(0.0, self._config.sigma_ref_jitter_s))
        self._ref_edge_s = float(t_s) + jitter
        return self._ref_edge_s

    def platform_edge(
        self,
        t_s: float,
        clk_bias_est_s: float,
        clk_bias_true_s: float,
    ) -> float:
        """Set/reference the platform PPS edge."""

        jitter = float(self._rng.normal(0.0, self._config.sigma_platform_jitter_s))
        self._clk_bias_est_s = float(clk_bias_est_s)
        self._clk_bias_true_s = float(clk_bias_true_s)
        self._platform_edge_s = float(t_s) + self._clk_bias_true_s + jitter
        return self._platform_edge_s

    def auth_edge(self, t_s: float, auth_clock_bias_s: float) -> float:
        """Set/reference the authenticator PPS edge."""

        jitter = float(self._rng.normal(0.0, self._config.sigma_auth_jitter_s))
        self._auth_clock_bias_s = float(auth_clock_bias_s)
        self._auth_edge_s = float(t_s) + self._auth_clock_bias_s + jitter
        return self._auth_edge_s

    def telemetry(self, t_s: float) -> PpsTelemetry:
        """Build telemetry from the current stored edges."""

        return self.build(
            t_s=t_s,
            ref_edge_s=self._ref_edge_s,
            platform_edge_s=self._platform_edge_s,
            auth_edge_s=self._auth_edge_s,
            clk_bias_est_s=self._clk_bias_est_s,
            clk_bias_true_s=self._clk_bias_true_s,
            auth_clock_bias_s=self._auth_clock_bias_s,
        )

    @staticmethod
    def build(
        t_s: float,
        ref_edge_s: float,
        platform_edge_s: float,
        auth_edge_s: float,
        clk_bias_est_s: float,
        clk_bias_true_s: float,
        auth_clock_bias_s: float,
    ) -> PpsTelemetry:
        """Build telemetry from explicit edge values and compute pairwise deltas."""

        platform_minus_ref_s = float(platform_edge_s) - float(ref_edge_s)
        auth_minus_ref_s = float(auth_edge_s) - float(ref_edge_s)
        auth_minus_platform_s = float(auth_edge_s) - float(platform_edge_s)
        return PpsTelemetry(
            t_s=float(t_s),
            ref_edge_s=float(ref_edge_s),
            platform_edge_s=float(platform_edge_s),
            auth_edge_s=float(auth_edge_s),
            platform_minus_ref_s=platform_minus_ref_s,
            auth_minus_ref_s=auth_minus_ref_s,
            auth_minus_platform_s=auth_minus_platform_s,
            clk_bias_est_s=float(clk_bias_est_s),
            clk_bias_true_s=float(clk_bias_true_s),
            auth_clock_bias_s=float(auth_clock_bias_s),
        )
