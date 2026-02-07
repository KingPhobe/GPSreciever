"""Receiver clock bias/drift evolution models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClockState:
    """Clock bias/drift state."""

    bias_s: float
    drift_sps: float


class RandomWalkClock:
    """Random-walk clock model for bias/drift evolution.

    The clock bias integrates the drift, and both bias and drift receive
    white-noise perturbations each step. This is a simple random walk model
    often used for receiver clock simulations.
    """

    def __init__(
        self,
        initial_bias_s: float = 0.0,
        initial_drift_sps: float = 0.0,
        sigma_bias_s: float = 5e-9,
        sigma_drift_sps: float = 5e-10,
        seed: int | None = None,
    ) -> None:
        self._bias_s = float(initial_bias_s)
        self._drift_sps = float(initial_drift_sps)
        self._sigma_bias_s = float(sigma_bias_s)
        self._sigma_drift_sps = float(sigma_drift_sps)
        self._rng = np.random.default_rng(seed)

    @property
    def state(self) -> ClockState:
        return ClockState(bias_s=self._bias_s, drift_sps=self._drift_sps)

    def step(self, dt_s: float) -> ClockState:
        """Propagate the clock state forward by dt_s seconds."""

        dt_s = float(dt_s)
        if dt_s < 0.0:
            raise ValueError("dt_s must be non-negative.")
        if dt_s == 0.0:
            return self.state

        self._bias_s += self._drift_sps * dt_s
        self._bias_s += self._rng.normal(0.0, self._sigma_bias_s * np.sqrt(dt_s))
        self._drift_sps += self._rng.normal(0.0, self._sigma_drift_sps * np.sqrt(dt_s))
        return self.state
