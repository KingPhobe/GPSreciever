"""Core data models and interfaces for GNSS twin."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GnssMeasurement:
    """Single-satellite measurement at a given epoch."""

    sv_id: str
    t: float
    pr_m: float
    prr_mps: float | None
    sigma_pr_m: float
    cn0_dbhz: float
    elev_deg: float
    az_deg: float
    pr_model_corr_m: float = 0.0  # Model correction to subtract in solver (e.g., iono+tropo).
    flags: Mapping[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SvState:
    """Satellite state at a given epoch."""

    sv_id: str
    t: float
    pos_ecef_m: np.ndarray
    vel_ecef_mps: np.ndarray
    clk_bias_s: float
    clk_drift_sps: float


@dataclass(frozen=True)
class ReceiverTruth:
    """Receiver truth state for an epoch."""

    pos_ecef_m: np.ndarray
    vel_ecef_mps: np.ndarray
    clk_bias_s: float
    clk_drift_sps: float


@dataclass(frozen=True)
class DopMetrics:
    """Dilution of precision metrics."""

    gdop: float
    pdop: float
    hdop: float
    vdop: float


@dataclass(frozen=True)
class ResidualStats:
    """Residual summary statistics for a solution."""

    rms_m: float
    mean_m: float
    max_m: float
    chi_square: float


@dataclass(frozen=True)
class FixFlags:
    """Fix status and satellite usage metadata."""

    fix_type: str
    valid: bool
    sv_used: list[str]
    sv_rejected: list[str]
    sv_count: int
    sv_in_view: int
    mask_ok: bool
    pdop: float
    gdop: float
    chi_square: float
    chi_square_threshold: float


@dataclass(frozen=True)
class PvtSolution:
    """Position/velocity/time solution summary."""

    pos_ecef: np.ndarray
    vel_ecef: np.ndarray | None
    clk_bias_s: float
    clk_drift_sps: float
    dop: DopMetrics
    residuals: ResidualStats
    fix_flags: FixFlags


@dataclass(frozen=True)
class EpochLog:
    """Per-epoch log data."""

    t: float
    meas: list[GnssMeasurement]
    solution: PvtSolution | None
    truth: ReceiverTruth | None
    nis: float | None = None
    nis_alarm: bool = False
    innov_dim: int | None = None
    per_sv_stats: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


class MeasurementSource(ABC):
    """Interface for measurement sources."""

    @abstractmethod
    def get_measurements(self, t: float) -> list[GnssMeasurement]:
        """Return measurements for the given epoch time."""


class Constellation(ABC):
    """Interface for satellite constellation state providers."""

    @abstractmethod
    def get_sv_states(self, t: float) -> list[SvState]:
        """Return satellite states for the given epoch time."""


class NavSolver(ABC):
    """Interface for navigation solvers."""

    @abstractmethod
    def solve(self, meas: list[GnssMeasurement], sv_states: list[SvState]) -> PvtSolution:
        """Solve for the navigation state using measurements and satellite states."""
