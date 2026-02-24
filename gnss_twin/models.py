"""Core data models and interfaces for GNSS twin."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
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
    raim_passed: bool
    validity_reason: str


class FixType(IntEnum):
    """Numeric fix type mapping for GUI-ready telemetry."""

    NO_FIX = 0
    FIX_2D = 1
    FIX_3D = 2


_FIX_TYPE_LABELS = {
    "NO FIX": FixType.NO_FIX,
    "2D": FixType.FIX_2D,
    "3D": FixType.FIX_3D,
}


def fix_type_from_label(label: str) -> FixType:
    """Convert a solver fix-type label into a numeric enum."""

    return _FIX_TYPE_LABELS.get(label.upper(), FixType.NO_FIX)


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
    t_s: float | None = None
    fix_valid: bool | None = None
    raim_pass: bool | None = None
    fix_type: FixType | int | None = None
    sats_used: int | None = None
    pdop: float | None = None
    hdop: float | None = None
    vdop: float | None = None
    residual_rms_m: float | None = None
    pos_ecef: np.ndarray | None = None
    vel_ecef: np.ndarray | None = None
    clk_bias_s: float | None = None
    clk_drift_sps: float | None = None
    nis: float | None = None
    nis_alarm: bool = False
    nis_stat_alarm: bool = False
    integrity_alarm: bool = False
    clock_drift_alarm: bool = False
    composite_alarm: bool = False
    attack_name: str | None = None
    attack_active: bool = False
    attack_pr_bias_mean_m: float = 0.0
    attack_prr_bias_mean_mps: float = 0.0
    innov_dim: int | None = None
    conops_status: str | None = None
    conops_mode5: str | None = None
    conops_reason_codes: list[str] = field(default_factory=list)
    integrity_p_value: float | None = None
    integrity_residual_rms: float | None = None
    integrity_num_sats_used: int | None = None
    integrity_excluded_sv_ids_count: int | None = None
    pps_ref_edge_s: float | None = None
    pps_platform_edge_s: float | None = None
    pps_auth_edge_s: float | None = None
    pps_platform_minus_ref_s: float | None = None
    pps_auth_minus_ref_s: float | None = None
    pps_platform_minus_auth_s: float | None = None
    auth_bit: int | None = None
    auth_locked: bool | None = None
    auth_mode: str | None = None
    auth_sigma_t_s: float | None = None
    auth_reason_codes: list[str] = field(default_factory=list)
    per_sv_stats: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    pps_err_s: float | None = None
    holdover_ok: bool | None = None
    time_since_ground_pps_s: float | None = None
    mode5_gate_latch_bit: bool | None = None
    mode5_timing_auth_bit: bool | None = None
    mode5_authorized_bit: bool | None = None
    mode5_auth_bit: bool | None = None


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
