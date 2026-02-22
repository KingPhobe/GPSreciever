"""Spoofing attack models for GNSS twin simulations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import warnings
import numpy as np

from gnss_twin.utils.wgs84 import ecef_to_lla

from gnss_twin.attacks.base import AttackDelta, AttackModel

if TYPE_CHECKING:
    from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


@dataclass
class SpoofClockRampAttack:
    """Apply a coherent clock pull-off ramp to all pseudoranges."""

    start_t: float = 20.0
    end_t: float | None = None
    ramp_rate_mps: float = 1.0

    def reset(self, seed: int | None = None) -> None:
        return None

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> tuple["GnssMeasurement", AttackDelta]:
        if meas.t < self.start_t:
            return meas, AttackDelta(applied=False)
        if self.end_t is not None and meas.t > self.end_t:
            return meas, AttackDelta(applied=False)
        bias_m = (meas.t - self.start_t) * self.ramp_rate_mps
        prr_mps = (
            None if meas.prr_mps is None else meas.prr_mps + self.ramp_rate_mps
        )
        return (
            replace(meas, pr_m=meas.pr_m + bias_m, prr_mps=prr_mps),
            AttackDelta(applied=True, pr_bias_m=bias_m, prr_bias_mps=self.ramp_rate_mps),
        )


@dataclass
class SpoofPrRampAttack:
    """Apply a ramped pseudorange bias to a specific satellite."""

    start_t: float = 20.0
    end_t: float | None = None
    ramp_rate_mps: float = 1.0
    target_sv: str = ""
    auto_select_visible_sv: bool = False
    strict_target_sv: bool = True

    def __post_init__(self) -> None:
        self._warned_missing_target = False
        self._resolved_target_sv: str | None = self.target_sv
        self._visible_sv_ids: tuple[str, ...] = tuple()
        self._has_visibility_context = False

    def reset(self, seed: int | None = None) -> None:
        self._warned_missing_target = False
        self._resolved_target_sv = self.target_sv
        self._visible_sv_ids = tuple()
        self._has_visibility_context = False
        return None

    def set_visible_sv_ids(self, sv_ids: list[str]) -> None:
        self._visible_sv_ids = tuple(sorted(set(sv_ids)))
        self._has_visibility_context = True

    def _warn_missing_target(self) -> None:
        if self._warned_missing_target:
            return
        warnings.warn(
            (
                f"spoof_pr_ramp target_sv '{self.target_sv}' is not visible; "
                "attack will not be applied"
            ),
            stacklevel=3,
        )
        self._warned_missing_target = True

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> tuple["GnssMeasurement", AttackDelta]:
        if meas.t < self.start_t:
            return meas, AttackDelta(applied=False)
        if self.end_t is not None and meas.t > self.end_t:
            return meas, AttackDelta(applied=False)

        target_sv = self._resolved_target_sv
        if self._has_visibility_context and self.target_sv not in self._visible_sv_ids:
            if self.auto_select_visible_sv and self._visible_sv_ids:
                target_sv = self._visible_sv_ids[0]
                self._resolved_target_sv = target_sv
                if not self._warned_missing_target:
                    warnings.warn(
                        (
                            f"spoof_pr_ramp target_sv '{self.target_sv}' is not visible; "
                            f"auto-selecting '{target_sv}'"
                        ),
                        stacklevel=3,
                    )
                    self._warned_missing_target = True
            else:
                if self.strict_target_sv:
                    self._warn_missing_target()
                elif not self._warned_missing_target:
                    warnings.warn(
                        (
                            f"spoof_pr_ramp target_sv '{self.target_sv}' is not visible; "
                            "continuing with no-op for this epoch"
                        ),
                        stacklevel=3,
                    )
                    self._warned_missing_target = True
                return meas, AttackDelta(applied=False)
        elif target_sv is None:
            target_sv = self.target_sv
            self._resolved_target_sv = target_sv

        if target_sv is None or meas.sv_id != target_sv:
            return meas, AttackDelta(applied=False)

        bias_m = (meas.t - self.start_t) * self.ramp_rate_mps
        prr_mps = (
            None if meas.prr_mps is None else meas.prr_mps + self.ramp_rate_mps
        )
        return (
            replace(meas, pr_m=meas.pr_m + bias_m, prr_mps=prr_mps),
            AttackDelta(applied=True, pr_bias_m=bias_m, prr_bias_mps=self.ramp_rate_mps),
        )


@dataclass
class SpoofPositionOffsetAttack:
    """Spoof a coherent receiver position offset (applied consistently across all SVs).

    Measurement-domain spoof: bias pseudoranges (and optionally pseudorange-rates) as if the
    receiver were located at a different position.
    """

    start_t: float = 20.0
    end_t: float | None = None

    north_m: float = 0.0
    east_m: float = 0.0
    up_m: float = 0.0

    ramp_time_s: float = 0.0  # 0 => step

    def reset(self, seed: int | None = None) -> None:
        return None

    def apply(
        self,
        meas: "GnssMeasurement",
        sv_state: "SvState",
        *,
        rx_truth: "ReceiverTruth",
    ) -> tuple["GnssMeasurement", AttackDelta]:
        if meas.t < self.start_t:
            return meas, AttackDelta(applied=False)
        if self.end_t is not None and meas.t > self.end_t:
            return meas, AttackDelta(applied=False)

        scale = 1.0
        scale_dot = 0.0
        if self.ramp_time_s and self.ramp_time_s > 0.0:
            elapsed = float(meas.t - self.start_t)
            if elapsed <= 0.0:
                scale = 0.0
                scale_dot = 0.0
            elif elapsed >= float(self.ramp_time_s):
                scale = 1.0
                scale_dot = 0.0
            else:
                scale = elapsed / float(self.ramp_time_s)
                scale_dot = 1.0 / float(self.ramp_time_s)

        offset_ecef = _neu_to_ecef_offset(
            rx_truth.pos_ecef_m,
            north_m=float(self.north_m),
            east_m=float(self.east_m),
            up_m=float(self.up_m),
        )
        spoof_pos = rx_truth.pos_ecef_m + scale * offset_ecef

        los_true = sv_state.pos_ecef_m - rx_truth.pos_ecef_m
        rho_true = float(np.linalg.norm(los_true))
        if rho_true <= 0.0:
            return meas, AttackDelta(applied=False)

        los_spoof = sv_state.pos_ecef_m - spoof_pos
        rho_spoof = float(np.linalg.norm(los_spoof))
        bias_m = rho_spoof - rho_true

        prr_bias_mps = 0.0
        if scale_dot != 0.0:
            los_unit = los_true / rho_true
            prr_bias_mps = float(-np.dot(los_unit, scale_dot * offset_ecef))

        prr_mps = meas.prr_mps
        if prr_mps is not None:
            prr_mps = prr_mps + prr_bias_mps

        return (
            replace(meas, pr_m=meas.pr_m + bias_m, prr_mps=prr_mps),
            AttackDelta(applied=True, pr_bias_m=bias_m, prr_bias_mps=prr_bias_mps),
        )


def _neu_to_ecef_offset(ref_pos_ecef_m: "np.ndarray", *, north_m: float, east_m: float, up_m: float) -> "np.ndarray":
    """Convert local N/E/U offset (meters) into an ECEF delta vector."""

    lat_deg, lon_deg, _ = ecef_to_lla(
        float(ref_pos_ecef_m[0]),
        float(ref_pos_ecef_m[1]),
        float(ref_pos_ecef_m[2]),
    )
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    sin_lat = float(np.sin(lat))
    cos_lat = float(np.cos(lat))
    sin_lon = float(np.sin(lon))
    cos_lon = float(np.cos(lon))

    n_hat = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float)
    e_hat = np.array([-sin_lon, cos_lon, 0.0], dtype=float)
    u_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=float)

    return north_m * n_hat + east_m * e_hat + up_m * u_hat


__all__ = ["SpoofClockRampAttack", "SpoofPrRampAttack", "SpoofPositionOffsetAttack", "AttackModel"]
