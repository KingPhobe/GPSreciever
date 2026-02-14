"""Pseudorange measurement models and synthetic measurement source."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gnss_twin.attacks import AttackModel
from gnss_twin.attacks.base import AttackReport
from gnss_twin.models import Constellation, GnssMeasurement, MeasurementSource, ReceiverTruth
from gnss_twin.meas.clock_models import RandomWalkClock
from gnss_twin.meas.iono_klobuchar import klobuchar_delay_m
from gnss_twin.meas.multipath import multipath_bias_m
from gnss_twin.meas.noise import cn0_from_elevation, pseudorange_sigma_m
from gnss_twin.meas.tropo_saastamoinen import saastamoinen_delay_m
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import ecef_to_lla

LIGHT_SPEED_MPS = 299_792_458.0


def geometric_range_m(receiver_ecef_m: np.ndarray, sv_ecef_m: np.ndarray) -> float:
    """Compute geometric range between receiver and satellite."""

    return float(np.linalg.norm(sv_ecef_m - receiver_ecef_m))


def pseudorange_m(
    receiver_ecef_m: np.ndarray,
    receiver_clock_bias_s: float,
    sv_ecef_m: np.ndarray,
    sv_clock_bias_s: float,
) -> float:
    """Compute pseudorange including receiver and satellite clock biases."""

    range_m = geometric_range_m(receiver_ecef_m, sv_ecef_m)
    clock_term_m = LIGHT_SPEED_MPS * (receiver_clock_bias_s - sv_clock_bias_s)
    return range_m + clock_term_m


@dataclass
class SyntheticMeasurementSource(MeasurementSource):
    """Generate pseudorange measurements with configurable error models."""

    constellation: Constellation
    receiver_truth: ReceiverTruth
    elevation_mask_deg: float = 10.0
    cn0_zenith_dbhz: float = 45.0
    cn0_min_dbhz: float = 25.0
    healthy_cn0_dbhz: float = 28.0
    pr_sigma_base_m: float = 0.6
    pr_elev_weight: float = 1.0
    prr_sigma_mps: float = 0.2
    iono_alpha: tuple[float, float, float, float] | None = None
    iono_beta: tuple[float, float, float, float] | None = None
    pressure_hpa: float = 1013.25
    temp_k: float = 293.15
    rel_humidity: float = 0.5
    enable_multipath: bool = True
    multipath_max_bias_m: float = 1.5
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    clock_model: RandomWalkClock | None = None
    attacks: list[AttackModel] = field(default_factory=list)

    _last_t: float | None = field(init=False, default=None)
    _last_sv_states: list = field(init=False, default_factory=list)
    _clock_model: RandomWalkClock = field(init=False)
    _rx_lat_deg: float = field(init=False)
    _rx_lon_deg: float = field(init=False)
    _rx_alt_m: float = field(init=False)
    last_attack_report: AttackReport = field(init=False, default_factory=AttackReport)

    def __post_init__(self) -> None:
        if self.clock_model is None:
            seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
            self._clock_model = RandomWalkClock(
                initial_bias_s=self.receiver_truth.clk_bias_s,
                initial_drift_sps=self.receiver_truth.clk_drift_sps,
                seed=seed,
            )
        else:
            self._clock_model = self.clock_model

        # Receiver truth is static in this phase, so cache the LLA conversion.
        # If receiver_truth becomes dynamic in a future phase, revisit this cache.
        self._rx_lat_deg, self._rx_lon_deg, self._rx_alt_m = ecef_to_lla(
            *self.receiver_truth.pos_ecef_m
        )

    @property
    def receiver_clock_bias_s(self) -> float:
        return self._clock_model.state.bias_s

    @property
    def receiver_clock_drift_sps(self) -> float:
        return self._clock_model.state.drift_sps

    def get_measurements(self, t: float) -> list[GnssMeasurement]:
        sv_states = self.constellation.get_sv_states(t)
        prev_t = self._last_t
        self._last_t = float(t)
        self._last_sv_states = list(sv_states)
        measurements: list[GnssMeasurement] = []
        attack_report = AttackReport()
        if prev_t is None:
            dt = 0.0
        else:
            dt = float(t - prev_t)
        if dt > 0.0:
            self._clock_model.step(dt)

        for state in sv_states:
            elev_deg, az_deg = elev_az_from_rx_sv(self.receiver_truth.pos_ecef_m, state.pos_ecef_m)
            if elev_deg < self.elevation_mask_deg:
                continue
            cn0_dbhz = cn0_from_elevation(
                elev_deg,
                cn0_zenith_dbhz=self.cn0_zenith_dbhz,
                cn0_min_dbhz=self.cn0_min_dbhz,
            )
            sigma_pr_m = pseudorange_sigma_m(
                cn0_dbhz,
                elev_deg,
                base_sigma_m=self.pr_sigma_base_m,
                elevation_weight=self.pr_elev_weight,
            )
            iono_delay_m = klobuchar_delay_m(
                t,
                self._rx_lat_deg,
                self._rx_lon_deg,
                elev_deg,
                az_deg,
                alpha=self.iono_alpha,
                beta=self.iono_beta,
            )
            tropo_delay_m = saastamoinen_delay_m(
                elev_deg,
                self._rx_lat_deg,
                self._rx_alt_m,
                pressure_hpa=self.pressure_hpa,
                temp_k=self.temp_k,
                rel_humidity=self.rel_humidity,
            )
            model_corr_m = iono_delay_m + tropo_delay_m
            multipath_m = (
                multipath_bias_m(
                    elev_deg,
                    rng=self.rng,
                    max_bias_m=self.multipath_max_bias_m,
                )
                if self.enable_multipath
                else 0.0
            )
            noise_m = float(self.rng.normal(0.0, sigma_pr_m))
            pr_m = pseudorange_m(
                self.receiver_truth.pos_ecef_m,
                self.receiver_clock_bias_s,
                state.pos_ecef_m,
                state.clk_bias_s,
            ) + model_corr_m + multipath_m + noise_m
            los = state.pos_ecef_m - self.receiver_truth.pos_ecef_m
            rho = float(np.linalg.norm(los))
            los_unit = los / rho
            range_rate_mps = float(
                np.dot(state.vel_ecef_mps - self.receiver_truth.vel_ecef_mps, los_unit)
            )
            prr_noise_mps = float(self.rng.normal(0.0, self.prr_sigma_mps))
            prr_mps = (
                range_rate_mps
                + LIGHT_SPEED_MPS * (self.receiver_clock_drift_sps - state.clk_drift_sps)
                + prr_noise_mps
            )
            meas = GnssMeasurement(
                sv_id=state.sv_id,
                t=t,
                pr_m=pr_m,
                pr_model_corr_m=model_corr_m,
                prr_mps=prr_mps,
                sigma_pr_m=sigma_pr_m,
                cn0_dbhz=cn0_dbhz,
                elev_deg=elev_deg,
                az_deg=az_deg,
                flags={"healthy": cn0_dbhz >= self.healthy_cn0_dbhz},
            )
            for attack in self.attacks:
                meas, delta = attack.apply(meas, state, rx_truth=self.receiver_truth)
                if delta.applied:
                    attack_report.applied_count += 1
                    attack_report.pr_bias_sum_m += delta.pr_bias_m
                    attack_report.prr_bias_sum_mps += delta.prr_bias_mps
            measurements.append(meas)
        self.last_attack_report = attack_report
        return measurements
