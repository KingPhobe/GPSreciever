"""Reusable Phase 1 validation scenario runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import ReceiverTruth
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    enable_multipath: bool
    pr_sigma_base_m: float
    prr_sigma_mps: float


@dataclass(frozen=True)
class Phase1Metrics:
    scenario_name: str
    total_epochs: int
    valid_epochs: int
    max_error_m: float
    rms_error_m: float
    spike_epochs: int
    low_elev_spike_epochs: int
    low_elev_spike_ratio: float


def run_phase1_scenario(
    config: ScenarioConfig,
    *,
    duration_s: float = 60.0,
    seed: int = 42,
    spike_threshold_m: float = 2.0,
    low_elev_deg: float = 20.0,
) -> Phase1Metrics:
    receiver_lla = (37.4275, -122.1697, 30.0)
    receiver_truth_ecef = lla_to_ecef(*receiver_lla)
    receiver_clock = 4.2e-6
    receiver_truth = ReceiverTruth(
        pos_ecef_m=receiver_truth_ecef,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=receiver_clock,
        clk_drift_sps=0.0,
    )

    rng = np.random.default_rng(seed)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=seed))
    measurement_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth,
        pr_sigma_base_m=config.pr_sigma_base_m,
        prr_sigma_mps=config.prr_sigma_mps,
        enable_multipath=config.enable_multipath,
        rng=rng,
    )
    integrity_cfg = IntegrityConfig()
    tracker = SvTracker(integrity_cfg)

    times = np.arange(0.0, duration_s, 1.0)
    last_pos = receiver_truth_ecef + 100.0
    last_clk = receiver_clock
    pos_errors: list[float] = []
    spike_epochs = 0
    spike_low_elev = 0
    valid_epochs = 0

    for t in times:
        sv_states = constellation.get_sv_states(float(t))
        meas = measurement_source.get_measurements(float(t))
        solution, per_sv_stats = integrity_pvt(
            meas,
            sv_states,
            initial_pos_ecef_m=last_pos,
            initial_clk_bias_s=last_clk,
            config=integrity_cfg,
            tracker=tracker,
        )
        if solution.fix_flags.fix_type != "NO FIX" and np.isfinite(solution.pos_ecef).all():
            valid_epochs += 1
            pos_errors.append(float(np.linalg.norm(solution.pos_ecef - receiver_truth_ecef)))
            last_pos = solution.pos_ecef
            last_clk = solution.clk_bias_s

        residuals = [
            abs(stats["residual_m"])
            for stats in per_sv_stats.values()
            if np.isfinite(stats["residual_m"])
        ]
        if residuals:
            max_resid = max(residuals)
            min_elev = min(m.elev_deg for m in meas) if meas else float("inf")
            if max_resid >= spike_threshold_m:
                spike_epochs += 1
                if min_elev <= low_elev_deg:
                    spike_low_elev += 1

    pos_errors_np = np.array(pos_errors, dtype=float)
    max_error = float(np.nanmax(pos_errors_np)) if pos_errors_np.size else float("nan")
    rms_error = float(np.sqrt(np.nanmean(np.square(pos_errors_np)))) if pos_errors_np.size else float("nan")
    low_elev_spike_ratio = (spike_low_elev / spike_epochs) if spike_epochs else 0.0

    return Phase1Metrics(
        scenario_name=config.name,
        total_epochs=int(times.size),
        valid_epochs=valid_epochs,
        max_error_m=max_error,
        rms_error_m=rms_error,
        spike_epochs=spike_epochs,
        low_elev_spike_epochs=spike_low_elev,
        low_elev_spike_ratio=low_elev_spike_ratio,
    )


def default_phase1_scenarios() -> list[ScenarioConfig]:
    return [
        ScenarioConfig(
            name="No noise + no multipath (iono+tropo on)",
            enable_multipath=False,
            pr_sigma_base_m=0.0,
            prr_sigma_mps=0.0,
        ),
        ScenarioConfig(
            name="Multipath only (noise off)",
            enable_multipath=True,
            pr_sigma_base_m=0.0,
            prr_sigma_mps=0.0,
        ),
    ]
