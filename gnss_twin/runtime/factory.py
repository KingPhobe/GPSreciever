"""Factories for assembling runtime simulation components."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from scipy.stats import chi2
from gnss_twin.attacks import AttackPipeline, create_attack
from gnss_twin.config import SimConfig
from gnss_twin.models import EpochLog, ReceiverTruth, fix_type_from_label
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.runtime.engine import RaimIntegrityChecker, SimulationEngine
from gnss_twin.runtime.pnt_config import default_pnt_config
from gnss_twin.runtime.solver import DefaultPvtSolver
from gnss_twin.runtime.state_machine import ConopsStateMachine
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


def build_engine_with_truth(cfg: SimConfig) -> tuple[SimulationEngine, ReceiverTruth]:
    """Return a fully wired SimulationEngine and receiver truth state."""
    seed = int(cfg.rng_seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    receiver_truth = lla_to_ecef(cfg.rx_lat_deg, cfg.rx_lon_deg, cfg.rx_alt_m)
    receiver_clock = 4.2e-6
    receiver_truth_state = ReceiverTruth(
        pos_ecef_m=receiver_truth,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=receiver_clock,
        clk_drift_sps=0.0,
    )
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=seed))
    measurement_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth_state,
        cn0_zenith_dbhz=47.0,
        cn0_min_dbhz=cfg.cn0_model_min_dbhz,
        rng=rng,
    )
    attack_name = cfg.attack_name or "none"
    attacks = [] if attack_name.lower() == "none" else [create_attack(attack_name, cfg.attack_params)]
    attack_pipeline = AttackPipeline(attacks) if attacks else None
    if attack_pipeline is not None:
        attack_pipeline.reset(seed)
    integrity_checker = RaimIntegrityChecker(sim_cfg=cfg)
    conops_sm = ConopsStateMachine(default_pnt_config())
    solver = DefaultPvtSolver(cfg, receiver_truth, receiver_clock)
    engine = SimulationEngine(
        measurement_source,
        solver,
        integrity_checker,
        attack_pipeline,
        conops_sm,
    )
    return engine, receiver_truth_state


def build_epoch_log(
    *,
    t_s: float,
    step_out: dict[str, Any],
    integrity_checker: RaimIntegrityChecker,
    attack_name: str,
) -> EpochLog:
    """Build an EpochLog entry from a SimulationEngine step output."""
    sol = step_out["sol"]
    truth = step_out.get("rx_truth")
    integrity = step_out["integrity"]
    conops = step_out.get("conops")
    attack_report = step_out.get("attack_report")
    applied_count = getattr(attack_report, "applied_count", 0)
    if applied_count:
        attack_pr_bias_mean_m = attack_report.pr_bias_sum_m / applied_count
        attack_prr_bias_mean_mps = attack_report.prr_bias_sum_mps / applied_count
    else:
        attack_pr_bias_mean_m = 0.0
        attack_prr_bias_mean_mps = 0.0
    per_sv_stats = getattr(integrity_checker, "last_per_sv_stats", {})
    sim_cfg = getattr(integrity_checker, "sim_cfg", None)
    include_attack_active = bool(getattr(sim_cfg, "nis_alarm_include_attack_active", False))
    clock_drift_alarm_sps = float(getattr(sim_cfg, "clock_drift_alarm_sps", 0.0) or 0.0)
    return EpochLog(
        t=float(t_s),
        meas=step_out["meas_attacked"],
        solution=sol,
        truth=truth,
        t_s=float(t_s),
        fix_valid=sol.fix_flags.valid if sol is not None else None,
        raim_pass=sol.fix_flags.raim_passed if sol is not None else None,
        fix_type=fix_type_from_label(sol.fix_flags.fix_type) if sol is not None else None,
        sats_used=sol.fix_flags.sv_count if sol is not None else None,
        pdop=sol.dop.pdop if sol is not None else None,
        hdop=sol.dop.hdop if sol is not None else None,
        vdop=sol.dop.vdop if sol is not None else None,
        residual_rms_m=sol.residuals.rms_m if sol is not None else None,
        pos_ecef=sol.pos_ecef.copy() if sol is not None else None,
        vel_ecef=sol.vel_ecef.copy() if sol is not None and sol.vel_ecef is not None else None,
        clk_bias_s=sol.clk_bias_s if sol is not None else None,
        clk_drift_sps=sol.clk_drift_sps if sol is not None else None,
        nis=_extract_float(step_out.get("nis")),
        innov_dim=_extract_int(step_out.get("innov_dim")),
        nis_alarm=_compute_nis_alarm(
            nis=_extract_float(step_out.get("nis")),
            innov_dim=_extract_int(step_out.get("innov_dim")),
            alpha=float(getattr(integrity_checker.integrity_cfg, "chi_square_alpha", 0.01)),
            integrity_alarm=bool(integrity.is_suspect or integrity.is_invalid),
            clk_drift_sps=(sol.clk_drift_sps if sol is not None else None),
            clock_drift_alarm_sps=clock_drift_alarm_sps,
            attack_active=bool(applied_count > 0),
            include_attack_active=include_attack_active,
        ),
        attack_name=attack_name,
        attack_active=applied_count > 0,
        attack_pr_bias_mean_m=attack_pr_bias_mean_m,
        attack_prr_bias_mean_mps=attack_prr_bias_mean_mps,
        conops_status=conops.status.value if conops is not None else None,
        conops_mode5=conops.mode5.value if conops is not None else None,
        conops_reason_codes=list(conops.reason_codes) if conops is not None else [],
        integrity_p_value=integrity.p_value,
        integrity_residual_rms=integrity.residual_rms,
        integrity_num_sats_used=integrity.num_sats_used,
        integrity_excluded_sv_ids_count=len(integrity.excluded_sv_ids),
        per_sv_stats=per_sv_stats,
    )
def _extract_float(value: Any | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _extract_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None

def _compute_nis_alarm(
    *,
    nis: float | None,
    innov_dim: int | None,
    alpha: float,
    integrity_alarm: bool,
    clk_drift_sps: float | None,
    clock_drift_alarm_sps: float,
    attack_active: bool,
    include_attack_active: bool,
) -> bool:
    """Return a best-effort alarm bit without leaking ground-truth scenario state.

    Alarm can be raised by:
    - NIS statistic exceeding a chi-square threshold.
    - Integrity layer marking suspect/invalid.
    - Clock-drift anomaly (useful for common-mode clock spoofing that may not trip NIS).

    NOTE: `attack_active` is *telemetry* and must not automatically imply an alarm.
    You may optionally OR it in for backwards-compatibility via `include_attack_active`.
    """
    nis_trigger = False
    if nis is not None and innov_dim is not None and innov_dim > 0:
        try:
            threshold = float(chi2.ppf(1.0 - alpha, innov_dim))
        except Exception:
            threshold = float("inf")
        nis_trigger = bool(np.isfinite(threshold) and nis > threshold)

    clock_trigger = False
    if (
        clk_drift_sps is not None
        and np.isfinite(float(clk_drift_sps))
        and float(clock_drift_alarm_sps) > 0.0
    ):
        clock_trigger = bool(abs(float(clk_drift_sps)) >= float(clock_drift_alarm_sps))

    alarm = bool(nis_trigger or integrity_alarm or clock_trigger)
    if include_attack_active:
        alarm = bool(alarm or attack_active)
    return alarm
