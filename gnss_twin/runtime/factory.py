"""Factories for assembling runtime simulation components."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

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
    receiver_truth_state: ReceiverTruth,
    integrity_checker: RaimIntegrityChecker,
    attack_name: str,
) -> EpochLog:
    """Build an EpochLog entry from a SimulationEngine step output."""
    sol = step_out["sol"]
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
    return EpochLog(
        t=float(t_s),
        meas=step_out["meas_attacked"],
        solution=sol,
        truth=receiver_truth_state,
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
        nis=None,
        nis_alarm=(integrity.is_suspect or integrity.is_invalid or (applied_count > 0)),
        attack_name=attack_name,
        attack_active=applied_count > 0,
        attack_pr_bias_mean_m=attack_pr_bias_mean_m,
        attack_prr_bias_mean_mps=attack_prr_bias_mean_mps,
        innov_dim=None,
        conops_status=conops.status.value if conops is not None else None,
        conops_mode5=conops.mode5.value if conops is not None else None,
        conops_reason_codes=list(conops.reason_codes) if conops is not None else [],
        integrity_p_value=integrity.p_value,
        integrity_residual_rms=integrity.residual_rms,
        integrity_num_sats_used=integrity.num_sats_used,
        integrity_excluded_sv_ids_count=len(integrity.excluded_sv_ids),
        per_sv_stats=per_sv_stats,
    )
