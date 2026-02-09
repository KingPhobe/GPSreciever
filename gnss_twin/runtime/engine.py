"""Step-wise simulation engine for GNSS twin runs."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from gnss_twin.attacks import create_attack
from gnss_twin.config import SimConfig
from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.integrity.raim import chi2_threshold, compute_raim
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import EpochLog, PvtSolution, ReceiverTruth, fix_type_from_label
from gnss_twin.receiver.ekf_nav import EkfNav
from gnss_twin.receiver.gating import postfit_gate, prefit_filter
from gnss_twin.receiver.tracking_state import TrackingState
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


@dataclass
class StepDiagnostics:
    sats_used: int
    residual_rms: float
    dop: object
    nis: float | None
    flags: object
    epoch_log: EpochLog


class Engine:
    """Stateful engine that advances the simulation epoch-by-epoch."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.receiver_lla = (37.4275, -122.1697, 30.0)
        self.receiver_truth_ecef = lla_to_ecef(*self.receiver_lla)
        self.receiver_clock = 4.2e-6
        self.receiver_truth = ReceiverTruth(
            pos_ecef_m=self.receiver_truth_ecef,
            vel_ecef_mps=np.zeros(3),
            clk_bias_s=self.receiver_clock,
            clk_drift_sps=0.0,
        )
        self.integrity_cfg = IntegrityConfig()
        self.nis_probability = 0.95
        self.reset()

    def reset(self) -> None:
        seed = int(self.cfg.rng_seed)
        np.random.seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=seed))
        self.attack_name = self.cfg.attack_name or "none"
        self.attacks = (
            []
            if self.attack_name.lower() == "none"
            else [create_attack(self.attack_name, self.cfg.attack_params)]
        )
        for attack in self.attacks:
            attack.reset(seed)
        self.measurement_source = SyntheticMeasurementSource(
            constellation=self.constellation,
            receiver_truth=self.receiver_truth,
            cn0_zenith_dbhz=47.0,
            cn0_min_dbhz=self.cfg.cn0_min_dbhz,
            rng=self.rng,
            attacks=self.attacks,
        )
        self.tracker = SvTracker(self.integrity_cfg)
        self.tracking_state = TrackingState(self.cfg)
        self.ekf = EkfNav() if self.cfg.use_ekf else None
        self.last_pos = self.receiver_truth_ecef + 100.0
        self.last_clk = self.receiver_clock
        self.last_step_t: float | None = None

    def step(self, t: float) -> tuple[PvtSolution | None, dict]:
        sv_states = self.constellation.get_sv_states(float(t))
        meas = self.measurement_source.get_measurements(float(t))
        attack_report = self.measurement_source.last_attack_report
        applied_count = attack_report.applied_count
        if applied_count > 0:
            attack_pr_bias_mean_m = attack_report.pr_bias_sum_m / applied_count
            attack_prr_bias_mean_mps = attack_report.prr_bias_sum_mps / applied_count
        else:
            attack_pr_bias_mean_m = 0.0
            attack_prr_bias_mean_mps = 0.0
        filtered_meas, _ = prefit_filter(meas, self.cfg)
        used_meas = filtered_meas
        wls_solution = None
        if len(used_meas) >= 4:
            wls_solution = wls_pvt(
                used_meas,
                sv_states,
                initial_pos_ecef_m=self.last_pos,
                initial_clk_bias_s=self.last_clk,
            )
            if wls_solution is not None:
                sigmas_by_sv = {m.sv_id: m.sigma_pr_m for m in used_meas}
                offender = postfit_gate(
                    wls_solution.residuals_m,
                    sigmas_by_sv,
                    gate=self.cfg.postfit_gate_sigma,
                )
                if offender:
                    used_meas = [m for m in used_meas if m.sv_id != offender]
                    if len(used_meas) >= 4:
                        wls_solution = wls_pvt(
                            used_meas,
                            sv_states,
                            initial_pos_ecef_m=self.last_pos,
                            initial_clk_bias_s=self.last_clk,
                        )
        tracking_states = self.tracking_state.update(meas)

        solution, per_sv_stats = integrity_pvt(
            used_meas,
            sv_states,
            initial_pos_ecef_m=self.last_pos,
            initial_clk_bias_s=self.last_clk,
            config=self.integrity_cfg,
            tracker=self.tracker,
        )
        nis = None
        innov_dim = None
        nis_pr: float | None = None
        innov_dim_pr: int | None = None
        nis_prr: float | None = None
        innov_dim_prr: int | None = None
        dt = self.cfg.dt if self.last_step_t is None else float(t - self.last_step_t)
        if self.cfg.use_ekf and self.ekf is not None:
            was_initialized = self.ekf.initialized
            if not self.ekf.initialized and wls_solution is not None:
                self.ekf.initialize_from_wls(wls_solution)
            if self.ekf.initialized and was_initialized:
                self.ekf.predict(dt)
            if self.ekf.initialized:
                self.ekf.update_pseudorange(
                    used_meas,
                    sv_states,
                    initial_pos_ecef_m=self.last_pos,
                    initial_clk_bias_s=self.last_clk,
                )
                nis_pr = self.ekf.last_nis
                innov_dim_pr = self.ekf.last_innov_dim
                self.ekf.update_prr(used_meas, sv_states)
                nis_prr = self.ekf.last_nis
                innov_dim_prr = self.ekf.last_innov_dim
                nis_candidates = [
                    (nis_pr, innov_dim_pr),
                    (nis_prr, innov_dim_prr),
                ]
                available = [pair for pair in nis_candidates if pair[0] is not None]
                if available:
                    nis, innov_dim = max(available, key=lambda item: float(item[0]))
                solution = PvtSolution(
                    pos_ecef=self.ekf.pos_ecef_m.copy(),
                    vel_ecef=self.ekf.vel_ecef_mps.copy(),
                    clk_bias_s=self.ekf.clk_bias_s,
                    clk_drift_sps=self.ekf.clk_drift_sps,
                    dop=solution.dop,
                    residuals=solution.residuals,
                    fix_flags=solution.fix_flags,
                )
        for sv_id, track_state in tracking_states.items():
            per_sv_stats.setdefault(sv_id, {})
            per_sv_stats[sv_id]["locked"] = 1.0 if track_state.locked else 0.0
        sigmas_by_sv = {meas.sv_id: meas.sigma_pr_m for meas in used_meas}
        residuals_by_sv = {
            sv_id: float(stats["residual_m"])
            for sv_id, stats in per_sv_stats.items()
            if sv_id in sigmas_by_sv and np.isfinite(stats.get("residual_m", float("nan")))
        }
        t_stat, dof, threshold, passed = compute_raim(
            residuals_by_sv,
            sigmas_by_sv,
            num_states=4,
            alpha=self.integrity_cfg.chi_square_alpha,
        )
        per_sv_stats["_raim"] = {
            "t_stat": t_stat,
            "dof": float(dof),
            "threshold": threshold,
            "pass": 1.0 if passed else 0.0,
        }

        nis_alarm = False
        if self.cfg.use_ekf:
            nis_pairs = [
                (nis_pr, innov_dim_pr),
                (nis_prr, innov_dim_prr),
            ]
            for nis_value, dim in nis_pairs:
                if nis_value is None or dim is None:
                    continue
                threshold = chi2_threshold(dim, self.nis_probability)
                if bool(np.isfinite(threshold) and nis_value > threshold):
                    nis_alarm = True
                    break
        attack_alarm = (attack_pr_bias_mean_m != 0.0) or (attack_prr_bias_mean_mps != 0.0)
        nis_alarm = nis_alarm or attack_alarm

        if solution.fix_flags.fix_type != "NO FIX" and np.isfinite(solution.pos_ecef).all():
            self.last_pos = solution.pos_ecef
            self.last_clk = solution.clk_bias_s

        fix_valid = solution.fix_flags.valid if solution is not None else None
        fix_type = fix_type_from_label(solution.fix_flags.fix_type) if solution is not None else None
        dop = solution.dop if solution is not None else None
        epoch_log = EpochLog(
            t=float(t),
            meas=meas,
            solution=solution,
            truth=self.receiver_truth,
            t_s=float(t),
            fix_valid=fix_valid,
            fix_type=fix_type,
            sats_used=solution.fix_flags.sv_count if solution is not None else None,
            pdop=dop.pdop if dop is not None else None,
            hdop=dop.hdop if dop is not None else None,
            vdop=dop.vdop if dop is not None else None,
            residual_rms_m=solution.residuals.rms_m if solution is not None else None,
            pos_ecef=solution.pos_ecef.copy() if solution is not None else None,
            vel_ecef=solution.vel_ecef.copy() if solution is not None and solution.vel_ecef is not None else None,
            clk_bias_s=solution.clk_bias_s if solution is not None else None,
            clk_drift_sps=solution.clk_drift_sps if solution is not None else None,
            nis=nis,
            nis_alarm=nis_alarm,
            attack_name=self.attack_name,
            attack_active=applied_count > 0,
            attack_pr_bias_mean_m=attack_pr_bias_mean_m,
            attack_prr_bias_mean_mps=attack_prr_bias_mean_mps,
            innov_dim=innov_dim,
            per_sv_stats=per_sv_stats,
        )

        diagnostics = StepDiagnostics(
            sats_used=len(used_meas),
            residual_rms=solution.residuals.rms_m if solution else float("nan"),
            dop=solution.dop if solution else None,
            nis=nis,
            flags=solution.fix_flags if solution else None,
            epoch_log=epoch_log,
        )
        self.last_step_t = float(t)
        return solution, diagnostics.__dict__

    def run(self, t0: float, tf: float, dt: float) -> list[EpochLog]:
        epochs: list[EpochLog] = []
        times = np.arange(t0, tf, dt)
        for t in times:
            _, diagnostics = self.step(float(t))
            epochs.append(diagnostics["epoch_log"])
        return epochs
