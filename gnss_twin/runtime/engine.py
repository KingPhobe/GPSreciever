"""Step-wise simulation engine for GNSS twin runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Any, Callable, Protocol

import numpy as np

from scipy.stats import chi2

from gnss_twin.attacks import AttackPipeline, create_attack
from gnss_twin.config import SimConfig
from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.integrity.report import IntegrityReport
from gnss_twin.integrity.raim import chi2_threshold, compute_raim
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import EpochLog, PvtSolution, ReceiverTruth, fix_type_from_label
from gnss_twin.receiver.ekf_nav import EkfNav
from gnss_twin.receiver.gating import postfit_gate, prefit_filter
from gnss_twin.receiver.tracking_state import TrackingState
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.runtime.state_machine import ConopsStateMachine
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


class _IntegrityChecker(Protocol):
    def gate(self, measurements: list[Any]) -> list[Any]: ...

    def check(
        self,
        measurements: list[Any],
        sv_states: list[Any],
        sol: Any,
    ) -> IntegrityReport: ...


class _Solver(Protocol):
    def solve(
        self,
        measurements: list[Any],
        sv_states: list[Any],
        *,
        t_s: float | None = None,
    ) -> PvtSolution | None: ...


@dataclass
class StepDiagnostics:
    sats_used: int
    residual_rms: float
    dop: object
    nis: float | None
    flags: object
    epoch_log: EpochLog


class RaimIntegrityChecker:
    """Integrity checker that runs prefit gating and RAIM-style validation."""

    def __init__(
        self,
        sim_cfg: SimConfig | None = None,
        integrity_cfg: IntegrityConfig | None = None,
    ) -> None:
        self.sim_cfg = sim_cfg or SimConfig()
        self.integrity_cfg = integrity_cfg or IntegrityConfig()
        self.tracker = SvTracker(self.integrity_cfg)
        self.last_solution: PvtSolution | None = None
        self.last_per_sv_stats: dict[str, dict[str, float]] = {}

    def gate(self, measurements: list[Any]) -> list[Any]:
        kept, _ = prefit_filter(list(measurements), self.sim_cfg)
        return kept

    def check(
        self,
        measurements: list[Any],
        sv_states: list[Any],
        sol: PvtSolution | None,
    ) -> IntegrityReport:
        initial_pos = sol.pos_ecef if sol is not None else None
        initial_clk = sol.clk_bias_s if sol is not None else 0.0
        integrity_solution, per_sv_stats = integrity_pvt(
            list(measurements),
            list(sv_states),
            initial_pos_ecef_m=initial_pos,
            initial_clk_bias_s=initial_clk,
            config=self.integrity_cfg,
            tracker=self.tracker,
        )
        self.last_solution = integrity_solution
        self.last_per_sv_stats = {key: dict(value) for key, value in per_sv_stats.items()}
        residuals_by_sv = {
            sv_id: float(stats["residual_m"])
            for sv_id, stats in per_sv_stats.items()
            if np.isfinite(stats.get("residual_m", float("nan")))
        }
        sigmas_by_sv = {
            meas.sv_id: float(meas.sigma_pr_m)
            for meas in measurements
            if hasattr(meas, "sv_id")
        }
        t_stat, dof, threshold, passed = compute_raim(
            residuals_by_sv,
            sigmas_by_sv,
            num_states=4,
            alpha=self.integrity_cfg.chi_square_alpha,
        )
        p_value = None
        if dof > 0 and np.isfinite(t_stat):
            p_value = float(1.0 - chi2.cdf(t_stat, dof))
        excluded_sv_ids = [_parse_sv_id(sv_id) for sv_id in integrity_solution.fix_flags.sv_rejected]
        excluded_sv_ids = [sv_id for sv_id in excluded_sv_ids if sv_id is not None]
        is_invalid = integrity_solution.fix_flags.fix_type == "NO FIX"
        reason_codes = []
        if not passed:
            reason_codes.append("raim_fail")
        if is_invalid:
            reason_codes.append("insufficient_sats")
        return IntegrityReport(
            chi2=t_stat if np.isfinite(t_stat) else None,
            p_value=p_value,
            residual_rms=integrity_solution.residuals.rms_m,
            num_sats_used=integrity_solution.fix_flags.sv_count,
            num_rejected=len(integrity_solution.fix_flags.sv_rejected),
            excluded_sv_ids=excluded_sv_ids,
            is_suspect=not passed,
            is_invalid=is_invalid,
            reason_codes=reason_codes or ["integrity_ok"],
        )


class SimulationEngine:
    """Composable engine for measurement-to-CONOPS runtime steps."""

    def __init__(
        self,
        meas_src: Any,
        solver: _Solver | Callable[..., PvtSolution | None],
        integrity_checker: _IntegrityChecker | Callable[..., IntegrityReport] | None,
        attack_pipeline: AttackPipeline | None,
        conops_sm: ConopsStateMachine | None,
        logger: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.meas_src = meas_src
        self.solver = solver
        self.integrity_checker = integrity_checker
        self.attack_pipeline = attack_pipeline
        self.conops_sm = conops_sm
        self.logger = logger
        self._default_gate_cfg = SimConfig()
        self._last_t_s: float | None = None

    def step(self, t_s: float) -> dict[str, Any]:
        meas_raw = list(self.meas_src.get_measurements(float(t_s)))
        sv_states = self._get_sv_states(float(t_s))
        meas_attacked, attack_report = self._apply_attacks(meas_raw, sv_states)
        gated_meas = self._gate_measurements(meas_attacked)
        sol = self._solve(gated_meas, sv_states, t_s=float(t_s))
        integrity = self._check_integrity(gated_meas, sv_states, sol)
        conops = self.conops_sm.step(float(t_s), integrity, sol, None) if self.conops_sm else None
        output = {
            "meas_raw": meas_raw,
            "meas_attacked": meas_attacked,
            "integrity": integrity,
            "sol": sol,
            "conops": conops,
            "attack_report": attack_report,
        }
        if self.logger is not None:
            self.logger(output)
        self._update_last_state(sol, float(t_s))
        return output

    def _get_sv_states(self, t_s: float) -> list[Any]:
        constellation = getattr(self.meas_src, "constellation", None)
        if constellation is not None and hasattr(constellation, "get_sv_states"):
            return list(constellation.get_sv_states(float(t_s)))
        get_states = getattr(self.meas_src, "get_sv_states", None)
        if callable(get_states):
            return list(get_states(float(t_s)))
        return []

    def _apply_attacks(
        self,
        measurements: list[Any],
        sv_states: list[Any],
    ) -> tuple[list[Any], Any | None]:
        if self.attack_pipeline is None:
            return measurements, None
        receiver_truth = getattr(self.meas_src, "receiver_truth", None)
        if not isinstance(receiver_truth, ReceiverTruth):
            return measurements, None
        return self.attack_pipeline.apply(measurements, sv_states, rx_truth=receiver_truth)

    def _gate_measurements(self, measurements: list[Any]) -> list[Any]:
        if self.integrity_checker is not None and hasattr(self.integrity_checker, "gate"):
            return list(self.integrity_checker.gate(measurements))
        gate_cfg = self._default_gate_cfg
        meas_cfg = getattr(self.meas_src, "cn0_min_dbhz", None)
        if meas_cfg is not None:
            gate_cfg = replace(gate_cfg, cn0_min_dbhz=float(meas_cfg))
        kept, _ = prefit_filter(list(measurements), gate_cfg)
        return kept

    def _solve(
        self,
        measurements: list[Any],
        sv_states: list[Any],
        *,
        t_s: float,
    ) -> PvtSolution | None:
        solver = self.solver
        if solver is None:
            return None
        if hasattr(solver, "solve"):
            return solver.solve(measurements, sv_states, t_s=t_s)
        return solver(measurements, sv_states, t_s=t_s)

    def _check_integrity(
        self,
        measurements: list[Any],
        sv_states: list[Any],
        sol: PvtSolution | None,
    ) -> IntegrityReport:
        if self.integrity_checker is None:
            return _trivial_integrity_report(measurements, sol)
        checker = self.integrity_checker
        if hasattr(checker, "check"):
            return checker.check(measurements, sv_states, sol)
        return checker(measurements, sv_states, sol)

    def _update_last_state(self, sol: PvtSolution | None, t_s: float) -> None:
        self._last_t_s = t_s


def _trivial_integrity_report(
    measurements: list[Any],
    sol: PvtSolution | None,
) -> IntegrityReport:
    if sol is None:
        return IntegrityReport(
            chi2=None,
            p_value=None,
            residual_rms=None,
            num_sats_used=len(measurements),
            num_rejected=0,
            excluded_sv_ids=[],
            is_suspect=False,
            is_invalid=True,
            reason_codes=["no_solution"],
        )
    return IntegrityReport(
        chi2=None,
        p_value=None,
        residual_rms=sol.residuals.rms_m,
        num_sats_used=sol.fix_flags.sv_count,
        num_rejected=len(sol.fix_flags.sv_rejected),
        excluded_sv_ids=[
            sv_id
            for sv_id in (_parse_sv_id(sv) for sv in sol.fix_flags.sv_rejected)
            if sv_id is not None
        ],
        is_suspect=False,
        is_invalid=sol.fix_flags.fix_type == "NO FIX",
        reason_codes=["trivial"],
    )


def _parse_sv_id(sv_id: str) -> int | None:
    digits = "".join(ch for ch in sv_id if ch.isdigit())
    if not digits:
        return None
    return int(digits)


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
