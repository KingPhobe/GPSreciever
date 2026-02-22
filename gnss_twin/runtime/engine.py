"""Step-wise simulation engine for GNSS twin runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
from typing import Any, Callable, Protocol

import numpy as np
from scipy.stats import chi2

from gnss_twin.attacks import AttackPipeline
from gnss_twin.config import SimConfig
from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.integrity.report import IntegrityReport
from gnss_twin.models import EpochLog, PvtSolution, ReceiverTruth
from gnss_twin.receiver.gating import prefit_filter
from gnss_twin.receiver.wls_pvt import WlsPvtResult
from gnss_twin.runtime.state_machine import ConopsStateMachine


class _IntegrityChecker(Protocol):
    def gate(self, measurements: list[Any]) -> list[Any]: ...

    def check(
        self,
        measurements: list[Any],
        sv_states: list[Any],
        sol: Any,
        *,
        precomputed_wls: WlsPvtResult | None = None,
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
        *,
        precomputed_wls: WlsPvtResult | None = None,
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
            precomputed=precomputed_wls,
        )
        self.last_solution = integrity_solution
        self.last_per_sv_stats = {key: dict(value) for key, value in per_sv_stats.items()}
        residuals_by_sv = {
            sv_id: float(stats["residual_m"])
            for sv_id, stats in per_sv_stats.items()
            if np.isfinite(stats.get("residual_m", float("nan")))
        }
        t_stat = float(integrity_solution.fix_flags.chi_square)
        threshold = float(integrity_solution.fix_flags.chi_square_threshold)
        passed = bool(integrity_solution.fix_flags.raim_passed)
        dof = max(0, len(residuals_by_sv) - 4)
        p_value = None
        if dof > 0 and np.isfinite(t_stat) and np.isfinite(threshold):
            p_value = float(1.0 - chi2.cdf(t_stat, dof))
        excluded_sv_ids = [_parse_sv_id(sv_id) for sv_id in integrity_solution.fix_flags.sv_rejected]
        excluded_sv_ids = [sv_id for sv_id in excluded_sv_ids if sv_id is not None]
        is_invalid = integrity_solution.fix_flags.fix_type == "NO FIX"
        reason_codes = []
        if not passed:
            reason_codes.append("raim_fail")
        if is_invalid:
            reason_codes.append("insufficient_sats")
        if len(excluded_sv_ids) > 0:
            reason_codes.append("sv_rejected")
        is_suspect = (not passed) or (len(excluded_sv_ids) > 0)
        return IntegrityReport(
            chi2=t_stat if np.isfinite(t_stat) else None,
            p_value=p_value,
            residual_rms=integrity_solution.residuals.rms_m,
            num_sats_used=integrity_solution.fix_flags.sv_count,
            num_rejected=len(integrity_solution.fix_flags.sv_rejected),
            excluded_sv_ids=excluded_sv_ids,
            is_suspect=is_suspect,
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
        t_s = float(t_s)
        get_batch = getattr(self.meas_src, "get_batch", None)
        if callable(get_batch):
            meas_raw, sv_states, rx_truth = get_batch(t_s)
            meas_raw = list(meas_raw)
            sv_states = list(sv_states)
        else:
            meas_raw = list(self.meas_src.get_measurements(t_s))
            sv_states = self._get_sv_states(t_s)
            rx_truth = getattr(self.meas_src, "receiver_truth", None)
        meas_attacked, attack_report = self._apply_attacks(meas_raw, sv_states)
        gated_meas = self._gate_measurements(meas_attacked)
        sol = self._solve(gated_meas, sv_states, t_s=t_s)
        integrity = self._check_integrity(gated_meas, sv_states, sol)
        conops = self.conops_sm.step(t_s, integrity, sol, None) if self.conops_sm else None
        output = {
            "meas_raw": meas_raw,
            "meas_attacked": meas_attacked,
            "integrity": integrity,
            "sol": sol,
            "conops": conops,
            "attack_report": attack_report,
            "rx_truth": rx_truth,
        }
        # Expose solver diagnostics (optional) for logging/metrics.
        output["nis"] = getattr(self.solver, "last_nis", None)
        output["innov_dim"] = getattr(self.solver, "last_innov_dim", None)
        if self.logger is not None:
            self.logger(output)
        self._update_last_state(sol, t_s)
        return output

    def _get_sv_states(self, t_s: float) -> list[Any]:
        t_s = float(t_s)
        constellation = getattr(self.meas_src, "constellation", None)
        if constellation is not None and hasattr(constellation, "get_sv_states"):
            return list(constellation.get_sv_states(t_s))
        get_states = getattr(self.meas_src, "get_sv_states", None)
        if callable(get_states):
            return list(get_states(t_s))
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
        self._sync_attack_pipeline_time(measurements)
        return _call_attack_pipeline_apply(
            self.attack_pipeline,
            measurements,
            sv_states,
            receiver_truth,
        )

    def _sync_attack_pipeline_time(self, measurements: list[Any]) -> None:
        if self.attack_pipeline is None:
            return
        t_s: float | None = None
        if measurements:
            t_s = getattr(measurements[0], "t", None)
        if t_s is None:
            return

        reset_fn = getattr(self.attack_pipeline, "reset", None)
        if self._last_t_s is not None and float(t_s) < float(self._last_t_s) and callable(reset_fn):
            reset_fn()

        step_fn = getattr(self.attack_pipeline, "step", None)
        if callable(step_fn):
            step_fn(float(t_s))

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
        precomputed_wls = _extract_wls_result(sol)
        if precomputed_wls is None:
            candidate = getattr(self.solver, "last_wls", None)
            if isinstance(candidate, WlsPvtResult):
                precomputed_wls = candidate

        if hasattr(checker, "check"):
            try:
                return checker.check(
                    measurements,
                    sv_states,
                    sol,
                    precomputed_wls=precomputed_wls,
                )
            except TypeError:
                return checker.check(measurements, sv_states, sol)
        return checker(measurements, sv_states, sol)

    def _update_last_state(self, sol: PvtSolution | None, t_s: float) -> None:
        self._last_t_s = t_s


def _extract_wls_result(sol: Any) -> WlsPvtResult | None:
    if isinstance(sol, WlsPvtResult):
        return sol
    candidate = getattr(sol, "wls_result", None)
    return candidate if isinstance(candidate, WlsPvtResult) else None


def _call_attack_pipeline_apply(
    attack_pipeline: Any,
    measurements: list[Any],
    sv_states: list[Any],
    receiver_truth: ReceiverTruth,
) -> tuple[list[Any], Any]:
    apply_fn = getattr(attack_pipeline, "apply")
    try:
        signature = inspect.signature(apply_fn)
    except (TypeError, ValueError):
        return apply_fn(measurements, sv_states, rx_truth=receiver_truth)

    params = signature.parameters
    if "rx_truth" in params:
        return apply_fn(measurements, sv_states, rx_truth=receiver_truth)

    positional_params = [
        param
        for param in params.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional_params) == 2:
        return apply_fn(measurements, receiver_truth)
    return apply_fn(measurements, sv_states, receiver_truth)


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
    """Backwards-compatible runtime wrapper around :class:`SimulationEngine`.

    The project historically exposed an `Engine` with a small API used by tests and simple scripts.
    We keep it as a thin wrapper to avoid duplicating logic across GUI/scenario runners.
    """

    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()
        self.sim_engine: SimulationEngine | None = None
        self.receiver_truth_state: ReceiverTruth | None = None
        self.attack_name = self.cfg.attack_name or "none"

    def reset(self) -> None:
        from gnss_twin.runtime.factory import build_engine_with_truth

        self.sim_engine, self.receiver_truth_state = build_engine_with_truth(self.cfg)
        self.attack_name = self.cfg.attack_name or "none"

    def step(self, t_s: float) -> tuple[PvtSolution, dict[str, Any]]:
        if self.sim_engine is None:
            self.reset()
        assert self.sim_engine is not None

        step_out = self.sim_engine.step(float(t_s))
        sol = step_out.get("sol")
        if sol is None:
            raise RuntimeError("Solver returned None; this should not happen in the default wiring.")

        from gnss_twin.runtime.factory import build_epoch_log

        epoch_log = build_epoch_log(
            t_s=float(t_s),
            step_out=step_out,
            integrity_checker=self.sim_engine.integrity_checker,
            attack_name=self.attack_name,
        )

        diagnostics = {
            "sats_used": int(epoch_log.sats_used or 0),
            "residual_rms": float(epoch_log.residual_rms_m or float("nan")),
            "dop": sol.dop,
            "nis": step_out.get("nis"),
            "flags": sol.fix_flags,
            "epoch_log": epoch_log,
        }
        return sol, diagnostics

    def run(self, start_t_s: float, end_t_s: float, dt_s: float) -> list[EpochLog]:
        epochs: list[EpochLog] = []
        t = float(start_t_s)
        while t < float(end_t_s):
            _, diag = self.step(t)
            epochs.append(diag["epoch_log"])
            t += float(dt_s)
        return epochs
