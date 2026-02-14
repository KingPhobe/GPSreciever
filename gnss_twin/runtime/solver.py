"""Default PVT solver implementation used by runtime engines."""

from __future__ import annotations

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.models import GnssMeasurement, PvtSolution, SvState
from gnss_twin.receiver.ekf_nav import EkfNav
from gnss_twin.receiver.gating import postfit_gate
from gnss_twin.receiver.wls_pvt import wls_pvt


class DefaultPvtSolver:
    """Default runtime PVT solver with optional EKF smoothing."""

    def __init__(self, cfg: SimConfig, initial_pos: np.ndarray, initial_clk: float) -> None:
        self.cfg = cfg
        self.integrity_cfg = IntegrityConfig()
        self.tracker = SvTracker(self.integrity_cfg)
        self.last_pos = initial_pos + 100.0
        self.last_clk = float(initial_clk)
        self.last_t_s: float | None = None
        self.ekf = EkfNav() if cfg.use_ekf else None

    def solve(
        self,
        measurements: list[GnssMeasurement],
        sv_states: list[SvState],
        *,
        t_s: float | None = None,
    ) -> PvtSolution | None:
        used_meas = list(measurements)
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

        solution, _ = integrity_pvt(
            used_meas,
            sv_states,
            initial_pos_ecef_m=self.last_pos,
            initial_clk_bias_s=self.last_clk,
            config=self.integrity_cfg,
            tracker=self.tracker,
        )

        if self.ekf is not None:
            dt = self.cfg.dt if self.last_t_s is None or t_s is None else float(t_s - self.last_t_s)
            if not self.ekf.initialized and wls_solution is not None:
                self.ekf.initialize_from_wls(wls_solution)
            if self.ekf.initialized and self.last_t_s is not None and t_s is not None:
                self.ekf.predict(dt)
            if self.ekf.initialized:
                self.ekf.update_pseudorange(
                    used_meas,
                    sv_states,
                    initial_pos_ecef_m=self.last_pos,
                    initial_clk_bias_s=self.last_clk,
                )
                self.ekf.update_prr(used_meas, sv_states)
                solution = PvtSolution(
                    pos_ecef=self.ekf.pos_ecef_m.copy(),
                    vel_ecef=self.ekf.vel_ecef_mps.copy(),
                    clk_bias_s=self.ekf.clk_bias_s,
                    clk_drift_sps=self.ekf.clk_drift_sps,
                    dop=solution.dop,
                    residuals=solution.residuals,
                    fix_flags=solution.fix_flags,
                )

        self.last_t_s = t_s
        if solution.fix_flags.fix_type != "NO FIX" and np.isfinite(solution.pos_ecef).all():
            self.last_pos = solution.pos_ecef
            self.last_clk = solution.clk_bias_s
        return solution
