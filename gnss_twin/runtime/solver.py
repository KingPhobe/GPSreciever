"""Default PVT solver implementation used by runtime engines.

Runtime-facing wrapper that:
  * runs WLS PVT
  * optional post-fit residual gating (single-SV removal)
  * RAIM/integrity (via integrity_pvt)
  * optional EKF smoothing + exposes diagnostics (NIS / innovation dimension)

The SimulationEngine expects the solver instance to expose:
  - last_wls
  - last_nis
  - last_innov_dim
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.models import GnssMeasurement, PvtSolution, SvState
from gnss_twin.receiver.ekf_nav import EkfNav
from gnss_twin.receiver.gating import postfit_gate
from gnss_twin.receiver.wls_pvt import WlsPvtResult, wls_pvt


class DefaultPvtSolver:
    """Default runtime PVT solver with optional EKF smoothing."""

    def __init__(self, cfg: SimConfig, initial_pos: np.ndarray, initial_clk: float) -> None:
        self.cfg = cfg
        self.integrity_cfg = IntegrityConfig()
        self.tracker = SvTracker(self.integrity_cfg)

        # Start slightly off from truth to avoid trivial convergence in demos.
        self.last_pos = np.array(initial_pos, dtype=float) + 100.0
        self.last_clk = float(initial_clk)
        self.last_t_s: float | None = None

        self.ekf: EkfNav | None = EkfNav() if cfg.use_ekf else None

        # Cached per-epoch diagnostics.
        self.last_wls: Optional[WlsPvtResult] = None
        self.last_nis: float | None = None
        self.last_innov_dim: int | None = None

    def solve(
        self,
        measurements: list[GnssMeasurement],
        sv_states: list[SvState],
        *,
        t_s: float | None = None,
    ) -> PvtSolution | None:
        # Reset per-epoch cached diagnostics.
        self.last_wls = None
        self.last_nis = None
        self.last_innov_dim = None

        used_meas = list(measurements)

        elev_mask = float(self.integrity_cfg.elevation_mask_deg)
        masked_meas = [m for m in used_meas if float(m.elev_deg) >= elev_mask]

        wls_solution: WlsPvtResult | None = None
        if len(masked_meas) >= 4:
            wls_solution = wls_pvt(
                masked_meas,
                sv_states,
                initial_pos_ecef_m=self.last_pos,
                initial_clk_bias_s=self.last_clk,
            )
            if wls_solution is not None:
                sigmas_by_sv = {m.sv_id: float(m.sigma_pr_m) for m in masked_meas}
                offender = postfit_gate(
                    wls_solution.residuals_m,
                    sigmas_by_sv,
                    gate=float(self.cfg.postfit_gate_sigma),
                )
                if offender:
                    used_meas = [m for m in used_meas if m.sv_id != offender]
                    masked_meas = [m for m in masked_meas if m.sv_id != offender]
                    wls_solution = None
                    if len(masked_meas) >= 4:
                        wls_solution = wls_pvt(
                            masked_meas,
                            sv_states,
                            initial_pos_ecef_m=self.last_pos,
                            initial_clk_bias_s=self.last_clk,
                        )

        self.last_wls = wls_solution

        solution, _ = integrity_pvt(
            used_meas,
            sv_states,
            initial_pos_ecef_m=self.last_pos,
            initial_clk_bias_s=self.last_clk,
            config=self.integrity_cfg,
            tracker=self.tracker,
            precomputed=wls_solution,
        )
        if self.ekf is not None:
            dt = float(self.cfg.dt)
            if self.last_t_s is not None and t_s is not None:
                dt = max(0.0, float(t_s - self.last_t_s))

            if not self.ekf.initialized and wls_solution is not None:
                self.ekf.initialize_from_wls(wls_solution)

            if self.ekf.initialized and self.last_t_s is not None and t_s is not None:
                self.ekf.predict(dt)

            if self.ekf.initialized:
                self.ekf.update_pseudorange(
                    masked_meas,
                    sv_states,
                    initial_pos_ecef_m=self.last_pos,
                    initial_clk_bias_s=self.last_clk,
                )
                self.ekf.update_prr(masked_meas, sv_states)

                self.last_nis = self.ekf.last_nis
                self.last_innov_dim = self.ekf.last_innov_dim

                solution = PvtSolution(
                    pos_ecef=self.ekf.pos_ecef_m.copy(),
                    vel_ecef=self.ekf.vel_ecef_mps.copy(),
                    clk_bias_s=float(self.ekf.clk_bias_s),
                    clk_drift_sps=float(self.ekf.clk_drift_sps),
                    dop=solution.dop,
                    residuals=solution.residuals,
                    fix_flags=solution.fix_flags,
                )

        self.last_t_s = t_s
        if solution.fix_flags.fix_type != "NO FIX" and np.isfinite(solution.pos_ecef).all():
            self.last_pos = solution.pos_ecef
            self.last_clk = float(solution.clk_bias_s)

        return solution
