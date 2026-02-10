"""Rcubed CONOPS scenario: nominal -> jamming -> detect invalid -> exit -> reacquire."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gnss_twin.attacks.jamming import JamCn0DropAttack
from gnss_twin.config import SimConfig
from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.logger import save_epochs_csv, save_epochs_npz
from gnss_twin.plots.conops_plots import save_conops_plots
from gnss_twin.models import EpochLog, FixType, GnssMeasurement, PvtSolution, ReceiverTruth, fix_type_from_label
from gnss_twin.receiver.ekf_nav import EkfNav
from gnss_twin.receiver.gating import postfit_gate
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.runtime import ConopsStateMachine, RaimIntegrityChecker
from gnss_twin.runtime.pnt_config import PntConfig
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.utils.wgs84 import lla_to_ecef

# --- Scenario and attack parameters (intended to be easy to tweak) ---
RNG_SEED = 2026
DT_S = 1.0
TOTAL_DURATION_S = 120.0
ATTACK_START_S = 25.0
ATTACK_END_S = 65.0
RECOVERY_START_S = ATTACK_END_S
JAM_CN0_DROP_DB = 28.0
JAM_SIGMA_PR_SCALE = 12.0
JAM_SIGMA_PRR_SCALE = 12.0
PNT_CFG = PntConfig(
    tta_s=5.0,
    suspect_hold_s=2.0,
    reacq_confirm_s=3.0,
    min_sats_valid=5,
    max_pdop_valid=8.0,
    residual_rms_suspect=25.0,
    residual_rms_invalid=45.0,
    chi2_p_suspect=1e-6,
    chi2_p_invalid=1e-9,
    clock_innov_suspect=1e3,
    clock_innov_invalid=2e3,
)


class _DemoSolver:
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
        sv_states: list,
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


def run_scenario() -> dict[str, float | int | None]:
    cfg = SimConfig(duration=TOTAL_DURATION_S, dt=DT_S, rng_seed=RNG_SEED, use_ekf=True)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)

    receiver_lla = (37.4275, -122.1697, 30.0)
    receiver_truth = lla_to_ecef(*receiver_lla)
    receiver_clock = 4.2e-6
    receiver_truth_state = ReceiverTruth(
        pos_ecef_m=receiver_truth,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=receiver_clock,
        clk_drift_sps=0.0,
    )

    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=RNG_SEED))
    measurement_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth_state,
        cn0_zenith_dbhz=47.0,
        cn0_min_dbhz=cfg.cn0_min_dbhz,
        rng=rng,
    )

    attack = JamCn0DropAttack(
        start_t=ATTACK_START_S,
        cn0_drop_db=JAM_CN0_DROP_DB,
        sigma_pr_scale=JAM_SIGMA_PR_SCALE,
        sigma_prr_scale=JAM_SIGMA_PRR_SCALE,
    )
    integrity_checker = RaimIntegrityChecker(sim_cfg=cfg)
    conops_sm = ConopsStateMachine(PNT_CFG)
    solver = _DemoSolver(cfg, receiver_truth, receiver_clock)

    statuses: list[tuple[float, str, bool]] = []
    epochs: list[EpochLog] = []

    for t_s in np.arange(0.0, TOTAL_DURATION_S, DT_S):
        t_s = float(t_s)
        meas_raw = list(measurement_source.get_measurements(t_s))
        sv_states = list(constellation.get_sv_states(t_s))

        attack_active = ATTACK_START_S <= t_s <= ATTACK_END_S
        if attack_active:
            meas_attacked = [attack.apply(meas, sv, rx_truth=receiver_truth_state)[0] for meas, sv in zip(meas_raw, sv_states)]
            attack_name = "jam_cn0_drop"
        else:
            meas_attacked = meas_raw
            attack_name = "none"

        gated_meas = integrity_checker.gate(meas_attacked)
        sol = solver.solve(gated_meas, sv_states, t_s=t_s)
        integrity = integrity_checker.check(gated_meas, sv_states, sol)
        conops = conops_sm.step(t_s, integrity, sol, None)
        statuses.append((t_s, conops.status.value, attack_active))

        epoch = EpochLog(
            t=t_s,
            meas=meas_attacked,
            solution=sol,
            truth=receiver_truth_state,
            t_s=t_s,
            fix_valid=sol.fix_flags.valid if sol is not None else None,
            fix_type=fix_type_from_label(sol.fix_flags.fix_type) if sol is not None else FixType.NO_FIX,
            sats_used=sol.fix_flags.sv_count if sol is not None else None,
            pdop=sol.dop.pdop if sol is not None else None,
            hdop=sol.dop.hdop if sol is not None else None,
            vdop=sol.dop.vdop if sol is not None else None,
            residual_rms_m=sol.residuals.rms_m if sol is not None else None,
            pos_ecef=sol.pos_ecef.copy() if sol is not None else None,
            vel_ecef=sol.vel_ecef.copy() if sol is not None and sol.vel_ecef is not None else None,
            clk_bias_s=sol.clk_bias_s if sol is not None else None,
            clk_drift_sps=sol.clk_drift_sps if sol is not None else None,
            nis_alarm=integrity.is_suspect or integrity.is_invalid,
            attack_name=attack_name,
            attack_active=attack_active,
            conops_status=conops.status.value,
            conops_mode5=conops.mode5.value,
            conops_reason_codes=list(conops.reason_codes),
            integrity_p_value=integrity.p_value,
            integrity_residual_rms=integrity.residual_rms,
            integrity_num_sats_used=integrity.num_sats_used,
            integrity_excluded_sv_ids_count=len(integrity.excluded_sv_ids),
            per_sv_stats=integrity_checker.last_per_sv_stats,
        )
        epochs.append(epoch)

    time_to_alert = next((t for t, status, active in statuses if active and status == "invalid"), None)
    attack_end = ATTACK_END_S
    time_to_reacquire = next(
        (t for t, status, _ in statuses if t >= max(attack_end, RECOVERY_START_S) and status == "valid"),
        None,
    )
    false_alert_count = sum(1 for _, status, active in statuses if status == "invalid" and not active)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"{timestamp}_scenario_jam_then_reacquire"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_epochs_csv(out_dir / "epoch_logs.csv", epochs)
    save_epochs_npz(out_dir / "epoch_logs.npz", epochs)
    save_conops_plots(epochs, out_dir)

    summary = {
        "scenario": "jam_then_reacquire",
        "attack_start_s": ATTACK_START_S,
        "attack_end_s": ATTACK_END_S,
        "recovery_start_s": RECOVERY_START_S,
        "total_duration_s": TOTAL_DURATION_S,
        "time_to_alert_s": time_to_alert,
        "time_to_reacquire_s": time_to_reacquire,
        "false_alert_count": false_alert_count,
        "run_dir": str(out_dir),
    }

    print("=== Scenario: jam_then_reacquire ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == "__main__":
    run_scenario()
