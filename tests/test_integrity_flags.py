from dataclasses import replace

import numpy as np

from gnss_twin.integrity.flags import IntegrityConfig, integrity_pvt
from gnss_twin.models import GnssMeasurement
from gnss_twin.meas.pseudorange import LIGHT_SPEED_MPS, geometric_range_m
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.sat.visibility import visible_sv_states
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import lla_to_ecef


def _make_measurements(
    receiver_pos: np.ndarray,
    receiver_clk_bias_s: float,
    receiver_vel_ecef_mps: np.ndarray,
    receiver_clk_drift_sps: float,
    sv_states,
    rng: np.random.Generator,
    noise_sigma_m: float,
) -> list[GnssMeasurement]:
    measurements: list[GnssMeasurement] = []
    for state in sv_states:
        pr_m = (
            geometric_range_m(receiver_pos, state.pos_ecef_m)
            + LIGHT_SPEED_MPS * (receiver_clk_bias_s - state.clk_bias_s)
            + float(rng.normal(0.0, noise_sigma_m))
        )
        los = state.pos_ecef_m - receiver_pos
        rho = float(np.linalg.norm(los))
        los_unit = los / rho
        prr_mps = float(
            np.dot(state.vel_ecef_mps - receiver_vel_ecef_mps, los_unit)
            + LIGHT_SPEED_MPS * (receiver_clk_drift_sps - state.clk_drift_sps)
        )
        elev_deg, az_deg = elev_az_from_rx_sv(receiver_pos, state.pos_ecef_m)
        measurements.append(
            GnssMeasurement(
                sv_id=state.sv_id,
                t=state.t,
                pr_m=pr_m,
                prr_mps=prr_mps,
                sigma_pr_m=max(noise_sigma_m, 0.5),
                cn0_dbhz=45.0,
                elev_deg=elev_deg,
                az_deg=az_deg,
                flags={"healthy": True},
            )
        )
    return measurements


def test_integrity_rejects_outlier() -> None:
    rng = np.random.default_rng(1)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=11))
    receiver_pos = lla_to_ecef(37.0, -121.0, 10.0)
    sv_states = visible_sv_states(receiver_pos, constellation.get_sv_states(0.0), elevation_mask_deg=5.0)
    measurements = _make_measurements(
        receiver_pos,
        1.2e-6,
        np.zeros(3),
        0.0,
        sv_states,
        rng,
        noise_sigma_m=0.5,
    )
    target_idx = int(np.argmax([meas.elev_deg for meas in measurements]))
    measurements[target_idx] = replace(
        measurements[target_idx],
        pr_m=measurements[target_idx].pr_m + 80.0,
    )

    baseline_cfg = IntegrityConfig(max_fde_iterations=0, max_residual_m=5.0, elevation_mask_deg=0.0)
    baseline_solution, _ = integrity_pvt(
        measurements,
        sv_states,
        initial_pos_ecef_m=receiver_pos + 50.0,
        config=baseline_cfg,
    )

    fde_cfg = IntegrityConfig(max_fde_iterations=1, max_residual_m=5.0, elevation_mask_deg=0.0)
    fde_solution, _ = integrity_pvt(
        measurements,
        sv_states,
        initial_pos_ecef_m=receiver_pos + 50.0,
        config=fde_cfg,
    )

    assert measurements[target_idx].sv_id in fde_solution.fix_flags.sv_rejected
    assert fde_solution.residuals.max_m < baseline_solution.residuals.max_m


def test_integrity_pdop_threshold_invalid() -> None:
    rng = np.random.default_rng(3)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=21))
    receiver_pos = lla_to_ecef(35.0, -120.0, 5.0)
    sv_states = visible_sv_states(receiver_pos, constellation.get_sv_states(10.0), elevation_mask_deg=5.0)
    measurements = _make_measurements(
        receiver_pos,
        -2.0e-6,
        np.zeros(3),
        0.0,
        sv_states,
        rng,
        noise_sigma_m=0.8,
    )
    config = IntegrityConfig(pdop_max=0.2, gdop_max=0.2)
    solution, _ = integrity_pvt(
        measurements,
        sv_states,
        initial_pos_ecef_m=receiver_pos + 80.0,
        config=config,
    )
    assert not solution.fix_flags.valid


def test_integrity_insufficient_sv_no_fix() -> None:
    rng = np.random.default_rng(5)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=4))
    receiver_pos = lla_to_ecef(0.0, 0.0, 0.0)
    sv_states = constellation.get_sv_states(0.0)[:3]
    measurements = _make_measurements(
        receiver_pos,
        0.0,
        np.zeros(3),
        0.0,
        sv_states,
        rng,
        noise_sigma_m=0.5,
    )
    solution, _ = integrity_pvt(
        measurements,
        sv_states,
        initial_pos_ecef_m=receiver_pos,
    )
    assert solution.fix_flags.fix_type == "NO FIX"
    assert not solution.fix_flags.valid
