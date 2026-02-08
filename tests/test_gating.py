import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.meas.pseudorange import LIGHT_SPEED_MPS, geometric_range_m
from gnss_twin.models import GnssMeasurement
from gnss_twin.receiver.gating import postfit_gate, prefit_filter
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.sat.visibility import visible_sv_states
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import lla_to_ecef


def _make_measurements_with_bias(
    receiver_pos: np.ndarray,
    receiver_clk_bias_s: float,
    sv_states,
    rng: np.random.Generator,
    *,
    noise_sigma_m: float,
    biased_sv_id: str,
    bias_m: float,
) -> list[GnssMeasurement]:
    measurements: list[GnssMeasurement] = []
    for state in sv_states:
        pr_m = (
            geometric_range_m(receiver_pos, state.pos_ecef_m)
            + LIGHT_SPEED_MPS * (receiver_clk_bias_s - state.clk_bias_s)
            + float(rng.normal(0.0, noise_sigma_m))
        )
        if state.sv_id == biased_sv_id:
            pr_m += bias_m
        elev_deg, az_deg = elev_az_from_rx_sv(receiver_pos, state.pos_ecef_m)
        measurements.append(
            GnssMeasurement(
                sv_id=state.sv_id,
                t=state.t,
                pr_m=pr_m,
                prr_mps=None,
                sigma_pr_m=max(noise_sigma_m, 1.0),
                cn0_dbhz=45.0,
                elev_deg=elev_deg,
                az_deg=az_deg,
                flags={"healthy": True},
            )
        )
    return measurements


def test_postfit_gate_excludes_biased_sv() -> None:
    rng = np.random.default_rng(21)
    cfg = SimConfig()
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=7))
    receiver_pos = lla_to_ecef(37.4275, -122.1697, 30.0)
    receiver_clk = 3.8e-6
    sv_states = visible_sv_states(receiver_pos, constellation.get_sv_states(0.0), elevation_mask_deg=5.0)
    biased_sv_id = sv_states[0].sv_id
    measurements = _make_measurements_with_bias(
        receiver_pos,
        receiver_clk,
        sv_states,
        rng,
        noise_sigma_m=1.0,
        biased_sv_id=biased_sv_id,
        bias_m=80.0,
    )
    used, _ = prefit_filter(measurements, cfg)
    solution = wls_pvt(used, sv_states, initial_pos_ecef_m=receiver_pos + 50.0)
    assert solution is not None
    sigmas_by_sv = {m.sv_id: m.sigma_pr_m for m in used}
    offender = postfit_gate(solution.residuals_m, sigmas_by_sv, gate=cfg.postfit_gate_sigma)
    assert offender == biased_sv_id

    err_before = np.linalg.norm(solution.pos_ecef_m - receiver_pos)
    filtered = [m for m in used if m.sv_id != offender]
    gated_solution = wls_pvt(filtered, sv_states, initial_pos_ecef_m=receiver_pos + 50.0)
    assert gated_solution is not None
    err_after = np.linalg.norm(gated_solution.pos_ecef_m - receiver_pos)
    assert err_after < err_before
