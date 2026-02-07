import numpy as np

from gnss_twin.meas.pseudorange import LIGHT_SPEED_MPS, geometric_range_m
from gnss_twin.models import GnssMeasurement
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.sat.visibility import visible_sv_states
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import lla_to_ecef


def _make_measurements(
    receiver_pos: np.ndarray,
    receiver_clk_bias_s: float,
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
        elev_deg, az_deg = elev_az_from_rx_sv(receiver_pos, state.pos_ecef_m)
        measurements.append(
            GnssMeasurement(
                sv_id=state.sv_id,
                t=state.t,
                pr_m=pr_m,
                prr_mps=None,
                sigma_pr_m=max(noise_sigma_m, 0.1),
                cn0_dbhz=45.0,
                elev_deg=elev_deg,
                az_deg=az_deg,
                flags={"healthy": True},
            )
        )
    return measurements


def test_wls_pvt_zero_noise() -> None:
    rng = np.random.default_rng(42)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=3))
    receiver_pos = lla_to_ecef(37.4275, -122.1697, 30.0)
    receiver_clk = 4.2e-6
    sv_states = visible_sv_states(receiver_pos, constellation.get_sv_states(0.0), elevation_mask_deg=5.0)
    measurements = _make_measurements(receiver_pos, receiver_clk, sv_states, rng, noise_sigma_m=0.0)
    solution = wls_pvt(measurements, sv_states, initial_pos_ecef_m=receiver_pos + 50.0)
    assert solution is not None
    assert np.linalg.norm(solution.pos_ecef_m - receiver_pos) < 1e-3
    assert np.isfinite(solution.dop.pdop)


def test_wls_pvt_realistic_noise() -> None:
    rng = np.random.default_rng(7)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=5))
    receiver_pos = lla_to_ecef(35.0, -120.0, 10.0)
    receiver_clk = -2.3e-6
    sv_states = visible_sv_states(receiver_pos, constellation.get_sv_states(10.0), elevation_mask_deg=5.0)
    measurements = _make_measurements(receiver_pos, receiver_clk, sv_states, rng, noise_sigma_m=3.0)
    solution = wls_pvt(measurements, sv_states, initial_pos_ecef_m=receiver_pos + 100.0)
    assert solution is not None
    assert np.linalg.norm(solution.pos_ecef_m - receiver_pos) < 10.0
    assert solution.dop.pdop > 0.0


def test_wls_pvt_handles_insufficient_sv() -> None:
    rng = np.random.default_rng(0)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=9))
    receiver_pos = lla_to_ecef(0.0, 0.0, 0.0)
    receiver_clk = 0.0
    sv_states = constellation.get_sv_states(0.0)[:3]
    measurements = _make_measurements(receiver_pos, receiver_clk, sv_states, rng, noise_sigma_m=1.0)
    assert wls_pvt(measurements, sv_states, initial_pos_ecef_m=receiver_pos) is None
