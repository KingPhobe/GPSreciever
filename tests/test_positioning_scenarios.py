import numpy as np

from gnss_twin.integrity import raim_fde
from gnss_twin.meas.models import LIGHT_SPEED, Measurement
from gnss_twin.receiver import wls_solve
from gnss_twin.sat import SyntheticOrbitModel


def _make_measurements(
    receiver_pos: np.ndarray,
    receiver_clk_bias_s: float,
    sats,
    rng: np.random.Generator,
    noise_sigma_m: float,
    bias_prn: int | None = None,
    bias_m: float = 0.0,
) -> list[Measurement]:
    measurements: list[Measurement] = []
    for sat in sats:
        truth_range = float(np.linalg.norm(sat.position_ecef_m - receiver_pos))
        noise = float(rng.normal(0.0, noise_sigma_m))
        bias = bias_m if bias_prn == sat.prn else 0.0
        pseudorange = truth_range + LIGHT_SPEED * (receiver_clk_bias_s - sat.clock_bias_s) + noise + bias
        measurements.append(
            Measurement(
                prn=sat.prn,
                pseudorange_m=pseudorange,
                truth_range_m=truth_range,
                iono_delay_m=0.0,
                tropo_delay_m=0.0,
                noise_m=0.0,
                elevation_rad=0.0,
                sat_clock_bias_s=sat.clock_bias_s,
                sat_position_ecef_m=sat.position_ecef_m,
            )
        )
    return measurements


def _residual_rms(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(residuals**2)))


def test_zero_noise_truth_case() -> None:
    rng = np.random.default_rng(0)
    orbit = SyntheticOrbitModel(num_sats=8)
    sats = orbit.get_state(0.0)
    receiver_truth = np.array([1_450_000.0, -4_100_000.0, 3_250_000.0])
    measurements = _make_measurements(receiver_truth, 0.0, sats, rng, noise_sigma_m=0.0)
    solution = wls_solve(measurements, receiver_truth + 75.0)
    position_error = np.linalg.norm(solution.position_ecef_m - receiver_truth)
    residual_rms = _residual_rms(solution.residuals_m)
    assert position_error < 1e-3
    assert residual_rms < 1e-6


def test_noise_only_case() -> None:
    rng = np.random.default_rng(1)
    orbit = SyntheticOrbitModel(num_sats=6, inclination_deg=1.0)
    sats = orbit.get_state(0.0)
    receiver_truth = np.array([1_600_000.0, -4_050_000.0, 3_300_000.0])
    measurements = _make_measurements(receiver_truth, 0.0, sats, rng, noise_sigma_m=1.0)
    solution = wls_solve(measurements, receiver_truth + 100.0)
    position_error = np.linalg.norm(solution.position_ecef_m - receiver_truth)
    residual_rms = _residual_rms(solution.residuals_m)
    assert 2.0 <= position_error <= 5.0
    assert 0.3 <= residual_rms <= 1.5


def test_single_outlier_fde_rejects_sv() -> None:
    rng = np.random.default_rng(2)
    orbit = SyntheticOrbitModel(num_sats=8)
    sats = orbit.get_state(0.0)
    receiver_truth = np.array([1_520_000.0, -4_180_000.0, 3_180_000.0])
    bias_prn = sats[0].prn
    measurements = _make_measurements(
        receiver_truth,
        0.0,
        sats,
        rng,
        noise_sigma_m=0.0,
        bias_prn=bias_prn,
        bias_m=100.0,
    )
    raw_solution = wls_solve(measurements, receiver_truth + 50.0)
    raw_rms = _residual_rms(raw_solution.residuals_m)
    fde_solution, report = raim_fde(measurements, receiver_truth + 50.0, threshold_m=10.0)
    fde_rms = _residual_rms(fde_solution.residuals_m)
    position_error = np.linalg.norm(fde_solution.position_ecef_m - receiver_truth)
    assert raw_rms > 10.0
    assert report.fde_used
    assert report.excluded_prn == bias_prn
    assert fde_rms < raw_rms
    assert position_error < 5.0
