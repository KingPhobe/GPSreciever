import numpy as np

from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import ReceiverTruth
from gnss_twin.receiver.ekf_nav import EkfNav
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


def _run_static_sim(duration_s: float = 60.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=7))
    receiver_pos = lla_to_ecef(37.4275, -122.1697, 30.0)
    receiver_truth = ReceiverTruth(
        pos_ecef_m=receiver_pos,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=3.1e-6,
        clk_drift_sps=0.0,
    )
    meas_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth,
        rng=rng,
        enable_multipath=False,
        pr_sigma_base_m=1.2,
        prr_sigma_mps=0.1,
    )
    ekf = EkfNav()
    times = np.arange(0.0, duration_s, 1.0)
    wls_errors: list[float] = []
    ekf_errors: list[float] = []
    ekf_speeds: list[float] = []
    last_pos = receiver_pos + 75.0
    last_clk = receiver_truth.clk_bias_s
    for idx, t in enumerate(times):
        sv_states = constellation.get_sv_states(float(t))
        meas = meas_source.get_measurements(float(t))
        wls_solution = wls_pvt(meas, sv_states, initial_pos_ecef_m=last_pos, initial_clk_bias_s=last_clk)
        if wls_solution is not None:
            last_pos = wls_solution.pos_ecef_m
            last_clk = wls_solution.clk_bias_s
            wls_errors.append(float(np.linalg.norm(wls_solution.pos_ecef_m - receiver_pos)))
        else:
            wls_errors.append(float("nan"))

        if not ekf.initialized and wls_solution is not None:
            ekf.initialize_from_wls(wls_solution)
        if ekf.initialized:
            if idx > 0:
                ekf.predict(1.0)
            ekf.update_pseudorange(meas, sv_states, initial_pos_ecef_m=last_pos, initial_clk_bias_s=last_clk)
            ekf.update_prr(meas, sv_states)
            ekf_errors.append(float(np.linalg.norm(ekf.pos_ecef_m - receiver_pos)))
            ekf_speeds.append(float(np.linalg.norm(ekf.vel_ecef_mps)))
        else:
            ekf_errors.append(float("nan"))
            ekf_speeds.append(float("nan"))
    return np.array(wls_errors), np.array(ekf_errors), np.array(ekf_speeds)


def test_ekf_static_speed() -> None:
    _, _, ekf_speeds = _run_static_sim()
    mean_speed = float(np.nanmean(ekf_speeds[5:]))
    assert mean_speed < 0.2


def test_ekf_rms_better_than_wls() -> None:
    wls_errors, ekf_errors, _ = _run_static_sim()
    wls_rms = float(np.sqrt(np.nanmean(np.square(wls_errors))))
    ekf_rms = float(np.sqrt(np.nanmean(np.square(ekf_errors))))
    assert ekf_rms < wls_rms
