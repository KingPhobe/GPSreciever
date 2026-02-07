import numpy as np

from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import ReceiverTruth
from gnss_twin.receiver.wls_pvt import wls_pvt
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


def test_atmospheric_correction_removes_bias() -> None:
    rng = np.random.default_rng(123)
    receiver_pos = lla_to_ecef(37.4275, -122.1697, 30.0)
    receiver_truth = ReceiverTruth(
        pos_ecef_m=receiver_pos,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=4.2e-6,
        clk_drift_sps=0.0,
    )
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=7))
    measurement_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth,
        enable_multipath=False,
        pr_sigma_base_m=0.0,
        rng=rng,
    )

    t = 0.0
    measurements = measurement_source.get_measurements(t)
    assert measurements

    lowest_elev = min(measurements, key=lambda m: m.elev_deg)
    assert lowest_elev.pr_model_corr_m > 0.0

    sv_states = constellation.get_sv_states(t)
    solution = wls_pvt(
        measurements,
        sv_states,
        initial_pos_ecef_m=receiver_pos + 100.0,
        initial_clk_bias_s=receiver_truth.clk_bias_s,
    )
    assert solution is not None
    position_error_m = float(np.linalg.norm(solution.pos_ecef_m - receiver_pos))
    assert position_error_m < 1.0
