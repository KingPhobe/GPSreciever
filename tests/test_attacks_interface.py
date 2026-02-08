import numpy as np

from gnss_twin.attacks import NoOpAttack
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import GnssMeasurement, ReceiverTruth
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


def test_noop_attack_interface() -> None:
    receiver_ecef = lla_to_ecef(37.0, -122.0, 10.0)
    receiver_truth = ReceiverTruth(
        pos_ecef_m=receiver_ecef,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=0.0,
        clk_drift_sps=0.0,
    )
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=7))
    source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth,
        elevation_mask_deg=-90.0,
        attacks=[NoOpAttack()],
    )

    measurements = source.get_measurements(0.0)

    assert isinstance(measurements, list)
    assert measurements
    assert isinstance(measurements[0], GnssMeasurement)
    assert measurements[0].sv_id
    assert measurements[0].t == 0.0
    assert np.isfinite(measurements[0].pr_m)
