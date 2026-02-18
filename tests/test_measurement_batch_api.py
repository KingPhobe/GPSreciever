import math

import numpy as np

from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import ReceiverTruth
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


def test_get_batch_returns_measurements_sv_states_and_truth_snapshot() -> None:
    receiver_truth = ReceiverTruth(
        pos_ecef_m=lla_to_ecef(37.0, -122.0, 10.0),
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=0.0,
        clk_drift_sps=0.0,
    )
    source = SyntheticMeasurementSource(
        constellation=SimpleGpsConstellation(SimpleGpsConfig(seed=11)),
        receiver_truth=receiver_truth,
        enable_multipath=False,
        rng=np.random.default_rng(1234),
    )

    measurements, sv_states, rx_truth_snapshot = source.get_batch(1.0)

    assert sv_states
    assert measurements
    assert math.isfinite(rx_truth_snapshot.clk_bias_s)
