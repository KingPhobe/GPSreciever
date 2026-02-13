import numpy as np

from gnss_twin.meas import pseudorange
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import ReceiverTruth
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


def test_receiver_lla_is_cached_once_for_static_truth(monkeypatch) -> None:
    receiver_ecef = lla_to_ecef(37.0, -122.0, 10.0)
    receiver_truth = ReceiverTruth(
        pos_ecef_m=receiver_ecef,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=0.0,
        clk_drift_sps=0.0,
    )
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=7))

    call_count = 0
    original_ecef_to_lla = pseudorange.ecef_to_lla

    def counting_ecef_to_lla(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
        nonlocal call_count
        call_count += 1
        return original_ecef_to_lla(x_m, y_m, z_m)

    monkeypatch.setattr(pseudorange, "ecef_to_lla", counting_ecef_to_lla)

    source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth,
        enable_multipath=False,
        pr_sigma_base_m=0.0,
        rng=np.random.default_rng(123),
    )

    source.get_measurements(0.0)
    source.get_measurements(1.0)
    source.get_measurements(2.0)

    assert call_count == 1
