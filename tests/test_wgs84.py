import numpy as np

from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import ecef_to_lla, lla_to_ecef


def test_lla_to_ecef_is_finite() -> None:
    ecef = lla_to_ecef(36.597383, -121.874300, 14.0)
    assert ecef.shape == (3,)
    assert np.all(np.isfinite(ecef))


def test_lla_ecef_roundtrip() -> None:
    lat_deg = 37.4275
    lon_deg = -122.1697
    alt_m = 30.0

    ecef = lla_to_ecef(lat_deg, lon_deg, alt_m)
    lat_rt, lon_rt, alt_rt = ecef_to_lla(*ecef)

    assert np.isclose(lat_rt, lat_deg, atol=1e-6)
    assert np.isclose(lon_rt, lon_deg, atol=1e-6)
    assert np.isclose(alt_rt, alt_m, atol=1e-3)


def test_elevation_overhead() -> None:
    lat_deg = 0.0
    lon_deg = 0.0
    alt_m = 0.0

    pos_rx = lla_to_ecef(lat_deg, lon_deg, alt_m)
    pos_sv = lla_to_ecef(lat_deg, lon_deg, 20_200_000.0)

    elev_deg, az_deg = elev_az_from_rx_sv(pos_rx, pos_sv)

    assert elev_deg > 89.9
    assert 0.0 <= az_deg <= 360.0
