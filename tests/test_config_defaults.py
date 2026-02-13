from gnss_twin.config import SimConfig


def test_simconfig_receiver_defaults() -> None:
    cfg = SimConfig()

    assert cfg.rx_lat_deg == 36.597383
    assert cfg.rx_lon_deg == -121.874300
    assert cfg.rx_alt_m == 14.0
