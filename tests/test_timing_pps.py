from gnss_twin.timing import PpsPulse, PpsSource, pps_error_s


def test_pps_error_uses_local_timestamps() -> None:
    t_true_s = 0.0
    ground_pulse = PpsPulse(
        t_true_s=t_true_s,
        t_local_s=t_true_s,
        source=PpsSource.GROUND,
    )
    rx_pulse = PpsPulse(
        t_true_s=t_true_s,
        t_local_s=t_true_s + 1e-3,
        source=PpsSource.RECEIVER,
    )

    assert pps_error_s(rx_pulse, ground_pulse) == 1e-3
