from gnss_twin.timing import HoldoverConfig, HoldoverMonitor


def test_holdover_ok_within_thresholds() -> None:
    monitor = HoldoverMonitor(HoldoverConfig(max_abs_pps_err_s=50e-6, max_time_since_ground_pps_s=2.0))

    monitor.update(t_true_s=10.0, pps_err_s=10e-6, saw_ground_pps=True)
    result = monitor.update(t_true_s=11.0, pps_err_s=40e-6, saw_ground_pps=False)

    assert result["holdover_ok"] is True
    assert result["time_since_ground_pps_s"] == 1.0
    assert result["abs_pps_err_s"] == 40e-6


def test_holdover_not_ok_when_pps_error_exceeds_threshold() -> None:
    monitor = HoldoverMonitor(HoldoverConfig(max_abs_pps_err_s=50e-6, max_time_since_ground_pps_s=2.0))

    monitor.update(t_true_s=5.0, pps_err_s=10e-6, saw_ground_pps=True)
    result = monitor.update(t_true_s=5.5, pps_err_s=100e-6, saw_ground_pps=False)

    assert result["holdover_ok"] is False
    assert result["time_since_ground_pps_s"] == 0.5
    assert result["abs_pps_err_s"] == 100e-6


def test_holdover_not_ok_when_ground_pps_too_old() -> None:
    monitor = HoldoverMonitor(HoldoverConfig(max_abs_pps_err_s=50e-6, max_time_since_ground_pps_s=2.0))

    monitor.update(t_true_s=0.0, pps_err_s=10e-6, saw_ground_pps=True)
    result = monitor.update(t_true_s=3.0, pps_err_s=20e-6, saw_ground_pps=False)

    assert result["holdover_ok"] is False
    assert result["time_since_ground_pps_s"] == 3.0
    assert result["abs_pps_err_s"] == 20e-6
