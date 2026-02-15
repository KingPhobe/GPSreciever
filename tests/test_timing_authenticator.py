from gnss_twin.timing.authenticator import Authenticator, AuthenticatorConfig


def test_authenticator_locks_and_sets_auth_bit_under_clean_pps() -> None:
    cfg = AuthenticatorConfig(
        seed=123,
        sigma_process_bias_s=0.0,
        sigma_process_drift_sps=0.0,
        gain_bias=0.25,
        gain_drift=0.05,
        rms_window=10,
        min_samples_to_lock=6,
        rms_lock_threshold_s=1.0e-9,
    )
    authenticator = Authenticator(cfg)

    telemetry = []
    for t_s in range(12):
        auth_tel, _ = authenticator.step(
            t_s=float(t_s),
            pps_platform_edge_s=0.0,
            pps_ref_edge_s=0.0,
            gnss_valid=True,
        )
        telemetry.append(auth_tel)

    assert any(sample.locked for sample in telemetry)
    assert telemetry[-1].locked
    assert telemetry[-1].auth_bit == 1


def test_authenticator_holdover_and_unlock_after_timeout() -> None:
    cfg = AuthenticatorConfig(
        seed=7,
        sigma_process_bias_s=0.0,
        sigma_process_drift_sps=0.0,
        min_samples_to_lock=3,
        rms_lock_threshold_s=1.0e-9,
        rms_holdover_threshold_s=1.0e-9,
        holdover_max_s=3.0,
    )
    authenticator = Authenticator(cfg)

    for t_s in range(5):
        auth_tel, _ = authenticator.step(
            t_s=float(t_s),
            pps_platform_edge_s=0.0,
            pps_ref_edge_s=0.0,
            gnss_valid=True,
        )
    assert auth_tel.locked
    assert auth_tel.auth_bit == 1

    holdover_tel, _ = authenticator.step(
        t_s=5.0,
        pps_platform_edge_s=0.0,
        pps_ref_edge_s=0.0,
        gnss_valid=False,
    )
    assert holdover_tel.holdover_active
    assert holdover_tel.reason_code == authenticator.REASON_CODES["HOLDOVER"]
    assert holdover_tel.auth_bit == 0

    unlocked_tel = holdover_tel
    for t_s in (6.0, 7.0, 8.0, 9.0):
        unlocked_tel, _ = authenticator.step(
            t_s=t_s,
            pps_platform_edge_s=0.0,
            pps_ref_edge_s=0.0,
            gnss_valid=False,
        )

    assert not unlocked_tel.locked
    assert not unlocked_tel.holdover_active
    assert unlocked_tel.auth_bit == 0
