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


def test_authenticator_holdover_then_expires() -> None:
    cfg = AuthenticatorConfig(
        seed=7,
        sigma_process_bias_s=0.0,
        sigma_process_drift_sps=0.0,
        drift_rw_rms_sps_sqrt=5.0e-6,
        min_samples_to_lock=3,
        rms_lock_threshold_s=1.0e-3,
        rms_holdover_threshold_s=1.0e-3,
        holdover_max_s=300.0,
        holdover_max_auth_minus_ref_s=5.0e-4,
        require_ref_for_holdover_auth=True,
    )
    authenticator = Authenticator(cfg)

    t0 = 5.0
    for t_s in range(int(t0)):
        tel, _ = authenticator.step(
            t_s=float(t_s),
            pps_platform_edge_s=0.0,
            pps_ref_edge_s=0.0,
            gnss_valid=True,
        )
    assert tel.locked
    assert tel.auth_bit == 1


    tel, _ = authenticator.step(
        t_s=t0,
        pps_platform_edge_s=0.0,
        pps_ref_edge_s=0.0,
        gnss_valid=False,
    )
    auth_mode = "holdover" if tel.holdover_active else ("locked" if tel.locked else "unlocked")
    auth_reason_codes = [tel.reason_code]
    assert auth_mode == "holdover"
    assert tel.locked is True
    assert tel.auth_bit == 1

    dropped = False
    for k in range(1, 200):
        tel, _ = authenticator.step(
            t_s=t0 + float(k),
            pps_platform_edge_s=0.0,
            pps_ref_edge_s=0.0,
            gnss_valid=False,
        )
        if tel.auth_bit == 0:
            auth_reason_codes = [tel.reason_code]
            dropped = True
            break

    assert dropped is True
    assert tel.locked is False
    assert (
        "holdover_drift_exceeded" in auth_reason_codes
        or "auth_fail_holdover_drift" in auth_reason_codes
    )
