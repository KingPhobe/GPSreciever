from gnss_twin.timing.authenticator import Mode5Authenticator


def test_mode5_authenticator_latch_behavior() -> None:
    authenticator = Mode5Authenticator()

    assert authenticator.step(mode5_gate="allow", gnss_valid=True, holdover_ok=True) is True
    assert authenticator.step(mode5_gate="hold_last", gnss_valid=False, holdover_ok=True) is True
    assert authenticator.step(mode5_gate="hold_last", gnss_valid=False, holdover_ok=False) is False


def test_mode5_authenticator_deny_forces_false() -> None:
    authenticator = Mode5Authenticator(auth_bit=True)

    assert authenticator.step(mode5_gate="deny", gnss_valid=True, holdover_ok=True) is False
