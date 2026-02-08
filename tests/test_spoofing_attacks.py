import numpy as np

from gnss_twin.attacks import SpoofClockRampAttack, SpoofPrRampAttack, create_attack
from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState


def _make_measurement(*, sv_id: str, t: float, pr_m: float, prr_mps: float | None) -> GnssMeasurement:
    return GnssMeasurement(
        sv_id=sv_id,
        t=t,
        pr_m=pr_m,
        prr_mps=prr_mps,
        sigma_pr_m=1.0,
        cn0_dbhz=45.0,
        elev_deg=20.0,
        az_deg=180.0,
        flags={"valid": True},
    )


def _make_sv_state(sv_id: str, t: float) -> SvState:
    return SvState(
        sv_id=sv_id,
        t=t,
        pos_ecef_m=np.array([15_600_000.0, 0.0, 21_400_000.0]),
        vel_ecef_mps=np.array([0.0, 2_600.0, 1_200.0]),
        clk_bias_s=0.0,
        clk_drift_sps=0.0,
    )


def _make_rx_truth() -> ReceiverTruth:
    return ReceiverTruth(
        pos_ecef_m=np.zeros(3),
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=0.0,
        clk_drift_sps=0.0,
    )


def test_spoof_clock_ramp_attack() -> None:
    attack = SpoofClockRampAttack(start_t=20.0, ramp_rate_mps=2.0)
    rx_truth = _make_rx_truth()
    sv_state = _make_sv_state("G01", 10.0)

    before = _make_measurement(sv_id="G01", t=10.0, pr_m=10_000.0, prr_mps=1.5)
    after = _make_measurement(sv_id="G01", t=25.0, pr_m=10_000.0, prr_mps=1.5)

    unchanged, delta = attack.apply(before, sv_state, rx_truth=rx_truth)
    assert unchanged is before
    assert not delta.applied

    modified, delta = attack.apply(after, sv_state, rx_truth=rx_truth)
    assert modified.pr_m == 10_000.0 + (25.0 - 20.0) * 2.0
    assert modified.prr_mps == 1.5 + 2.0
    assert delta.applied
    assert delta.pr_bias_m == (25.0 - 20.0) * 2.0


def test_spoof_pr_ramp_attack_targets_single_sv() -> None:
    attack = SpoofPrRampAttack(start_t=20.0, ramp_rate_mps=3.0, target_sv="G02")
    rx_truth = _make_rx_truth()
    sv_state = _make_sv_state("G02", 30.0)

    target = _make_measurement(sv_id="G02", t=30.0, pr_m=20_000.0, prr_mps=0.5)
    other = _make_measurement(sv_id="G01", t=30.0, pr_m=20_000.0, prr_mps=0.5)

    modified, delta = attack.apply(target, sv_state, rx_truth=rx_truth)
    assert modified.pr_m == 20_000.0 + (30.0 - 20.0) * 3.0
    assert modified.prr_mps == 0.5 + 3.0
    assert delta.applied

    untouched, delta = attack.apply(other, sv_state, rx_truth=rx_truth)
    assert untouched is other
    assert not delta.applied


def test_create_attack_requires_target_sv() -> None:
    try:
        create_attack("spoof_pr_ramp", {})
    except ValueError as exc:
        assert "target_sv" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing target_sv")


def test_create_attack_returns_spoof_clock_ramp() -> None:
    attack = create_attack("spoof_clock_ramp", {"start_t": 10.0, "ramp_rate_mps": 2.5})
    assert isinstance(attack, SpoofClockRampAttack)
