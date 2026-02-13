import numpy as np

import warnings

from gnss_twin.attacks import AttackPipeline, SpoofClockRampAttack, SpoofPrRampAttack, create_attack
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


def test_spoof_clock_ramp_attack_respects_end_t() -> None:
    attack = SpoofClockRampAttack(start_t=10.0, end_t=20.0, ramp_rate_mps=50.0)
    rx_truth = _make_rx_truth()
    sv_state = _make_sv_state("G01", 10.0)

    before = _make_measurement(sv_id="G01", t=9.0, pr_m=1_000.0, prr_mps=0.0)
    at_start = _make_measurement(sv_id="G01", t=10.0, pr_m=1_000.0, prr_mps=0.0)
    active = _make_measurement(sv_id="G01", t=15.0, pr_m=1_000.0, prr_mps=0.0)
    after = _make_measurement(sv_id="G01", t=21.0, pr_m=1_000.0, prr_mps=0.0)

    unchanged, delta = attack.apply(before, sv_state, rx_truth=rx_truth)
    assert unchanged is before
    assert not delta.applied

    modified, delta = attack.apply(at_start, sv_state, rx_truth=rx_truth)
    assert delta.applied
    assert modified.pr_m == 1_000.0

    modified, delta = attack.apply(active, sv_state, rx_truth=rx_truth)
    assert delta.applied
    assert modified.pr_m == 1_000.0 + 250.0

    unchanged, delta = attack.apply(after, sv_state, rx_truth=rx_truth)
    assert unchanged is after
    assert not delta.applied


def test_spoof_pr_ramp_attack_respects_end_t() -> None:
    attack = SpoofPrRampAttack(start_t=10.0, end_t=20.0, ramp_rate_mps=50.0, target_sv="G02")
    rx_truth = _make_rx_truth()
    sv_state = _make_sv_state("G02", 10.0)

    before = _make_measurement(sv_id="G02", t=9.0, pr_m=1_000.0, prr_mps=0.0)
    at_start = _make_measurement(sv_id="G02", t=10.0, pr_m=1_000.0, prr_mps=0.0)
    active = _make_measurement(sv_id="G02", t=15.0, pr_m=1_000.0, prr_mps=0.0)
    after = _make_measurement(sv_id="G02", t=21.0, pr_m=1_000.0, prr_mps=0.0)

    unchanged, delta = attack.apply(before, sv_state, rx_truth=rx_truth)
    assert unchanged is before
    assert not delta.applied

    modified, delta = attack.apply(at_start, sv_state, rx_truth=rx_truth)
    assert delta.applied
    assert modified.pr_m == 1_000.0

    modified, delta = attack.apply(active, sv_state, rx_truth=rx_truth)
    assert delta.applied
    assert modified.pr_m == 1_000.0 + 250.0

    unchanged, delta = attack.apply(after, sv_state, rx_truth=rx_truth)
    assert unchanged is after
    assert not delta.applied


def test_create_attack_accepts_optional_end_t() -> None:
    clock_attack = create_attack(
        "spoof_clock_ramp", {"start_t": 10.0, "end_t": 20.0, "ramp_rate_mps": 2.5}
    )
    assert isinstance(clock_attack, SpoofClockRampAttack)
    assert clock_attack.end_t == 20.0

    pr_attack = create_attack(
        "spoof_pr_ramp",
        {"target_sv": "G02", "start_t": 10.0, "end_t": 20.0, "ramp_rate_mps": 2.5},
    )
    assert isinstance(pr_attack, SpoofPrRampAttack)
    assert pr_attack.end_t == 20.0


def test_create_attack_accepts_sv_resolution_flags() -> None:
    attack = create_attack(
        "spoof_pr_ramp",
        {
            "target_sv": "G02",
            "auto_select_visible_sv": True,
            "strict_target_sv": False,
        },
    )
    assert isinstance(attack, SpoofPrRampAttack)
    assert attack.auto_select_visible_sv is True
    assert attack.strict_target_sv is False


def test_spoof_pr_ramp_warns_once_when_target_not_visible_strict() -> None:
    attack = SpoofPrRampAttack(start_t=1.0, ramp_rate_mps=5.0, target_sv="G99", strict_target_sv=True)
    pipeline = AttackPipeline(attacks=[attack])
    rx_truth = _make_rx_truth()
    measurements = [
        _make_measurement(sv_id="G01", t=2.0, pr_m=1_000.0, prr_mps=0.0),
        _make_measurement(sv_id="G02", t=2.0, pr_m=1_001.0, prr_mps=0.0),
    ]
    sv_states = [_make_sv_state("G01", 2.0), _make_sv_state("G02", 2.0)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        attacked_1, report_1 = pipeline.apply(measurements, sv_states, rx_truth=rx_truth)
        attacked_2, report_2 = pipeline.apply(measurements, sv_states, rx_truth=rx_truth)

    assert attacked_1 == measurements
    assert attacked_2 == measurements
    assert report_1.applied_count == 0
    assert report_2.applied_count == 0
    assert len(caught) == 1
    assert "not visible" in str(caught[0].message)


def test_spoof_pr_ramp_auto_selects_visible_sv_and_applies() -> None:
    attack = SpoofPrRampAttack(
        start_t=1.0,
        ramp_rate_mps=2.0,
        target_sv="G99",
        auto_select_visible_sv=True,
    )
    pipeline = AttackPipeline(attacks=[attack])
    rx_truth = _make_rx_truth()
    measurements = [
        _make_measurement(sv_id="G03", t=3.0, pr_m=1_200.0, prr_mps=1.0),
        _make_measurement(sv_id="G01", t=3.0, pr_m=1_100.0, prr_mps=1.0),
    ]
    sv_states = [_make_sv_state("G01", 3.0), _make_sv_state("G03", 3.0)]

    attacked, report = pipeline.apply(measurements, sv_states, rx_truth=rx_truth)

    assert report.applied_count > 0
    assert attacked[1].sv_id == "G01"
    assert attacked[1].pr_m == 1_100.0 + (3.0 - 1.0) * 2.0
    assert attacked[1].prr_mps == 1.0 + 2.0
