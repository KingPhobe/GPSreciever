import numpy as np

from gnss_twin.attacks import JamCn0DropAttack
from gnss_twin.config import SimConfig
from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState
from gnss_twin.receiver.gating import prefit_filter


def _make_measurement(*, sv_id: str, t: float, cn0_dbhz: float, sigma_pr_m: float) -> GnssMeasurement:
    return GnssMeasurement(
        sv_id=sv_id,
        t=t,
        pr_m=20_000.0,
        prr_mps=None,
        sigma_pr_m=sigma_pr_m,
        cn0_dbhz=cn0_dbhz,
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


def test_jamming_attack_applies_cn0_drop_and_sigma_inflation() -> None:
    attack = JamCn0DropAttack(start_t=20.0, cn0_drop_db=10.0, sigma_pr_scale=5.0)
    rx_truth = _make_rx_truth()
    sv_state = _make_sv_state("G01", 25.0)

    before = _make_measurement(sv_id="G01", t=10.0, cn0_dbhz=40.0, sigma_pr_m=1.0)
    after = _make_measurement(sv_id="G01", t=25.0, cn0_dbhz=40.0, sigma_pr_m=1.0)

    unchanged, delta = attack.apply(before, sv_state, rx_truth=rx_truth)
    assert unchanged is before
    assert not delta.applied

    modified, delta = attack.apply(after, sv_state, rx_truth=rx_truth)
    assert modified.cn0_dbhz == 30.0
    assert modified.sigma_pr_m == 5.0
    assert delta.applied


def test_jamming_attack_triggers_gating_rejections() -> None:
    attack = JamCn0DropAttack(start_t=5.0, cn0_drop_db=12.0, sigma_pr_scale=6.0)
    cfg = SimConfig(cn0_min_dbhz=30.0, sigma_pr_max_m=4.0)
    rx_truth = _make_rx_truth()

    measurements = [
        _make_measurement(sv_id="G01", t=10.0, cn0_dbhz=40.0, sigma_pr_m=1.0),
        _make_measurement(sv_id="G02", t=10.0, cn0_dbhz=41.0, sigma_pr_m=1.2),
        _make_measurement(sv_id="G03", t=10.0, cn0_dbhz=39.0, sigma_pr_m=0.9),
    ]
    sv_state = _make_sv_state("G01", 10.0)

    kept_before, _ = prefit_filter(measurements, cfg)
    jammed = [attack.apply(meas, sv_state, rx_truth=rx_truth)[0] for meas in measurements]
    kept_after, _ = prefit_filter(jammed, cfg)

    assert len(kept_before) == len(measurements)
    assert len(kept_after) < len(kept_before)
