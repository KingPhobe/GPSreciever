from gnss_twin.integrity.report import IntegrityReport
from gnss_twin.runtime.conops_types import Mode5Gate, PntStatus
from gnss_twin.runtime.pnt_config import PntConfig
from gnss_twin.runtime.state_machine import ConopsStateMachine


def test_conops_state_machine_transitions() -> None:
    cfg = PntConfig(
        tta_s=1.0,
        suspect_hold_s=1.0,
        reacq_confirm_s=1.0,
        min_sats_valid=5,
        max_pdop_valid=6.0,
        residual_rms_suspect=3.0,
        residual_rms_invalid=6.0,
        chi2_p_suspect=0.05,
        chi2_p_invalid=0.01,
        clock_innov_suspect=1e3,
        clock_innov_invalid=2e3,
    )
    sm = ConopsStateMachine(cfg)

    out_nominal = sm.step(
        0.0,
        IntegrityReport(),
        {"num_sats": 6, "pdop": 1.2, "residual_rms": 1.0},
        None,
    )
    assert out_nominal.status == PntStatus.VALID
    assert out_nominal.mode5 == Mode5Gate.ALLOW

    out_suspect = sm.step(
        1.0,
        IntegrityReport(is_suspect=True),
        {"num_sats": 6, "pdop": 1.2, "residual_rms": 4.0},
        None,
    )
    assert out_suspect.status == PntStatus.SUSPECT
    assert out_suspect.mode5 == Mode5Gate.HOLD_LAST

    out_invalid = sm.step(
        2.0,
        IntegrityReport(is_suspect=True),
        {"num_sats": 6, "pdop": 1.2, "residual_rms": 4.0},
        None,
    )
    assert out_invalid.status == PntStatus.INVALID
    assert out_invalid.mode5 == Mode5Gate.DENY

    out_recovery = sm.step(
        3.0,
        IntegrityReport(),
        {"num_sats": 6, "pdop": 1.2, "residual_rms": 1.0},
        None,
    )
    assert out_recovery.status == PntStatus.INVALID
    assert out_recovery.mode5 == Mode5Gate.DENY

    out_reacquired = sm.step(
        4.0,
        IntegrityReport(),
        {"num_sats": 6, "pdop": 1.2, "residual_rms": 1.0},
        None,
    )
    assert out_reacquired.status == PntStatus.VALID
    assert out_reacquired.mode5 == Mode5Gate.ALLOW
