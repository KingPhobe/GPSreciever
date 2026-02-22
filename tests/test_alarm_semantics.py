from __future__ import annotations

from gnss_twin.runtime.factory import _compute_nis_alarm


def test_nis_alarm_does_not_leak_attack_active() -> None:
    # If we have no statistical triggers and no integrity triggers, a scenario's
    # "attack_active" telemetry must NOT force an alarm.
    assert (
        _compute_nis_alarm(
            nis=None,
            innov_dim=None,
            alpha=0.01,
            integrity_alarm=False,
            clk_drift_sps=None,
            clock_drift_alarm_sps=0.0,
            attack_active=True,
            include_attack_active=False,
        )
        is False
    )
