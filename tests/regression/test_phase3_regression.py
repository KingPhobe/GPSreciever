from __future__ import annotations

import pytest

from gnss_twin.logger import EPOCH_CSV_COLUMNS
from sim.validation.phase3 import has_required_columns, run_phase3_scenarios


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.regression
def test_phase3_attack_regression(tmp_path) -> None:
    results = run_phase3_scenarios(run_root=tmp_path / "phase3", duration_s=20.0, seed=123)

    baseline = results["baseline_ekf"]
    spoof = results["spoof_ekf"]
    jam = results["jam_ekf"]

    for result in results.values():
        assert has_required_columns(result.rows)
        assert set(result.rows[0].keys()) == set(EPOCH_CSV_COLUMNS)

    spoof_worse = (
        spoof.metrics["nis_alarm_rate"] > baseline.metrics["nis_alarm_rate"]
        or spoof.metrics["position_rms_m"] > baseline.metrics["position_rms_m"]
    )
    jam_worse = (
        jam.metrics["nis_alarm_rate"] > baseline.metrics["nis_alarm_rate"]
        or jam.metrics["position_rms_m"] > baseline.metrics["position_rms_m"]
    )
    assert spoof_worse or jam_worse
