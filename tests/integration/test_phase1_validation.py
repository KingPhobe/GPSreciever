from __future__ import annotations

import pytest

from sim.validation.phase1 import default_phase1_scenarios, run_phase1_scenario


@pytest.mark.slow
@pytest.mark.integration
def test_phase1_validation_thresholds() -> None:
    scenarios = {scenario.name: scenario for scenario in default_phase1_scenarios()}

    no_multipath = run_phase1_scenario(scenarios["No noise + no multipath (iono+tropo on)"], duration_s=30.0, seed=42)
    multipath = run_phase1_scenario(scenarios["Multipath only (noise off)"], duration_s=30.0, seed=42)

    assert no_multipath.valid_epochs == no_multipath.total_epochs
    assert no_multipath.rms_error_m < 1e-5
    assert no_multipath.max_error_m < 1e-4

    assert multipath.valid_epochs == multipath.total_epochs
    assert multipath.rms_error_m > no_multipath.rms_error_m + 0.5
    assert multipath.max_error_m > no_multipath.max_error_m + 0.5
