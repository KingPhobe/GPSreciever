from __future__ import annotations

from sim.run_static_demo import _parse_attack_params


def test_attack_param_parsing_converts_numeric_values() -> None:
    params = _parse_attack_params(["start_t=20", "ramp_rate_mps=3"], "spoof_clock_ramp")
    assert params["start_t"] == 20.0
    assert params["ramp_rate_mps"] == 3.0


def test_attack_param_parsing_allows_target_sv_string() -> None:
    params = _parse_attack_params(["target_sv=G05"], "spoof_pr_ramp")
    assert params["target_sv"] == "G05"
