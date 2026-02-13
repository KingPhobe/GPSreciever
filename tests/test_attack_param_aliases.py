from __future__ import annotations

from sim.live_runner import parse_kv_list
from sim.run_static_demo import _parse_attack_params
from gnss_twin.attacks import SpoofClockRampAttack, create_attack


def test_live_runner_parse_kv_list_maps_slope_mps_alias() -> None:
    params = parse_kv_list(["slope_mps=10"])
    assert params["slope_mps"] == 10
    assert params["ramp_rate_mps"] == 10


def test_static_demo_parse_attack_params_maps_slope_mps_alias() -> None:
    params = _parse_attack_params(["slope_mps=10"], "spoof_clock_ramp")
    assert params["slope_mps"] == 10.0
    assert params["ramp_rate_mps"] == 10.0


def test_create_attack_maps_slope_mps_alias() -> None:
    attack = create_attack("spoof_clock_ramp", {"slope_mps": 10})
    assert isinstance(attack, SpoofClockRampAttack)
    assert attack.ramp_rate_mps == 10.0
