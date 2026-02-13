from __future__ import annotations

from gnss_twin.config import SimConfig
from sim.live_runner import parse_kv_list, run_headless


def test_headless_spoof_clock_ramp_applies_and_respects_window() -> None:
    cfg = SimConfig(
        duration=30.0,
        dt=1.0,
        use_ekf=True,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 10.0, "end_t": 20.0, "ramp_rate_mps": 10.0},
    )

    logs = run_headless(cfg)

    assert any(10.0 <= row["t_s"] <= 20.0 and row["attack_active"] for row in logs)
    assert any(row["t_s"] < 10.0 and not row["attack_active"] for row in logs)
    assert any(row["t_s"] > 20.0 and not row["attack_active"] for row in logs)

    pre_window = [row["attack_pr_bias_mean_m"] for row in logs if row["t_s"] < 10.0]
    assert all(abs(float(v)) < 1e-9 for v in pre_window)

    ramp_window = [row for row in logs if 10.0 <= row["t_s"] <= 20.0]
    assert max(float(row["attack_pr_bias_mean_m"]) for row in ramp_window) > 50.0


def test_headless_spoof_pr_ramp_applies_with_auto_select_visible_sv() -> None:
    cfg = SimConfig(
        duration=30.0,
        dt=1.0,
        use_ekf=True,
        attack_name="spoof_pr_ramp",
        attack_params={
            "start_t": 10.0,
            "end_t": 20.0,
            "ramp_rate_mps": 10.0,
            "target_sv": "G99",
            "auto_select_visible_sv": True,
            "strict_target_sv": False,
        },
    )

    logs = run_headless(cfg)

    assert any(10.0 <= row["t_s"] <= 20.0 and row["attack_active"] for row in logs)
    assert any(row["t_s"] < 10.0 and not row["attack_active"] for row in logs)
    assert any(row["t_s"] > 20.0 and not row["attack_active"] for row in logs)

    ramp_window = [row for row in logs if 10.0 <= row["t_s"] <= 20.0]
    assert max(float(row["attack_pr_bias_mean_m"]) for row in ramp_window) > 50.0


def test_parse_kv_list_maps_slope_mps_alias_for_cli_examples() -> None:
    params = parse_kv_list(["target_sv=G12", "start_t=10", "end_t=20", "slope_mps=10"])

    assert params["target_sv"] == "G12"
    assert params["slope_mps"] == 10
    assert params["ramp_rate_mps"] == 10
