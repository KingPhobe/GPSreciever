from __future__ import annotations

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.runtime import Engine


def test_attack_telemetry_reports_spoof_biases() -> None:
    cfg = SimConfig(
        duration=3.0,
        dt=1.0,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 0.0, "ramp_rate_mps": 10.0},
    )
    engine = Engine(cfg)
    epochs = engine.run(0.0, cfg.duration, cfg.dt)

    attack_flags = [epoch.attack_active for epoch in epochs]
    assert all(attack_flags)

    pr_biases = [epoch.attack_pr_bias_mean_m for epoch in epochs]
    assert pr_biases == sorted(pr_biases)

    prr_biases = [epoch.attack_prr_bias_mean_mps for epoch in epochs]
    assert np.isclose(prr_biases[-1], 10.0, atol=1e-6)
