from __future__ import annotations

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.runtime import Engine


def _run_engine(cfg: SimConfig) -> list[float]:
    engine = Engine(cfg)
    epochs = engine.run(0.0, cfg.duration, cfg.dt)
    return [
        epoch.clk_bias_s
        for epoch in epochs
        if epoch.t >= 2.0 and epoch.clk_bias_s is not None
    ]


def test_spoof_run_changes_solution_clock_bias() -> None:
    baseline_cfg = SimConfig(duration=10.0, dt=1.0, seed=123)
    spoof_cfg = SimConfig(
        duration=10.0,
        dt=1.0,
        seed=123,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 0.0, "ramp_rate_mps": 50.0},
    )

    baseline = np.array(_run_engine(baseline_cfg), dtype=float)
    spoofed = np.array(_run_engine(spoof_cfg), dtype=float)
    assert baseline.size == spoofed.size

    mean_abs_diff = float(np.mean(np.abs(spoofed - baseline)))
    assert mean_abs_diff > 1e-9
