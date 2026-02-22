from __future__ import annotations

import math

from sim.validation.monte_carlo import _metrics_from_epoch_dicts


def test_metrics_handles_malformed_pos_without_crashing() -> None:
    epochs = [
        {
            "t_s": 0.0,
            "solution": {"pos_ecef": [1.0, 2.0], "clk_bias_s": 0.0, "fix_flags": {}},
            "truth": {"pos_ecef_m": [1.0, 2.0, 3.0], "clk_bias_s": 0.0},
            "nis_alarm": False,
        }
    ]

    metrics = _metrics_from_epoch_dicts(epochs, attack_start_t=None)

    assert math.isnan(metrics["pos_err_rms_m"])
