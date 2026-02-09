from __future__ import annotations

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.runtime import Engine


def _run_jam_epochs() -> list:
    cfg = SimConfig(
        duration=20.0,
        dt=1.0,
        rng_seed=123,
        attack_name="jam_cn0_drop",
        attack_params={"start_t": 0.0, "cn0_drop_db": 50.0, "sigma_pr_scale": 50.0},
    )
    engine = Engine(cfg)
    return engine.run(0.0, cfg.duration, cfg.dt)


def test_no_fix_outputs_nan_dop_and_state() -> None:
    epochs = _run_jam_epochs()
    no_fix_epochs = [
        epoch
        for epoch in epochs
        if (epoch.fix_valid is False)
        or (epoch.sats_used is not None and epoch.sats_used < 4)
    ]
    assert no_fix_epochs, "Expected at least one no-fix epoch under aggressive jamming."

    for epoch in no_fix_epochs:
        solution = epoch.solution
        assert solution is not None
        dop = solution.dop
        assert np.isnan(dop.pdop)
        assert np.isnan(dop.gdop)
        assert np.isnan(dop.hdop)
        assert np.isnan(dop.vdop)
        assert np.isnan(solution.pos_ecef).all()
        assert np.isnan(solution.clk_bias_s)

    fix_epochs = [epoch for epoch in epochs if epoch.fix_valid]
    for epoch in fix_epochs:
        solution = epoch.solution
        assert solution is not None
        dop = solution.dop
        assert np.isfinite(dop.pdop)
        assert np.isfinite(dop.gdop)
