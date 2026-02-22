from __future__ import annotations

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.runtime import Engine


def test_conops_baseline_is_not_mostly_deny() -> None:
    # Baseline expectation for operator docs: Mode-5 gating should not sit in DENY
    # for the majority of a nominal run.
    cfg = SimConfig(duration=30.0, dt=1.0, rng_seed=42, use_ekf=True)
    engine = Engine(cfg)
    epochs = engine.run(0.0, cfg.duration, cfg.dt)

    gates = [str(epoch.conops_mode5 or "") for epoch in epochs]
    deny_rate = float(np.mean([1.0 if g.lower() == "deny" else 0.0 for g in gates]))
    assert deny_rate <= 0.2
