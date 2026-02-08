"""Tests for step-wise runtime engine."""

from gnss_twin.config import SimConfig
from gnss_twin.models import EpochLog, PvtSolution
from gnss_twin.runtime import Engine


def test_engine_step_sequence() -> None:
    cfg = SimConfig(duration=3.0, dt=1.0)
    engine = Engine(cfg)
    engine.reset()

    times = [0.0, 1.0, 2.0]
    epoch_logs: list[EpochLog] = []
    for t in times:
        solution, diagnostics = engine.step(t)
        assert isinstance(solution, PvtSolution)
        assert diagnostics["sats_used"] >= 0
        assert diagnostics["dop"] is not None
        assert diagnostics["flags"] is not None
        assert "epoch_log" in diagnostics
        epoch_log = diagnostics["epoch_log"]
        assert isinstance(epoch_log, EpochLog)
        epoch_logs.append(epoch_log)

    assert [epoch.t for epoch in epoch_logs] == sorted(epoch.t for epoch in epoch_logs)
