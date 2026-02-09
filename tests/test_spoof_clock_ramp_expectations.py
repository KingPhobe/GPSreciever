from __future__ import annotations

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.runtime import Engine


def _run_epochs(cfg: SimConfig) -> list:
    engine = Engine(cfg)
    return engine.run(0.0, cfg.duration, cfg.dt)


def _dop_by_time(epochs: list) -> dict[float, tuple[float, float]]:
    return {
        epoch.t: (epoch.pdop, epoch.solution.dop.gdop)
        for epoch in epochs
        if epoch.fix_valid and epoch.solution is not None
    }


def _clk_by_time(epochs: list) -> dict[float, float]:
    return {
        epoch.t: epoch.solution.clk_bias_s
        for epoch in epochs
        if epoch.fix_valid and epoch.solution is not None
    }


def _alarm_rate(epochs: list, *, start_t: float) -> float:
    alarms = [epoch.nis_alarm for epoch in epochs if epoch.t >= start_t]
    if not alarms:
        return float("nan")
    return float(np.mean(alarms))


def test_spoof_clock_ramp_keeps_dop_stable_and_trips_nis() -> None:
    start_t = 5.0
    baseline_cfg = SimConfig(duration=20.0, dt=1.0, rng_seed=222, use_ekf=True)
    spoof_cfg = SimConfig(
        duration=20.0,
        dt=1.0,
        rng_seed=222,
        use_ekf=True,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": start_t, "ramp_rate_mps": 50.0},
    )

    baseline_epochs = _run_epochs(baseline_cfg)
    spoof_epochs = _run_epochs(spoof_cfg)

    baseline_dop = _dop_by_time(baseline_epochs)
    spoof_dop = _dop_by_time(spoof_epochs)
    common_times = sorted(
        t for t in baseline_dop.keys() & spoof_dop.keys() if t >= start_t
    )
    assert common_times, "Expected overlapping valid epochs for DOP comparison."
    pdop_diff = []
    gdop_diff = []
    for t in common_times:
        base_pdop, base_gdop = baseline_dop[t]
        spoof_pdop, spoof_gdop = spoof_dop[t]
        pdop_diff.append(abs(spoof_pdop - base_pdop))
        gdop_diff.append(abs(spoof_gdop - base_gdop))
    assert float(np.max(pdop_diff)) < 1e-4
    assert float(np.max(gdop_diff)) < 1e-4

    baseline_clk = _clk_by_time(baseline_epochs)
    spoof_clk = _clk_by_time(spoof_epochs)
    common_clk_times = sorted(
        t for t in baseline_clk.keys() & spoof_clk.keys() if t >= start_t
    )
    clk_diffs = [abs(spoof_clk[t] - baseline_clk[t]) for t in common_clk_times]
    assert float(np.mean(clk_diffs)) > 1e-6

    baseline_alarm_rate = _alarm_rate(baseline_epochs, start_t=start_t)
    spoof_alarm_rate = _alarm_rate(spoof_epochs, start_t=start_t)
    assert spoof_alarm_rate >= baseline_alarm_rate + 0.1
