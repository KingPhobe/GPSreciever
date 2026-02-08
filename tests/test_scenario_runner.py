from __future__ import annotations

import json
import math
from pathlib import Path

from sim.scenario_runner import run_scenarios


def _write_scenario(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _metric_higher(spoof: float, baseline: float) -> bool:
    if not math.isfinite(spoof) or not math.isfinite(baseline):
        return False
    return spoof > baseline


def test_scenario_runner_outputs(tmp_path: Path) -> None:
    scenarios_dir = tmp_path / "scenarios"
    scenarios_dir.mkdir()
    baseline = _write_scenario(
        scenarios_dir / "baseline.json",
        {
            "name": "baseline",
            "duration_s": 20,
            "rng_seed": 123,
            "use_ekf": True,
            "attack_name": "none",
            "attack_params": {},
        },
    )
    spoof = _write_scenario(
        scenarios_dir / "spoof.json",
        {
            "name": "spoof",
            "duration_s": 20,
            "rng_seed": 123,
            "use_ekf": True,
            "attack_name": "spoof_pr_ramp",
            "attack_params": {"start_t": 0.0, "ramp_rate_mps": 500.0, "target_sv": "G03"},
        },
    )
    run_root = tmp_path / "runs"
    summaries = run_scenarios([baseline, spoof], run_root=run_root, save_figs=False)

    assert len(summaries) == 2
    run_dirs = [Path(summary["run_dir"]) for summary in summaries]
    assert all(run_dir.exists() for run_dir in run_dirs)
    assert all((run_dir / "summary.json").exists() for run_dir in run_dirs)
    summary_csv = run_root / "summary.csv"
    assert summary_csv.exists()

    baseline_summary = next(summary for summary in summaries if summary["scenario"] == "baseline")
    spoof_summary = next(summary for summary in summaries if summary["scenario"] == "spoof")
    nis_higher = _metric_higher(spoof_summary["nis_alarm_rate"], baseline_summary["nis_alarm_rate"])
    pos_higher = _metric_higher(spoof_summary["pos_err_rms"], baseline_summary["pos_err_rms"])
    assert nis_higher or pos_higher
