"""Phase 3 exit checklist for attack separation and determinism."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import EPOCH_CSV_COLUMNS
from sim.run_static_demo import run_static_demo


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    run_dir: Path
    rows: list[dict[str, str]]


def main() -> None:
    run_root = Path("runs") / "phase3_exit"
    run_root.mkdir(parents=True, exist_ok=True)

    baseline_cfg = SimConfig(duration=30.0, dt=1.0, seed=123, use_ekf=True)
    spoof_cfg = SimConfig(
        duration=30.0,
        dt=1.0,
        seed=123,
        use_ekf=True,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 10.0, "ramp_rate_mps": 50.0},
    )
    jam_cfg = SimConfig(
        duration=30.0,
        dt=1.0,
        seed=123,
        use_ekf=True,
        attack_name="jam_cn0_drop",
        attack_params={"start_t": 10.0, "cn0_drop_db": 20.0, "sigma_pr_scale": 8.0},
    )

    scenarios = [
        ("baseline_ekf", baseline_cfg),
        ("spoof_ekf", spoof_cfg),
        ("jam_ekf", jam_cfg),
        ("baseline_ekf_repeat", baseline_cfg),
    ]

    results = [
        _run_scenario(name, cfg, run_root=run_root) for name, cfg in scenarios
    ]

    baseline = _find_result(results, "baseline_ekf")
    spoof = _find_result(results, "spoof_ekf")
    jam = _find_result(results, "jam_ekf")
    baseline_repeat = _find_result(results, "baseline_ekf_repeat")

    checks = [
        ("contract_columns", _check_contract_columns(results)),
        ("determinism", _check_determinism(baseline, baseline_repeat)),
        ("baseline_health", _check_baseline_health(baseline)),
        ("spoof_separation", _check_spoof_separation(baseline, spoof)),
        ("jam_separation", _check_jam_separation(baseline, jam)),
    ]

    _print_results(checks)
    failed = [name for name, passed in checks if not passed]
    if failed:
        sys.exit(1)


def _run_scenario(name: str, cfg: SimConfig, *, run_root: Path) -> ScenarioResult:
    run_dir = run_root / name
    epoch_log_path = run_static_demo(cfg, run_dir, save_figs=False)
    rows = _load_epoch_rows(epoch_log_path)
    return ScenarioResult(name=name, run_dir=run_dir, rows=rows)


def _load_epoch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _find_result(results: list[ScenarioResult], name: str) -> ScenarioResult:
    return next(result for result in results if result.name == name)


def _check_contract_columns(results: list[ScenarioResult]) -> bool:
    required = set(EPOCH_CSV_COLUMNS)
    for result in results:
        if not result.rows:
            return False
        if set(result.rows[0].keys()) != required:
            return False
    return True


def _check_determinism(baseline: ScenarioResult, repeat: ScenarioResult) -> bool:
    baseline_metrics = _summary_metrics(baseline.rows)
    repeat_metrics = _summary_metrics(repeat.rows)
    for key, value in baseline_metrics.items():
        repeat_value = repeat_metrics.get(key)
        if repeat_value is None:
            return False
        if abs(value - repeat_value) > 1e-12:
            return False
    return True


def _check_baseline_health(baseline: ScenarioResult) -> bool:
    metrics = _summary_metrics(baseline.rows)
    return metrics["nis_alarm_rate"] < 0.05 and metrics["fix_valid_rate"] > 0.9


def _check_spoof_separation(baseline: ScenarioResult, spoof: ScenarioResult) -> bool:
    baseline_metrics = _summary_metrics(baseline.rows, start_t=10.0)
    spoof_metrics = _summary_metrics(spoof.rows, start_t=10.0)
    attack_active_rate = spoof_metrics["attack_active_rate_after_start"]
    nis_alarm_rate_after_start = spoof_metrics["nis_alarm_rate_after_start"]
    baseline_nis_after_start = baseline_metrics["nis_alarm_rate_after_start"]
    return attack_active_rate > 0.5 and nis_alarm_rate_after_start > baseline_nis_after_start + 0.10


def _check_jam_separation(baseline: ScenarioResult, jam: ScenarioResult) -> bool:
    baseline_metrics = _summary_metrics(baseline.rows)
    jam_metrics = _summary_metrics(jam.rows)
    sats_drop = jam_metrics["sats_used_min"] <= baseline_metrics["sats_used_min"] - 2.0
    fix_drop = jam_metrics["fix_valid_rate"] <= baseline_metrics["fix_valid_rate"] - 0.2
    return sats_drop or fix_drop


def _summary_metrics(rows: list[dict[str, str]], *, start_t: float | None = None) -> dict[str, float]:
    t_vals = _float_column(rows, "t")
    fix_valid = _float_column(rows, "fix_valid")
    nis_alarm = _float_column(rows, "nis_alarm")
    sats_used = _float_column(rows, "sats_used")
    attack_active = _float_column(rows, "attack_active")

    metrics = {
        "nis_alarm_rate": _safe_mean(nis_alarm),
        "fix_valid_rate": _safe_mean(fix_valid),
        "sats_used_min": _safe_min(sats_used),
        "attack_active_rate": _safe_mean(attack_active),
    }

    if start_t is not None:
        mask = t_vals >= start_t
        metrics["nis_alarm_rate_after_start"] = _safe_mean(nis_alarm[mask])
        metrics["attack_active_rate_after_start"] = _safe_mean(attack_active[mask])
    else:
        metrics["nis_alarm_rate_after_start"] = float("nan")
        metrics["attack_active_rate_after_start"] = float("nan")

    return metrics


def _float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = _parse_float(row.get(key))
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return np.array([], dtype=float)
    return np.array(values, dtype=float)


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.min(values))


def _print_results(checks: Iterable[tuple[str, bool]]) -> None:
    print("Phase 3 Exit Checklist")
    print("-" * 30)
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{name:20s} : {status}")


if __name__ == "__main__":
    main()
