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


WARMUP_S = 10.0
NIS_THRESHOLD_P = 0.95


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    run_dir: Path
    rows: list[dict[str, str]]


def main() -> None:
    run_root = Path("runs") / "phase3_exit"
    run_root.mkdir(parents=True, exist_ok=True)

    baseline_cfg = SimConfig(duration=30.0, dt=1.0, rng_seed=123, use_ekf=True)
    spoof_cfg = SimConfig(
        duration=30.0,
        dt=1.0,
        rng_seed=123,
        use_ekf=True,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 10.0, "ramp_rate_mps": 50.0},
    )
    jam_cfg = SimConfig(
        duration=30.0,
        dt=1.0,
        rng_seed=123,
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
    baseline_nis_alarm_rate = metrics["nis_alarm_rate"]
    nis_samples = int(metrics["nis_alarm_samples"])
    assert baseline_nis_alarm_rate < 0.030, (
        "Baseline NIS alarm rate too high: "
        f"baseline_nis_alarm_rate={baseline_nis_alarm_rate:.4f}, "
        f"nis_threshold_p={NIS_THRESHOLD_P}, "
        f"WARMUP_S={WARMUP_S}, "
        f"samples={nis_samples}"
    )
    return metrics["fix_valid_rate"] > 0.9


def _check_spoof_separation(baseline: ScenarioResult, spoof: ScenarioResult) -> bool:
    attack_start_t = 10.0
    after_start_t = max(attack_start_t + 2.0, WARMUP_S)
    baseline_metrics = _summary_metrics(baseline.rows, start_t=after_start_t)
    spoof_metrics = _summary_metrics(spoof.rows, start_t=after_start_t)
    attack_active_rate = spoof_metrics["attack_active_rate_after_start"]
    nis_alarm_rate_after_start = spoof_metrics["nis_alarm_rate_after_start"]
    baseline_nis_after_start = baseline_metrics["nis_alarm_rate_after_start"]
    assert nis_alarm_rate_after_start >= baseline_nis_after_start + 0.10, (
        "Spoof NIS separation too small: "
        f"spoof_nis_alarm_rate_after_start={nis_alarm_rate_after_start:.4f}, "
        f"baseline_nis_alarm_rate_after_start={baseline_nis_after_start:.4f}, "
        f"WARMUP_S={WARMUP_S}, "
        f"start_t={after_start_t}"
    )
    return attack_active_rate > 0.5


def _check_jam_separation(baseline: ScenarioResult, jam: ScenarioResult) -> bool:
    baseline_metrics = _summary_metrics(baseline.rows)
    jam_metrics = _summary_metrics(jam.rows)
    sats_drop = jam_metrics["sats_used_min"] <= baseline_metrics["sats_used_min"] - 2.0
    fix_drop = jam_metrics["fix_valid_rate"] <= baseline_metrics["fix_valid_rate"] - 0.2
    return sats_drop or fix_drop


def _summary_metrics(
    rows: list[dict[str, str]], *, start_t: float | None = None, warmup_s: float = WARMUP_S
) -> dict[str, float]:
    t_vals = _float_column(rows, "t")
    fix_valid = _float_column(rows, "fix_valid")
    sats_used = _float_column(rows, "sats_used")
    attack_active = _float_column(rows, "attack_active")
    nis_alarm_rate, nis_alarm_samples = _nis_alarm_rate(rows, warmup_s=warmup_s)

    metrics = {
        "nis_alarm_rate": nis_alarm_rate,
        "nis_alarm_samples": float(nis_alarm_samples),
        "fix_valid_rate": _safe_mean(fix_valid),
        "sats_used_min": _safe_min(sats_used),
        "attack_active_rate": _safe_mean(attack_active),
    }

    if start_t is not None:
        effective_start_t = max(start_t, warmup_s)
        mask = t_vals >= effective_start_t
        nis_alarm_after_start, nis_samples_after_start = _nis_alarm_rate(
            rows,
            start_t=effective_start_t,
            warmup_s=warmup_s,
        )
        metrics["nis_alarm_rate_after_start"] = nis_alarm_after_start
        metrics["nis_alarm_samples_after_start"] = float(nis_samples_after_start)
        metrics["attack_active_rate_after_start"] = _safe_mean(attack_active[mask])
    else:
        metrics["nis_alarm_rate_after_start"] = float("nan")
        metrics["nis_alarm_samples_after_start"] = 0.0
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


def _nis_alarm_rate(
    rows: list[dict[str, str]],
    *,
    start_t: float | None = None,
    warmup_s: float = WARMUP_S,
) -> tuple[float, int]:
    min_t = warmup_s if start_t is None else max(start_t, warmup_s)
    samples: list[float] = []
    for row in rows:
        t_s = _parse_float(row.get("t_s"))
        if t_s is None or not np.isfinite(t_s) or t_s < min_t:
            continue
        nis_value = _parse_float(row.get("nis"))
        if nis_value is None or not np.isfinite(nis_value):
            continue
        innov_dim = _parse_float(row.get("innov_dim"))
        if innov_dim is None or not np.isfinite(innov_dim) or innov_dim <= 0.0:
            continue
        nis_alarm_value = _parse_float(row.get("nis_alarm"))
        if nis_alarm_value is None or not np.isfinite(nis_alarm_value):
            continue
        samples.append(float(nis_alarm_value))
    if not samples:
        return float("nan"), 0
    values = np.array(samples, dtype=float)
    return float(np.mean(values)), int(values.size)


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
