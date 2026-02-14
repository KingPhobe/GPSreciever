"""Reusable Phase 3 regression helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import EPOCH_CSV_COLUMNS
from sim.run_static_demo import run_static_demo


WARMUP_S = 10.0


@dataclass(frozen=True)
class Phase3Result:
    name: str
    run_dir: Path
    rows: list[dict[str, str]]
    metrics: dict[str, float]


def run_phase3_scenarios(
    *,
    run_root: Path,
    duration_s: float = 30.0,
    seed: int = 123,
) -> dict[str, Phase3Result]:
    run_root.mkdir(parents=True, exist_ok=True)

    baseline_cfg = SimConfig(duration=duration_s, dt=1.0, rng_seed=seed, use_ekf=True)
    spoof_cfg = SimConfig(
        duration=duration_s,
        dt=1.0,
        rng_seed=seed,
        use_ekf=True,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 10.0, "ramp_rate_mps": 50.0},
    )
    jam_cfg = SimConfig(
        duration=duration_s,
        dt=1.0,
        rng_seed=seed,
        use_ekf=True,
        attack_name="jam_cn0_drop",
        attack_params={"start_t": 10.0, "cn0_drop_db": 20.0, "sigma_pr_scale": 8.0},
    )

    scenarios = {
        "baseline_ekf": baseline_cfg,
        "spoof_ekf": spoof_cfg,
        "jam_ekf": jam_cfg,
        "baseline_ekf_repeat": baseline_cfg,
    }

    results: dict[str, Phase3Result] = {}
    for name, cfg in scenarios.items():
        run_dir = run_root / name
        epoch_log_path = run_static_demo(cfg, run_dir, save_figs=False)
        rows = _load_epoch_rows(epoch_log_path)
        results[name] = Phase3Result(
            name=name,
            run_dir=run_dir,
            rows=rows,
            metrics=summary_metrics(rows),
        )

    baseline_pos = _median_position(results["baseline_ekf"].rows)
    for name, result in results.items():
        metrics = dict(result.metrics)
        metrics["position_rms_m"] = position_rms_m(result.rows, baseline_pos)
        results[name] = Phase3Result(name=result.name, run_dir=result.run_dir, rows=result.rows, metrics=metrics)
    return results


def has_required_columns(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    return set(rows[0].keys()) == set(EPOCH_CSV_COLUMNS)


def summary_metrics(rows: list[dict[str, str]], *, start_t: float | None = None, warmup_s: float = WARMUP_S) -> dict[str, float]:
    t_vals = _float_column(rows, "t")
    fix_valid = _float_column(rows, "fix_valid")
    sats_used = _float_column(rows, "sats_used")
    attack_active = _float_column(rows, "attack_active")
    nis_alarm_rate, nis_alarm_samples = nis_alarm_rate_after_warmup(rows, warmup_s=warmup_s)

    metrics = {
        "nis_alarm_rate": nis_alarm_rate,
        "nis_alarm_samples": float(nis_alarm_samples),
        "fix_valid_rate": _safe_mean(fix_valid),
        "sats_used_min": _safe_min(sats_used),
        "attack_active_rate": _safe_mean(attack_active),
    }

    if start_t is not None:
        effective_start_t = max(start_t, warmup_s)
        nis_rate_after, nis_samples_after = nis_alarm_rate_after_warmup(
            rows,
            start_t=effective_start_t,
            warmup_s=warmup_s,
        )
        metrics["nis_alarm_rate_after_start"] = nis_rate_after
        metrics["nis_alarm_samples_after_start"] = float(nis_samples_after)
        metrics["attack_active_rate_after_start"] = _safe_mean(attack_active[t_vals >= effective_start_t])

    return metrics


def position_rms_m(rows: list[dict[str, str]], reference_pos_ecef_m: np.ndarray) -> float:
    coords = []
    for row in rows:
        x = _parse_float(row.get("pos_ecef_x"))
        y = _parse_float(row.get("pos_ecef_y"))
        z = _parse_float(row.get("pos_ecef_z"))
        if x is None or y is None or z is None:
            continue
        vec = np.array([x, y, z], dtype=float)
        if np.isfinite(vec).all():
            coords.append(vec)
    if not coords:
        return float("nan")
    deltas = np.vstack(coords) - reference_pos_ecef_m
    return float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))


def _median_position(rows: list[dict[str, str]]) -> np.ndarray:
    coords = []
    for row in rows:
        x = _parse_float(row.get("pos_ecef_x"))
        y = _parse_float(row.get("pos_ecef_y"))
        z = _parse_float(row.get("pos_ecef_z"))
        if x is None or y is None or z is None:
            continue
        vec = np.array([x, y, z], dtype=float)
        if np.isfinite(vec).all():
            coords.append(vec)
    if not coords:
        return np.zeros(3, dtype=float)
    return np.median(np.vstack(coords), axis=0)


def _load_epoch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = _parse_float(row.get(key))
        if value is not None and np.isfinite(value):
            values.append(value)
    return np.array(values, dtype=float) if values else np.array([], dtype=float)


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def nis_alarm_rate_after_warmup(
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
        nis_alarm_value = _parse_float(row.get("nis_alarm"))
        if nis_alarm_value is None or not np.isfinite(nis_alarm_value):
            continue
        samples.append(float(nis_alarm_value))
    if not samples:
        return float("nan"), 0
    values = np.array(samples, dtype=float)
    return float(np.mean(values)), int(values.size)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else float("nan")
