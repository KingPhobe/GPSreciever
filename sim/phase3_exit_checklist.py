"""Phase 3 exit checklist runs for GNSS twin scenarios."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import EPOCH_CSV_COLUMNS
from sim.run_static_demo import run_static_demo


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class Metrics:
    nis_alarm_rate: float
    nis_alarm_rate_after_start: float | None
    sats_used_min: float
    pdop_max: float


def _load_scenario(path: Path) -> dict[str, Any]:
    scenario = json.loads(path.read_text())
    required = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
    missing = required - scenario.keys()
    if missing:
        raise ValueError(f"Scenario {path} missing required keys: {sorted(missing)}")
    return scenario


def _build_sim_config(scenario: dict[str, Any]) -> SimConfig:
    cfg_kwargs: dict[str, Any] = {
        "duration": float(scenario["duration_s"]),
        "seed": int(scenario["rng_seed"]),
        "use_ekf": True,
        "attack_name": str(scenario["attack_name"]),
        "attack_params": dict(scenario["attack_params"] or {}),
    }
    for key, value in scenario.items():
        if key in {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}:
            continue
        cfg_kwargs[key] = value
    return SimConfig(**cfg_kwargs)


def _read_epoch_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Epoch log not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return rows, header


def _assert_required_columns(columns: Iterable[str]) -> None:
    required = {"t", "nis_alarm", "sats_used", "pdop"}
    missing = required - set(columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")
    missing_contract = set(EPOCH_CSV_COLUMNS) - set(columns)
    if missing_contract:
        raise AssertionError(f"CSV contract missing columns: {sorted(missing_contract)}")


def _float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)
    return np.array(values, dtype=float)


def _float_column_after(rows: list[dict[str, str]], key: str, t_start: float) -> np.ndarray:
    values = []
    for row in rows:
        raw_t = row.get("t")
        if raw_t is None or raw_t == "":
            continue
        try:
            t_value = float(raw_t)
        except ValueError:
            continue
        if t_value < t_start:
            continue
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)
    return np.array(values, dtype=float)


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.min(values))


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _compute_metrics(
    rows: list[dict[str, str]],
    *,
    attack_start_t: float | None = None,
) -> Metrics:
    nis_alarm = _float_column(rows, "nis_alarm")
    sats_used = _float_column(rows, "sats_used")
    pdop = _float_column(rows, "pdop")
    nis_alarm_rate = _safe_mean(nis_alarm)
    if attack_start_t is None:
        nis_alarm_rate_after_start = None
    else:
        nis_after = _float_column_after(rows, "nis_alarm", attack_start_t)
        nis_alarm_rate_after_start = _safe_mean(nis_after)
    return Metrics(
        nis_alarm_rate=nis_alarm_rate,
        nis_alarm_rate_after_start=nis_alarm_rate_after_start,
        sats_used_min=_safe_min(sats_used),
        pdop_max=_safe_max(pdop),
    )


def _run_scenario(spec: ScenarioSpec, run_dir: Path) -> tuple[dict[str, Any], Metrics]:
    scenario = _load_scenario(spec.path)
    cfg = _build_sim_config(scenario)
    epoch_log_path = run_static_demo(cfg, run_dir, save_figs=False)
    rows, header = _read_epoch_rows(epoch_log_path)
    _assert_required_columns(header)
    attack_start = None
    if scenario["attack_params"] and "start_t" in scenario["attack_params"]:
        attack_start = float(scenario["attack_params"]["start_t"])
    metrics = _compute_metrics(rows, attack_start_t=attack_start)
    return scenario, metrics


def _assert_determinism(metrics_a: Metrics, metrics_b: Metrics, *, rtol: float, atol: float) -> None:
    metrics_pairs = [
        ("nis_alarm_rate", metrics_a.nis_alarm_rate, metrics_b.nis_alarm_rate),
        ("sats_used_min", metrics_a.sats_used_min, metrics_b.sats_used_min),
        ("pdop_max", metrics_a.pdop_max, metrics_b.pdop_max),
    ]
    for label, value_a, value_b in metrics_pairs:
        if not np.isclose(value_a, value_b, rtol=rtol, atol=atol, equal_nan=True):
            raise AssertionError(f"Determinism check failed for {label}: {value_a} vs {value_b}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 exit checklist scenarios.")
    parser.add_argument("--run-root", type=str, default="runs/phase3_exit", help="Output directory root.")
    parser.add_argument("--nis-baseline-max", type=float, default=0.01, help="Baseline NIS alarm rate upper bound.")
    parser.add_argument(
        "--nis-spoof-min",
        type=float,
        default=0.20,
        help="Spoofing NIS alarm rate lower bound after attack start.",
    )
    parser.add_argument(
        "--jam-pdop-min",
        type=float,
        default=4.0,
        help="Jam PDOP max lower bound when sats_used_min drop is not observed.",
    )
    parser.add_argument(
        "--determinism-rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for deterministic metric comparisons.",
    )
    parser.add_argument(
        "--determinism-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for deterministic metric comparisons.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    specs = [
        ScenarioSpec(name="baseline", path=Path("sim/scenarios/baseline.json")),
        ScenarioSpec(name="spoof_clock_ramp", path=Path("sim/scenarios/spoof_clock_ramp.json")),
        ScenarioSpec(name="jam_cn0_drop", path=Path("sim/scenarios/jam_cn0_drop.json")),
    ]

    run_root.mkdir(parents=True, exist_ok=True)
    baseline_scenario, baseline_metrics = _run_scenario(specs[0], run_root / "baseline")
    spoof_scenario, spoof_metrics = _run_scenario(specs[1], run_root / "spoof_clock_ramp")
    jam_scenario, jam_metrics = _run_scenario(specs[2], run_root / "jam_cn0_drop")
    _, baseline_repeat_metrics = _run_scenario(specs[0], run_root / "baseline_repeat")

    if not baseline_scenario.get("use_ekf", False):
        raise AssertionError("Baseline scenario must run with EKF enabled.")
    if baseline_metrics.nis_alarm_rate >= args.nis_baseline_max:
        raise AssertionError(
            f"Baseline NIS alarm rate {baseline_metrics.nis_alarm_rate:.3f} "
            f">= {args.nis_baseline_max:.3f}"
        )

    if spoof_metrics.nis_alarm_rate_after_start is None:
        raise AssertionError("Spoof scenario missing attack start time for NIS check.")
    if spoof_metrics.nis_alarm_rate_after_start <= args.nis_spoof_min:
        raise AssertionError(
            f"Spoof NIS alarm rate after start {spoof_metrics.nis_alarm_rate_after_start:.3f} "
            f"<= {args.nis_spoof_min:.3f}"
        )
    if spoof_scenario.get("use_ekf") is not True:
        raise AssertionError("Spoof scenario must run with EKF enabled.")

    jam_condition = jam_metrics.sats_used_min <= baseline_metrics.sats_used_min - 2
    pdop_condition = jam_metrics.pdop_max > args.jam_pdop_min
    if not (jam_condition or pdop_condition):
        raise AssertionError(
            "Jam scenario did not degrade satellites used or PDOP as expected: "
            f"sats_used_min {jam_metrics.sats_used_min:.1f} vs baseline "
            f"{baseline_metrics.sats_used_min:.1f}, pdop_max {jam_metrics.pdop_max:.2f}"
        )
    if jam_scenario.get("use_ekf") is not True:
        raise AssertionError("Jam scenario must run with EKF enabled.")

    _assert_determinism(
        baseline_metrics,
        baseline_repeat_metrics,
        rtol=args.determinism_rtol,
        atol=args.determinism_atol,
    )

    print("Phase 3 exit checklist passed.")
    print(f"Baseline metrics: {baseline_metrics}")
    print(f"Spoof metrics: {spoof_metrics}")
    print(f"Jam metrics: {jam_metrics}")


if __name__ == "__main__":
    main()
