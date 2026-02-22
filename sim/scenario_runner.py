"""Scenario runner for GNSS twin experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import load_epochs_npz
from sim.run_static_demo import run_static_demo


def run_scenarios(
    scenario_paths: list[Path],
    *,
    run_root: Path = Path("runs"),
    save_figs: bool = True,
) -> list[dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summaries: list[dict[str, Any]] = []
    run_root.mkdir(parents=True, exist_ok=True)

    for path in scenario_paths:
        scenario = _load_scenario(path)
        scenario_name = str(scenario["name"])
        run_dir = run_root / f"{timestamp}_{_slugify(scenario_name)}"
        cfg = _build_sim_config(scenario)
        epoch_log_path = run_static_demo(cfg, run_dir, save_figs=save_figs)
        metrics = _summary_from_epoch_logs(epoch_log_path, attack_start_t=_attack_start_t(cfg))
        summary = {
            "scenario": scenario_name,
            "run_dir": str(run_dir),
            "attack_name": cfg.attack_name or "none",
            **metrics,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(_sanitize_json(summary), indent=2, allow_nan=False))
        summaries.append(summary)
        _append_summary_csv(run_root / "summary.csv", summary)

    return summaries


def _load_scenario(path: Path) -> dict[str, Any]:
    scenario = json.loads(path.read_text())
    required = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
    missing = required - scenario.keys()
    if missing:
        raise ValueError(f"Scenario {path} missing required keys: {sorted(missing)}")
    return scenario


def _build_sim_config(scenario: dict[str, Any]) -> SimConfig:
    reserved = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
    fields_by_name = {field.name for field in fields(SimConfig)}
    cfg_kwargs: dict[str, Any] = {
        "duration": float(scenario["duration_s"]),
        "rng_seed": int(scenario["rng_seed"]),
        "use_ekf": bool(scenario["use_ekf"]),
        "attack_name": str(scenario["attack_name"]),
        "attack_params": dict(scenario["attack_params"] or {}),
    }
    for key, value in scenario.items():
        if key in reserved:
            continue
        if key not in fields_by_name:
            raise ValueError(f"Unknown SimConfig override '{key}' in scenario '{scenario['name']}'")
        cfg_kwargs[key] = value
    return SimConfig(**cfg_kwargs)


def _summary_from_epoch_logs(path: Path, *, attack_start_t: float | None = None) -> dict[str, float]:
    rows = _load_epoch_rows(path)

    # Prefer truth-based RMS error (epoch_logs.npz) when available.
    npz_path = path.with_suffix(".npz")
    pos_err_rms = _pos_error_rms_from_npz(npz_path)
    if not math.isfinite(pos_err_rms):
        pos = _positions_from_rows(rows)
        pos_err_rms = _wander_rms(pos)
    residual_rms = _float_column(rows, "residual_rms_m")
    sats_used = _float_column(rows, "sats_used")
    nis_alarm = _float_column(rows, "nis_alarm")
    attack_active = _float_column(rows, "attack_active")
    attack_pr_bias = _float_column(rows, "attack_pr_bias_mean_m")
    attack_prr_bias = _float_column(rows, "attack_prr_bias_mean_mps")
    nis_after_start = _float_column_after_start(rows, "nis_alarm", attack_start_t)

    return {
        "pos_err_rms": pos_err_rms,
        "residual_rms_mean": _safe_mean(residual_rms),
        "residual_rms_max": _safe_max(residual_rms),
        "sats_used_mean": _safe_mean(sats_used),
        "sats_used_min": _safe_min(sats_used),
        "nis_alarm_rate": _safe_mean(nis_alarm),
        "attack_active_rate": _safe_mean(attack_active),
        "attack_pr_bias_mean_m_mean": _safe_mean(attack_pr_bias),
        "attack_pr_bias_mean_m_max": _safe_max(attack_pr_bias),
        "attack_prr_bias_mean_mps_mean": _safe_mean(attack_prr_bias),
        "nis_alarm_rate_after_start": _safe_mean(nis_after_start),
    }


def _pos_error_rms_from_npz(npz_path: Path) -> float:
    """Compute RMS position error using truth from epoch_logs.npz.

    Returns NaN if NPZ is missing or does not contain solution+truth.
    """

    if not npz_path.exists():
        return float("nan")
    try:
        epochs = load_epochs_npz(npz_path)
    except Exception:
        return float("nan")

    errors: list[float] = []
    for epoch in epochs:
        sol = epoch.get("solution")
        truth = epoch.get("truth")
        if not isinstance(sol, dict) or not isinstance(truth, dict):
            continue
        pos = sol.get("pos_ecef")
        tpos = truth.get("pos_ecef_m")
        if pos is None or tpos is None:
            continue
        try:
            p = np.array(pos, dtype=float).reshape(3)
            tp = np.array(tpos, dtype=float).reshape(3)
        except Exception:
            continue
        if not np.isfinite(p).all() or not np.isfinite(tp).all():
            continue
        errors.append(float(np.linalg.norm(p - tp)))

    if not errors:
        return float("nan")
    arr = np.array(errors, dtype=float)
    return float(np.sqrt(np.mean(arr**2)))


def _sanitize_json(obj: Any) -> Any:
    """Replace NaN/Inf floats with None so JSON is standards-compliant."""

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


def _load_epoch_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Epoch log not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _positions_from_rows(rows: list[dict[str, str]]) -> np.ndarray:
    """Extract estimated ECEF positions from epoch_logs.csv rows.

    Supports both legacy (pos_x/pos_y/pos_z) and current (pos_ecef_x/...) column names.
    """

    positions = []
    for row in rows:
        if "pos_ecef_x" in row:
            keys = ("pos_ecef_x", "pos_ecef_y", "pos_ecef_z")
        else:
            keys = ("pos_x", "pos_y", "pos_z")
        values = [_parse_float(row.get(key)) for key in keys]
        if all(value is not None and math.isfinite(value) for value in values):
            positions.append(values)
    if not positions:
        return np.empty((0, 3))
    return np.array(positions, dtype=float)


def _wander_rms(positions: np.ndarray) -> float:
    if positions.size == 0:
        return float("nan")
    mean_pos = positions.mean(axis=0)
    diffs = positions - mean_pos
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))


def _float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = _parse_float(row.get(key))
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return np.array([], dtype=float)
    return np.array(values, dtype=float)


def _float_column_after_start(
    rows: list[dict[str, str]],
    key: str,
    start_t: float | None,
) -> np.ndarray:
    if start_t is None:
        return np.array([], dtype=float)
    values = []
    for row in rows:
        t_value = _parse_float(row.get("t"))
        value = _parse_float(row.get(key))
        if (
            t_value is not None
            and value is not None
            and math.isfinite(t_value)
            and math.isfinite(value)
            and t_value >= start_t
        ):
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


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.min(values))


def _append_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    header = [
        "scenario",
        "run_dir",
        "attack_name",
        "pos_err_rms",
        "residual_rms_mean",
        "residual_rms_max",
        "sats_used_mean",
        "sats_used_min",
        "nis_alarm_rate",
        "attack_active_rate",
        "attack_pr_bias_mean_m_mean",
        "attack_pr_bias_mean_m_max",
        "attack_prr_bias_mean_mps_mean",
        "nis_alarm_rate_after_start",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({key: summary.get(key) for key in header})


def _slugify(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in name.lower())


def _attack_start_t(cfg: SimConfig) -> float | None:
    start_t = cfg.attack_params.get("start_t") if cfg.attack_params else None
    if start_t is None:
        return None
    try:
        return float(start_t)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GNSS twin scenarios from JSON configs.")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        help="Scenario JSON files to run.",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs",
        help="Root directory for run outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving run plots.",
    )

    # Monte Carlo mode (still scenario-style: headless runs producing CSV/plots).
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        help="If >0, run Monte Carlo for each scenario with N seeds.",
    )
    parser.add_argument(
        "--mc-seed-mode",
        type=str,
        default="offset",
        choices=["offset", "absolute"],
        help="Seed mode: offset uses scenario rng_seed as base; absolute uses mc-seed-start directly.",
    )
    parser.add_argument(
        "--mc-seed-start",
        type=int,
        default=0,
        help="Seed start (offset or absolute depending on mc-seed-mode).",
    )
    parser.add_argument(
        "--mc-seed-step",
        type=int,
        default=1,
        help="Seed stride for Monte Carlo.",
    )
    parser.add_argument(
        "--mc-per-run-plots",
        action="store_true",
        help="Save plots for each MC seed run (can be heavy).",
    )
    parser.add_argument(
        "--mc-no-aggregate-plots",
        action="store_true",
        help="Disable aggregate histogram plots for Monte Carlo.",
    )
    args = parser.parse_args()
    scenario_paths = [Path(path) for path in args.scenarios]

    if int(args.monte_carlo) > 0:
        from sim.validation.monte_carlo import run_monte_carlo

        for scenario_path in scenario_paths:
            run_monte_carlo(
                scenario_path,
                n=int(args.monte_carlo),
                seed_mode=str(args.mc_seed_mode),
                seed_start=int(args.mc_seed_start),
                seed_step=int(args.mc_seed_step),
                run_root=Path(args.run_root),
                per_run_plots=bool(args.mc_per_run_plots),
                aggregate_plots=not bool(args.mc_no_aggregate_plots),
            )
        return

    run_scenarios(scenario_paths, run_root=Path(args.run_root), save_figs=not args.no_plots)


if __name__ == "__main__":
    main()
