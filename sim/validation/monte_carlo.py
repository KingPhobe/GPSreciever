"""Monte Carlo runner for GNSS twin scenarios.

Scenario-mode tool: runs the same scenario many times with different RNG seeds,
and produces one aggregate summary (+ optional histograms).

Usage:
  python -m sim.validation.monte_carlo --scenario sim/scenarios/baseline.json --n 50
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import load_epochs_npz
from sim.run_static_demo import run_static_demo


def run_monte_carlo(
    scenario_path: Path,
    *,
    n: int,
    seed_mode: str = "offset",
    seed_start: int = 0,
    seed_step: int = 1,
    seeds: list[int] | None = None,
    run_root: Path = Path("runs"),
    per_run_plots: bool = False,
    aggregate_plots: bool = True,
) -> dict[str, Any]:
    scenario = _load_scenario(scenario_path)
    base_seed = int(scenario["rng_seed"])
    scenario_name = str(scenario["name"])
    attack_name = str(scenario.get("attack_name") or "none")

    if seeds is None:
        if n <= 0:
            raise ValueError("n must be > 0")
        seeds = list(_generate_seeds(base_seed, n=n, mode=seed_mode, start=seed_start, step=seed_step))
    if not seeds:
        raise ValueError("No seeds to run")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mc_dir = run_root / f"{timestamp}_mc_{_slugify(scenario_name)}"
    mc_dir.mkdir(parents=True, exist_ok=True)

    cfg_template = _build_sim_config(scenario)
    attack_start_t = _attack_start_t(cfg_template)

    mc_cfg = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_path": str(scenario_path),
        "scenario": scenario,
        "n": int(len(seeds)),
        "seed_mode": seed_mode,
        "seed_start": int(seed_start),
        "seed_step": int(seed_step),
        "seeds": [int(s) for s in seeds],
        "per_run_plots": bool(per_run_plots),
        "aggregate_plots": bool(aggregate_plots),
    }
    (mc_dir / "mc_config.json").write_text(json.dumps(mc_cfg, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    runs_csv = mc_dir / "mc_runs.csv"
    for seed in seeds:
        run_dir = mc_dir / f"seed_{int(seed):06d}"
        cfg = _copy_with_seed(cfg_template, int(seed))
        epoch_csv = run_static_demo(cfg, run_dir, save_figs=per_run_plots)
        metrics = _metrics_from_run_dir(run_dir, attack_start_t=attack_start_t)
        row = {
            "scenario": scenario_name,
            "attack_name": attack_name,
            "seed": int(seed),
            "run_dir": str(run_dir),
            "epoch_logs": str(epoch_csv),
            **metrics,
        }
        rows.append(row)
        _append_csv_row(runs_csv, row)

    report = {
        "scenario": scenario_name,
        "attack_name": attack_name,
        "scenario_path": str(scenario_path),
        "mc_dir": str(mc_dir),
        "n": int(len(rows)),
        "seeds": [int(r["seed"]) for r in rows],
        "aggregate": _aggregate_rows(rows),
    }
    (mc_dir / "mc_aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if aggregate_plots:
        try:
            _save_aggregate_plots(mc_dir, rows)
        except Exception:
            pass

    return report


def _metrics_from_run_dir(run_dir: Path, *, attack_start_t: float | None) -> dict[str, float]:
    npz = run_dir / "epoch_logs.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing epoch_logs.npz in {run_dir}")
    epochs = load_epochs_npz(npz)
    return _metrics_from_epoch_dicts(epochs, attack_start_t=attack_start_t)


def _metrics_from_epoch_dicts(epochs: list[dict[str, Any]], *, attack_start_t: float | None) -> dict[str, float]:
    t_s = np.array([_epoch_time_s(e) for e in epochs], dtype=float)
    pos_err = np.array([_pos_error_m(e) for e in epochs], dtype=float)
    clk_err = np.array([_clk_bias_error_s(e) for e in epochs], dtype=float)
    residual = np.array([_residual_rms_m(e) for e in epochs], dtype=float)
    sats_used = np.array([_sats_used(e) for e in epochs], dtype=float)
    raim_pass = np.array([_raim_pass(e) for e in epochs], dtype=float)
    fix_valid = np.array([_fix_valid(e) for e in epochs], dtype=float)
    nis_alarm = np.array([_nis_alarm(e) for e in epochs], dtype=float)
    attack_active = np.array([1.0 if bool(e.get("attack_active")) else 0.0 for e in epochs], dtype=float)

    after_mask = np.isfinite(t_s) & (t_s >= float(attack_start_t)) if attack_start_t is not None else None

    out = {
        "pos_err_rms_m": _rms(pos_err),
        "pos_err_p95_m": _pctl(pos_err, 95),
        "clk_bias_err_rms_s": _rms(clk_err),
        "clk_bias_err_p95_s": _pctl(clk_err, 95),
        "residual_rms_mean_m": _mean(residual),
        "residual_rms_max_m": _max(residual),
        "sats_used_mean": _mean(sats_used),
        "sats_used_min": _min(sats_used),
        "raim_pass_rate": _mean(raim_pass),
        "fix_valid_rate": _mean(fix_valid),
        "nis_alarm_rate": _mean(nis_alarm),
        "attack_active_rate": _mean(attack_active),
    }

    if after_mask is not None and np.any(after_mask):
        out.update(
            {
                "pos_err_rms_after_attack_m": _rms(pos_err[after_mask]),
                "nis_alarm_rate_after_attack": _mean(nis_alarm[after_mask]),
                "raim_pass_rate_after_attack": _mean(raim_pass[after_mask]),
                "fix_valid_rate_after_attack": _mean(fix_valid[after_mask]),
            }
        )
    else:
        out.update(
            {
                "pos_err_rms_after_attack_m": float("nan"),
                "nis_alarm_rate_after_attack": float("nan"),
                "raim_pass_rate_after_attack": float("nan"),
                "fix_valid_rate_after_attack": float("nan"),
            }
        )

    return out


def _epoch_time_s(epoch: dict[str, Any]) -> float:
    t_s = epoch.get("t_s", epoch.get("t"))
    try:
        return float(t_s)
    except Exception:
        return float("nan")


def _pos_error_m(epoch: dict[str, Any]) -> float:
    sol = epoch.get("solution")
    truth = epoch.get("truth")
    if not isinstance(sol, dict) or not isinstance(truth, dict):
        return float("nan")
    pos = sol.get("pos_ecef")
    truth_pos = truth.get("pos_ecef_m")
    if pos is None or truth_pos is None:
        return float("nan")
    try:
        p = np.array(pos, dtype=float).reshape(3)
        tp = np.array(truth_pos, dtype=float).reshape(3)
    except Exception:
        return float("nan")
    if not np.isfinite(p).all() or not np.isfinite(tp).all():
        return float("nan")
    return float(np.linalg.norm(p - tp))


def _clk_bias_error_s(epoch: dict[str, Any]) -> float:
    sol = epoch.get("solution")
    truth = epoch.get("truth")
    if not isinstance(sol, dict) or not isinstance(truth, dict):
        return float("nan")
    try:
        return float(sol.get("clk_bias_s")) - float(truth.get("clk_bias_s"))
    except Exception:
        return float("nan")


def _residual_rms_m(epoch: dict[str, Any]) -> float:
    v = epoch.get("residual_rms_m")
    if v is not None:
        try:
            return float(v)
        except Exception:
            return float("nan")
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("residuals"), dict):
        try:
            return float(sol["residuals"].get("rms_m"))
        except Exception:
            return float("nan")
    return float("nan")


def _sats_used(epoch: dict[str, Any]) -> float:
    v = epoch.get("sats_used")
    if v is not None:
        try:
            return float(v)
        except Exception:
            return float("nan")
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("fix_flags"), dict):
        try:
            return float(sol["fix_flags"].get("sv_count"))
        except Exception:
            return float("nan")
    return float("nan")


def _raim_pass(epoch: dict[str, Any]) -> float:
    v = epoch.get("raim_pass")
    if v is not None:
        return 1.0 if bool(v) else 0.0
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("fix_flags"), dict):
        return 1.0 if bool(sol["fix_flags"].get("raim_passed")) else 0.0
    return float("nan")


def _fix_valid(epoch: dict[str, Any]) -> float:
    v = epoch.get("fix_valid")
    if v is not None:
        return 1.0 if bool(v) else 0.0
    sol = epoch.get("solution")
    if isinstance(sol, dict) and isinstance(sol.get("fix_flags"), dict):
        return 1.0 if bool(sol["fix_flags"].get("valid")) else 0.0
    return float("nan")


def _nis_alarm(epoch: dict[str, Any]) -> float:
    v = epoch.get("nis_alarm")
    if v is None:
        return float("nan")
    return 1.0 if bool(v) else 0.0


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and k != "seed"})
    out: dict[str, Any] = {}
    for k in numeric_keys:
        arr = np.array([float(r.get(k, float("nan"))) for r in rows], dtype=float)
        arr = arr[np.isfinite(arr)]
        out[k] = _summary_stats(arr)
    return out


def _summary_stats(v: np.ndarray) -> dict[str, float]:
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def _save_aggregate_plots(mc_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    out = mc_dir / "mc_plots"
    out.mkdir(parents=True, exist_ok=True)

    def hist(metric: str, filename: str) -> None:
        vals = np.array([float(r.get(metric, float("nan"))) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        plt.figure()
        plt.hist(vals, bins=25)
        plt.title(metric)
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out / filename)
        plt.close()

    hist("pos_err_rms_m", "hist_pos_err_rms_m.png")
    hist("nis_alarm_rate", "hist_nis_alarm_rate.png")
    hist("raim_pass_rate", "hist_raim_pass_rate.png")
    hist("residual_rms_mean_m", "hist_residual_rms_mean_m.png")


def _generate_seeds(base_seed: int, *, n: int, mode: str, start: int, step: int) -> Iterable[int]:
    if step <= 0:
        raise ValueError("seed_step must be > 0")
    mode = (mode or "").lower().strip()
    for i in range(int(n)):
        if mode in {"offset", "relative"}:
            yield int(base_seed + start + i * step)
        elif mode in {"absolute", "override"}:
            yield int(start + i * step)
        else:
            raise ValueError("seed_mode must be one of: offset|absolute")


def _load_scenario(path: Path) -> dict[str, Any]:
    scenario = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
    missing = required - scenario.keys()
    if missing:
        raise ValueError(f"Scenario {path} missing required keys: {sorted(missing)}")
    return scenario


def _build_sim_config(scenario: dict[str, Any]) -> SimConfig:
    reserved = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
    fields_by_name = {f.name for f in fields(SimConfig)}
    cfg_kwargs: dict[str, Any] = {
        "duration": float(scenario["duration_s"]),
        "rng_seed": int(scenario["rng_seed"]),
        "use_ekf": bool(scenario["use_ekf"]),
        "attack_name": str(scenario["attack_name"]),
        "attack_params": dict(scenario["attack_params"] or {}),
    }
    for k, v in scenario.items():
        if k in reserved:
            continue
        if k not in fields_by_name:
            raise ValueError(f"Unknown SimConfig override '{k}' in scenario '{scenario['name']}'")
        cfg_kwargs[k] = v
    return SimConfig(**cfg_kwargs)


def _copy_with_seed(cfg: SimConfig, seed: int) -> SimConfig:
    d = asdict(cfg)
    d["rng_seed"] = int(seed)
    return SimConfig(**d)


def _attack_start_t(cfg: SimConfig) -> float | None:
    start_t = cfg.attack_params.get("start_t") if cfg.attack_params else None
    if start_t is None:
        return None
    try:
        return float(start_t)
    except (TypeError, ValueError):
        return None


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _slugify(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.lower())


def _finite(v: np.ndarray) -> np.ndarray:
    return v[np.isfinite(v)]


def _mean(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.mean(v)) if v.size else float("nan")


def _max(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.max(v)) if v.size else float("nan")


def _min(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.min(v)) if v.size else float("nan")


def _rms(v: np.ndarray) -> float:
    v = _finite(v)
    return float(np.sqrt(np.mean(v**2))) if v.size else float("nan")


def _pctl(v: np.ndarray, p: float) -> float:
    v = _finite(v)
    return float(np.percentile(v, p)) if v.size else float("nan")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run a Monte Carlo sweep for a GNSS twin scenario.")
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed-mode", choices=["offset", "absolute"], default="offset")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-step", type=int, default=1)
    p.add_argument("--run-root", type=str, default="runs")
    p.add_argument("--per-run-plots", action="store_true")
    p.add_argument("--no-aggregate-plots", action="store_true")
    args = p.parse_args(argv)

    report = run_monte_carlo(
        Path(args.scenario),
        n=int(args.n),
        seed_mode=str(args.seed_mode),
        seed_start=int(args.seed_start),
        seed_step=int(args.seed_step),
        run_root=Path(args.run_root),
        per_run_plots=bool(args.per_run_plots),
        aggregate_plots=not bool(args.no_aggregate_plots),
    )
    print(json.dumps({"mc_dir": report["mc_dir"], "n": report["n"]}, indent=2))


if __name__ == "__main__":
    main()
