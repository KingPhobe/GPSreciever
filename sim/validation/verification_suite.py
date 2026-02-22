"""Verification / Validation suite for GNSS Twin simulation.

Goals:
  - One-button verification that answers: "is the simulation good?"
  - Covers 3 situations:
      1) Normal (baseline)
      2) Jamming
      3) Spoofing (multiple variants)
  - Produces a machine-readable report JSON + optional CSVs for analysis.
  - Supports:
      * Quick mode (single-seed, fast)
      * Monte Carlo mode (distribution + confidence)
      * Optional "golden" regression comparisons (CI-friendly)

Usage:
  Quick:
    python -m sim.validation.verification_suite --run-root runs_verify --quick

  Monte Carlo (N seeds):
    python -m sim.validation.verification_suite --run-root runs_verify --monte-carlo 30

  Write golden after a good MC run:
    python -m sim.validation.verification_suite --run-root runs_verify --monte-carlo 50 --write-golden golden_verify.json

  Compare to golden (fail if outside tolerances):
    python -m sim.validation.verification_suite --run-root runs_verify --monte-carlo 50 --compare-golden golden_verify.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.logger import load_epochs_npz
from sim.run_static_demo import run_static_demo


# ----------------------------
# Metrics extraction (truth-based via epoch_logs.npz)
# ----------------------------

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


def metrics_from_epoch_npz(npz_path: Path, *, attack_start_t: float | None) -> dict[str, float]:
    epochs = load_epochs_npz(npz_path)
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
                "attack_active_rate_after_attack": _mean(attack_active[after_mask]),
            }
        )
    else:
        out.update(
            {
                "pos_err_rms_after_attack_m": float("nan"),
                "nis_alarm_rate_after_attack": float("nan"),
                "raim_pass_rate_after_attack": float("nan"),
                "fix_valid_rate_after_attack": float("nan"),
                "attack_active_rate_after_attack": float("nan"),
            }
        )

    return out


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj


# ----------------------------
# Scenario specs (3 situations)
# ----------------------------

@dataclass(frozen=True)
class AttackSpec:
    name: str
    params: dict[str, float | str] = field(default_factory=dict)

    @property
    def start_t(self) -> float | None:
        v = self.params.get("start_t")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None


@dataclass(frozen=True)
class ScenarioSpec:
    label: str  # "normal" | "jamming" | "spoof_*"
    duration_s: float = 60.0
    dt_s: float = 1.0
    use_ekf: bool = True
    attack: AttackSpec | None = None

    def build_cfg(self, *, seed: int) -> SimConfig:
        return SimConfig(
            duration=float(self.duration_s),
            dt=float(self.dt_s),
            rng_seed=int(seed),
            use_ekf=bool(self.use_ekf),
            attack_name=(self.attack.name if self.attack else "none"),
            attack_params=(dict(self.attack.params) if self.attack else {}),
        )

    def attack_start_t(self) -> float | None:
        return self.attack.start_t if self.attack else None


def default_suite_specs() -> list[ScenarioSpec]:
    # Start time shared across attacks so "after_attack" windows line up.
    t0 = 10.0
    return [
        ScenarioSpec(label="normal", attack=None),
        ScenarioSpec(
            label="jamming",
            attack=AttackSpec(
                name="jam_cn0_drop",
                params={
                    "start_t": t0,
                    "cn0_drop_db": 15.0,
                    "sigma_pr_scale": 5.0,
                    "sigma_prr_scale": 5.0,
                },
            ),
        ),
        # Spoofing is treated as a family (3 variants) to avoid overfitting.
        ScenarioSpec(
            label="spoof_clock_ramp",
            attack=AttackSpec(
                name="spoof_clock_ramp",
                params={"start_t": t0, "ramp_rate_mps": 50.0},
            ),
        ),
        ScenarioSpec(
            label="spoof_pr_ramp",
            attack=AttackSpec(
                name="spoof_pr_ramp",
                params={
                    "start_t": t0,
                    "ramp_rate_mps": 15.0,
                    "auto_select_visible_sv": True,
                    "strict_target_sv": False,
                },
            ),
        ),
        ScenarioSpec(
            label="spoof_pos_offset",
            attack=AttackSpec(
                name="spoof_pos_offset",
                params={
                    "start_t": t0,
                    "north_m": 60.0,
                    "east_m": 25.0,
                    "up_m": 0.0,
                    "ramp_time_s": 5.0,
                },
            ),
        ),
    ]


# ----------------------------
# Pass/fail logic
# ----------------------------

@dataclass(frozen=True)
class BaselineThresholds:
    fix_valid_rate_min: float = 0.95
    raim_pass_rate_min: float = 0.95
    nis_alarm_rate_max: float = 0.05
    pos_err_rms_m_max: float = 10.0
    pos_err_p95_m_max: float = 20.0


@dataclass(frozen=True)
class DetectionThresholds:
    # "Detected" if ANY one of these triggers (relative to baseline after_attack window).
    nis_alarm_delta_min: float = 0.10
    fix_valid_drop_min: float = 0.20
    raim_pass_drop_min: float = 0.20


def evaluate_baseline(metrics: dict[str, float], thr: BaselineThresholds) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not (metrics.get("fix_valid_rate", float("nan")) >= thr.fix_valid_rate_min):
        reasons.append(f"fix_valid_rate<{thr.fix_valid_rate_min}")
    if not (metrics.get("raim_pass_rate", float("nan")) >= thr.raim_pass_rate_min):
        reasons.append(f"raim_pass_rate<{thr.raim_pass_rate_min}")
    if not (metrics.get("nis_alarm_rate", float("nan")) <= thr.nis_alarm_rate_max):
        reasons.append(f"nis_alarm_rate>{thr.nis_alarm_rate_max}")
    if not (metrics.get("pos_err_rms_m", float("nan")) <= thr.pos_err_rms_m_max):
        reasons.append(f"pos_err_rms_m>{thr.pos_err_rms_m_max}")
    if not (metrics.get("pos_err_p95_m", float("nan")) <= thr.pos_err_p95_m_max):
        reasons.append(f"pos_err_p95_m>{thr.pos_err_p95_m_max}")
    return (len(reasons) == 0), reasons


def detected_relative(
    baseline_after: dict[str, float],
    attacked_after: dict[str, float],
    thr: DetectionThresholds,
) -> tuple[bool, dict[str, float]]:
    b_nis = float(baseline_after.get("nis_alarm_rate_after_attack", float("nan")))
    a_nis = float(attacked_after.get("nis_alarm_rate_after_attack", float("nan")))
    b_fix = float(baseline_after.get("fix_valid_rate_after_attack", float("nan")))
    a_fix = float(attacked_after.get("fix_valid_rate_after_attack", float("nan")))
    b_raim = float(baseline_after.get("raim_pass_rate_after_attack", float("nan")))
    a_raim = float(attacked_after.get("raim_pass_rate_after_attack", float("nan")))

    nis_delta = a_nis - b_nis
    fix_drop = b_fix - a_fix
    raim_drop = b_raim - a_raim

    detected = False
    if math.isfinite(nis_delta) and nis_delta >= thr.nis_alarm_delta_min:
        detected = True
    if math.isfinite(fix_drop) and fix_drop >= thr.fix_valid_drop_min:
        detected = True
    if math.isfinite(raim_drop) and raim_drop >= thr.raim_pass_drop_min:
        detected = True

    return detected, {
        "nis_alarm_delta": nis_delta,
        "fix_valid_drop": fix_drop,
        "raim_pass_drop": raim_drop,
    }


# ----------------------------
# Running the suite
# ----------------------------

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(sanitize_json(payload), indent=2, allow_nan=False), encoding="utf-8")


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _run_one(spec: ScenarioSpec, *, seed: int, out_dir: Path, save_figs: bool) -> dict[str, Any]:
    cfg = spec.build_cfg(seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_static_demo(cfg, out_dir, save_figs=save_figs)
    npz = out_dir / "epoch_logs.npz"
    metrics = metrics_from_epoch_npz(npz, attack_start_t=spec.attack_start_t())
    return {"cfg": cfg, "metrics": metrics}


def run_quick_verification(
    *,
    run_root: Path = Path("runs_verify"),
    seed: int = 42,
    save_figs: bool = False,
    baseline_thr: BaselineThresholds | None = None,
    detect_thr: DetectionThresholds | None = None,
) -> dict[str, Any]:
    baseline_thr = baseline_thr or BaselineThresholds()
    detect_thr = detect_thr or DetectionThresholds()

    suite_dir = run_root / f"{_utc_stamp()}_quick"
    suite_dir.mkdir(parents=True, exist_ok=True)

    specs = default_suite_specs()
    normal = next(s for s in specs if s.label == "normal")
    t0 = next((s.attack_start_t() for s in specs if s.attack is not None), None) or 10.0

    # Run baseline. For fair "after_attack" comparisons, compute baseline after-window with same t0.
    base_dir = suite_dir / "normal"
    base = _run_one(normal, seed=seed, out_dir=base_dir, save_figs=save_figs)
    base_metrics_full = dict(base["metrics"])
    # Recompute baseline with explicit t0 windowing by temporarily faking attack_start_t.
    # (We just want baseline_*_after_attack values to be defined on same time slice.)
    base_metrics = metrics_from_epoch_npz(base_dir / "epoch_logs.npz", attack_start_t=t0)

    baseline_ok, baseline_reasons = evaluate_baseline(base_metrics_full, baseline_thr)

    results: dict[str, Any] = {
        "normal": {
            "pass": baseline_ok,
            "reasons": baseline_reasons,
            "metrics": base_metrics_full,
            "run_dir": str(base_dir),
        }
    }

    # Run attacked scenarios and evaluate detection relative to baseline(after window).
    for spec in specs:
        if spec.label == "normal":
            continue
        run_dir = suite_dir / spec.label
        out = _run_one(spec, seed=seed, out_dir=run_dir, save_figs=save_figs)
        attacked_metrics = out["metrics"]
        # Ensure after_attack values exist using the attack start_t (from spec).
        detected, deltas = detected_relative(base_metrics, attacked_metrics, detect_thr)
        results[spec.label] = {
            "pass": bool(detected),
            "reasons": ([] if detected else ["not_detected"]),
            "deltas_vs_baseline_after_window": deltas,
            "metrics": attacked_metrics,
            "run_dir": str(run_dir),
        }

    # Aggregate spoofing family verdict.
    spoof_labels = [s.label for s in specs if s.label.startswith("spoof_")]
    spoof_pass = all(bool(results[l]["pass"]) for l in spoof_labels)
    results["spoofing_overall"] = {
        "pass": spoof_pass,
        "required_variants": spoof_labels,
        "failed_variants": [l for l in spoof_labels if not bool(results[l]["pass"])],
    }

    report = {
        "mode": "quick",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "suite_dir": str(suite_dir),
        "seed": int(seed),
        "thresholds": {
            "baseline": baseline_thr.__dict__,
            "detection": detect_thr.__dict__,
        },
        "results": results,
        "overall_pass": bool(results["normal"]["pass"]) and bool(results["jamming"]["pass"]) and bool(results["spoofing_overall"]["pass"]),
    }

    _write_json(suite_dir / "verification_report.json", report)
    return report


def run_monte_carlo_verification(
    *,
    run_root: Path = Path("runs_verify"),
    n: int = 30,
    seed_start: int = 42,
    seed_step: int = 1,
    save_figs_per_run: bool = False,
    baseline_thr: BaselineThresholds | None = None,
    detect_thr: DetectionThresholds | None = None,
    detect_rate_min: float = 0.70,
) -> dict[str, Any]:
    baseline_thr = baseline_thr or BaselineThresholds()
    detect_thr = detect_thr or DetectionThresholds()

    suite_dir = run_root / f"{_utc_stamp()}_mc"
    suite_dir.mkdir(parents=True, exist_ok=True)

    specs = default_suite_specs()
    normal = next(s for s in specs if s.label == "normal")
    t0 = next((s.attack_start_t() for s in specs if s.attack is not None), None) or 10.0

    seeds = [int(seed_start + i * seed_step) for i in range(int(n))]

    # We store per-seed metrics in CSV for each scenario.
    per_scenario_csv: dict[str, Path] = {s.label: (suite_dir / f"{s.label}_mc_runs.csv") for s in specs}

    # Collect per-seed pass/fail rates.
    baseline_passes = 0
    detection_hits: dict[str, int] = {s.label: 0 for s in specs if s.label != "normal"}

    # We'll keep a small set of aggregate arrays per scenario for final summary.
    aggregates: dict[str, dict[str, list[float]]] = {s.label: {} for s in specs}

    for seed in seeds:
        # Baseline run (needed as reference for detection).
        base_dir = suite_dir / "normal" / f"seed_{seed:06d}"
        base_out = _run_one(normal, seed=seed, out_dir=base_dir, save_figs=save_figs_per_run)
        base_metrics_full = base_out["metrics"]
        base_metrics_after = metrics_from_epoch_npz(base_dir / "epoch_logs.npz", attack_start_t=t0)
        base_ok, _ = evaluate_baseline(base_metrics_full, baseline_thr)
        baseline_passes += int(base_ok)

        # Record baseline row.
        row = {"seed": seed, **base_metrics_full}
        _append_csv(per_scenario_csv["normal"], row)
        _accumulate(aggregates["normal"], base_metrics_full)

        # Attacks
        for spec in specs:
            if spec.label == "normal":
                continue
            run_dir = suite_dir / spec.label / f"seed_{seed:06d}"
            out = _run_one(spec, seed=seed, out_dir=run_dir, save_figs=save_figs_per_run)
            attacked_metrics = out["metrics"]
            detected, deltas = detected_relative(base_metrics_after, attacked_metrics, detect_thr)
            detection_hits[spec.label] += int(detected)

            row = {"seed": seed, **attacked_metrics, **{f"delta_{k}": v for k, v in deltas.items()}, "detected": int(detected)}
            _append_csv(per_scenario_csv[spec.label], row)
            _accumulate(aggregates[spec.label], attacked_metrics)

    # Summaries
    baseline_pass_rate = baseline_passes / float(len(seeds)) if seeds else 0.0
    detection_rates = {k: (v / float(len(seeds)) if seeds else 0.0) for k, v in detection_hits.items()}

    spoof_labels = [s.label for s in specs if s.label.startswith("spoof_")]
    spoof_detect_rate_min = min(detection_rates.get(l, 0.0) for l in spoof_labels) if spoof_labels else 0.0

    overall_pass = True
    reasons: list[str] = []

    if baseline_pass_rate < detect_rate_min:
        overall_pass = False
        reasons.append(f"baseline_pass_rate<{detect_rate_min}")

    for label, rate in detection_rates.items():
        if rate < detect_rate_min:
            overall_pass = False
            reasons.append(f"{label}_detect_rate<{detect_rate_min}")

    # Add aggregate stats (mean/std/p95) per scenario.
    aggregate_stats = {label: _aggregate_stats(vals) for label, vals in aggregates.items()}

    report = {
        "mode": "monte_carlo",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "suite_dir": str(suite_dir),
        "n": int(n),
        "seed_start": int(seed_start),
        "seed_step": int(seed_step),
        "seeds": seeds,
        "thresholds": {
            "baseline": baseline_thr.__dict__,
            "detection": detect_thr.__dict__,
            "detect_rate_min": float(detect_rate_min),
        },
        "baseline_pass_rate": float(baseline_pass_rate),
        "detection_rates": detection_rates,
        "spoof_detect_rate_min": float(spoof_detect_rate_min),
        "aggregate_metrics": aggregate_stats,
        "overall_pass": bool(overall_pass),
        "reasons": reasons,
        "artifacts": {label: str(path) for label, path in per_scenario_csv.items()},
    }

    _write_json(suite_dir / "verification_report.json", report)
    return report


def _accumulate(bucket: dict[str, list[float]], metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        fv = float(v)
        if not math.isfinite(fv):
            continue
        bucket.setdefault(k, []).append(fv)


def _aggregate_stats(bucket: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k, arr in bucket.items():
        v = np.array(arr, dtype=float)
        if v.size == 0:
            continue
        out[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "p95": float(np.percentile(v, 95)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
    return out


# ----------------------------
# Golden regression support
# ----------------------------

@dataclass(frozen=True)
class GoldenTolerances:
    # Error metrics are allowed to increase by at most this relative factor vs golden.
    pos_err_rms_rel_increase_max: float = 0.20
    pos_err_p95_rel_increase_max: float = 0.20
    residual_rms_mean_rel_increase_max: float = 0.20
    # Rate metrics are allowed to decrease (worse) by at most these absolute deltas.
    fix_valid_abs_decrease_max: float = 0.05
    raim_pass_abs_decrease_max: float = 0.05
    # False alarms allowed to increase by at most abs delta.
    nis_alarm_abs_increase_max: float = 0.05
    # Detection rate allowed to drop by at most abs delta.
    detect_rate_abs_decrease_max: float = 0.10


def write_golden(report: dict[str, Any], path: Path) -> None:
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "gnss_twin_verification_golden_v1",
        "mode": report.get("mode"),
        "n": report.get("n"),
        "thresholds": report.get("thresholds"),
        "baseline_pass_rate": report.get("baseline_pass_rate"),
        "detection_rates": report.get("detection_rates"),
        "aggregate_metrics": report.get("aggregate_metrics"),
    }
    _write_json(path, payload)


def compare_to_golden(
    *,
    report: dict[str, Any],
    golden_path: Path,
    tol: GoldenTolerances | None = None,
) -> tuple[bool, list[str]]:
    tol = tol or GoldenTolerances()
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    reasons: list[str] = []

    # Compare baseline aggregate if present.
    cur_agg = report.get("aggregate_metrics", {}) or {}
    gold_agg = golden.get("aggregate_metrics", {}) or {}

    def get_stat(agg: dict, scenario: str, metric: str, stat: str) -> float:
        try:
            return float(agg[scenario][metric][stat])
        except Exception:
            return float("nan")

    # Baseline quality: compare NORMAL scenario mean/p95 error metrics.
    for metric, rel_max in [
        ("pos_err_rms_m", tol.pos_err_rms_rel_increase_max),
        ("pos_err_p95_m", tol.pos_err_p95_rel_increase_max),
        ("residual_rms_mean_m", tol.residual_rms_mean_rel_increase_max),
    ]:
        cur = get_stat(cur_agg, "normal", metric, "mean")
        ref = get_stat(gold_agg, "normal", metric, "mean")
        if math.isfinite(cur) and math.isfinite(ref):
            if cur > ref * (1.0 + rel_max):
                reasons.append(f"normal.{metric}.mean increased too much: {cur} > {ref}*(1+{rel_max})")

    # Baseline rates: mean values
    for metric, abs_dec in [
        ("fix_valid_rate", tol.fix_valid_abs_decrease_max),
        ("raim_pass_rate", tol.raim_pass_abs_decrease_max),
    ]:
        cur = get_stat(cur_agg, "normal", metric, "mean")
        ref = get_stat(gold_agg, "normal", metric, "mean")
        if math.isfinite(cur) and math.isfinite(ref):
            if cur < ref - abs_dec:
                reasons.append(f"normal.{metric}.mean decreased too much: {cur} < {ref}-{abs_dec}")

    # False alarms: nis_alarm_rate mean
    cur_nis = get_stat(cur_agg, "normal", "nis_alarm_rate", "mean")
    ref_nis = get_stat(gold_agg, "normal", "nis_alarm_rate", "mean")
    if math.isfinite(cur_nis) and math.isfinite(ref_nis):
        if cur_nis > ref_nis + tol.nis_alarm_abs_increase_max:
            reasons.append(f"normal.nis_alarm_rate.mean increased too much: {cur_nis} > {ref_nis}+{tol.nis_alarm_abs_increase_max}")

    # Detection rates: ensure they don't drop too much vs golden
    cur_det = report.get("detection_rates", {}) or {}
    ref_det = golden.get("detection_rates", {}) or {}
    for label, ref_rate in ref_det.items():
        try:
            ref_rate_f = float(ref_rate)
        except Exception:
            continue
        try:
            cur_rate_f = float(cur_det.get(label, float("nan")))
        except Exception:
            continue
        if math.isfinite(cur_rate_f) and math.isfinite(ref_rate_f):
            if cur_rate_f < ref_rate_f - tol.detect_rate_abs_decrease_max:
                reasons.append(f"{label}.detect_rate dropped too much: {cur_rate_f} < {ref_rate_f}-{tol.detect_rate_abs_decrease_max}")

    return (len(reasons) == 0), reasons


# ----------------------------
# CLI entrypoint
# ----------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="GNSS Twin verification suite")
    p.add_argument("--run-root", type=str, default="runs_verify")
    p.add_argument("--quick", action="store_true", help="Run quick single-seed verification")
    p.add_argument("--seed", type=int, default=42, help="Seed for quick verification")
    p.add_argument("--save-figs", action="store_true", help="Save plots in quick runs")
    p.add_argument("--monte-carlo", type=int, default=0, help="If >0, run Monte Carlo with N runs")
    p.add_argument("--mc-seed-start", type=int, default=42)
    p.add_argument("--mc-seed-step", type=int, default=1)
    p.add_argument("--mc-save-figs-per-run", action="store_true", help="Save plots for every MC run (heavy)")
    p.add_argument("--detect-rate-min", type=float, default=0.70, help="Minimum required detection rate in MC")

    p.add_argument("--write-golden", type=str, default=None, help="Write golden JSON to this path (after MC)")
    p.add_argument("--compare-golden", type=str, default=None, help="Compare MC report to golden JSON at this path")

    args = p.parse_args(argv)
    run_root = Path(args.run_root)

    if args.quick:
        report = run_quick_verification(
            run_root=run_root,
            seed=int(args.seed),
            save_figs=bool(args.save_figs),
        )
        print(json.dumps({"suite_dir": report["suite_dir"], "overall_pass": report["overall_pass"]}, indent=2))
        return

    if int(args.monte_carlo) > 0:
        report = run_monte_carlo_verification(
            run_root=run_root,
            n=int(args.monte_carlo),
            seed_start=int(args.mc_seed_start),
            seed_step=int(args.mc_seed_step),
            save_figs_per_run=bool(args.mc_save_figs_per_run),
            detect_rate_min=float(args.detect_rate_min),
        )

        if args.write_golden:
            write_golden(report, Path(args.write_golden))

        if args.compare_golden:
            ok, reasons = compare_to_golden(report=report, golden_path=Path(args.compare_golden))
            if not ok:
                print(json.dumps({"overall_pass": False, "golden_compare_failures": reasons}, indent=2))
                raise SystemExit(2)

        print(json.dumps({"suite_dir": report["suite_dir"], "overall_pass": report["overall_pass"]}, indent=2))
        return

    raise SystemExit("Choose --quick or --monte-carlo N")


if __name__ == "__main__":
    main()
