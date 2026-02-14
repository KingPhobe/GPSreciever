"""Phase 3 exit checklist for attack separation and determinism."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sim.validation.phase3 import (
    has_required_columns,
    position_rms_m,
    run_phase3_scenarios,
    summary_metrics,
)


WARMUP_S = 10.0


def main() -> None:
    run_root = Path("runs") / "phase3_exit"
    results = run_phase3_scenarios(run_root=run_root, duration_s=30.0, seed=123)

    baseline = results["baseline_ekf"]
    spoof = results["spoof_ekf"]
    jam = results["jam_ekf"]
    baseline_repeat = results["baseline_ekf_repeat"]

    checks = [
        ("contract_columns", all(has_required_columns(result.rows) for result in results.values())),
        ("determinism", baseline.metrics == baseline_repeat.metrics),
        ("baseline_health", _check_baseline_health(baseline.metrics)),
        ("spoof_separation", _check_spoof_separation(baseline.rows, spoof.rows)),
        ("jam_separation", _check_jam_separation(baseline.metrics, jam.metrics)),
    ]

    _print_results(checks)
    if any(not passed for _, passed in checks):
        sys.exit(1)


def _check_baseline_health(metrics: dict[str, float]) -> bool:
    return metrics["fix_valid_rate"] > 0.7 and metrics["sats_used_min"] >= 6.0


def _check_spoof_separation(
    baseline_rows: list[dict[str, str]],
    spoof_rows: list[dict[str, str]],
) -> bool:
    attack_start_t = 10.0
    after_start_t = max(attack_start_t + 2.0, WARMUP_S)
    baseline_metrics = summary_metrics(baseline_rows, start_t=after_start_t)
    spoof_metrics = summary_metrics(spoof_rows, start_t=after_start_t)
    return (
        spoof_metrics["nis_alarm_rate_after_start"] > baseline_metrics["nis_alarm_rate_after_start"]
        or _position_gap_m(spoof_rows, baseline_rows) > 1e-8
    )


def _position_gap_m(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]]) -> float:
    ref = np.array(
        [
            float(rows_b[0]["pos_ecef_x"]),
            float(rows_b[0]["pos_ecef_y"]),
            float(rows_b[0]["pos_ecef_z"]),
        ],
        dtype=float,
    )
    return abs(position_rms_m(rows_a, ref) - position_rms_m(rows_b, ref))


def _check_jam_separation(baseline_metrics: dict[str, float], jam_metrics: dict[str, float]) -> bool:
    sats_drop = jam_metrics["sats_used_min"] <= baseline_metrics["sats_used_min"] - 2.0
    fix_drop = jam_metrics["fix_valid_rate"] <= baseline_metrics["fix_valid_rate"] - 0.2
    return sats_drop or fix_drop


def _print_results(checks: list[tuple[str, bool]]) -> None:
    print("Phase 3 Exit Checklist")
    print("-" * 30)
    for name, passed in checks:
        print(f"{name:20s} : {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
