"""Offline validation report generator.

Usage:
  python -m sim.validation.offline_report --run-dir <run_output_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _stats(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "rms": float("nan"), "max": float("nan"), "p95": float("nan")}
    return {
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values**2))),
        "max": float(values.max()),
        "p95": float(np.percentile(values, 95)),
    }


def build_report(run_dir: Path) -> dict:
    epoch_path = run_dir / "epoch_logs.csv"
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch_logs.csv in {run_dir}")

    df = pd.read_csv(epoch_path)
    report: dict[str, object] = {
        "run_dir": str(run_dir),
        "epochs": int(len(df)),
        "attack_name": str(df.get("attack_name", pd.Series([""])).iloc[0] if len(df) else ""),
        "raim_pass_rate": float(pd.to_numeric(df.get("raim_pass", 0), errors="coerce").fillna(0).mean()),
        "fix_valid_rate": float(pd.to_numeric(df.get("fix_valid", 0), errors="coerce").fillna(0).mean()),
        "nis_alarm_rate": float(pd.to_numeric(df.get("nis_alarm", 0), errors="coerce").fillna(0).mean()),
        "position_error_m": _stats(df.get("pos_error_m", pd.Series([], dtype=float))),
        "clock_bias_error_s": _stats(df.get("clk_bias_error_s", pd.Series([], dtype=float))),
        "residual_rms_m": _stats(df.get("residual_rms_m", pd.Series([], dtype=float))),
        "pdop": _stats(df.get("pdop", pd.Series([], dtype=float))),
        "sats_used": _stats(df.get("sats_used", pd.Series([], dtype=float))),
        "attack_pr_bias_mean_m": _stats(df.get("attack_pr_bias_mean_m", pd.Series([], dtype=float))),
        "attack_prr_bias_mean_mps": _stats(df.get("attack_prr_bias_mean_mps", pd.Series([], dtype=float))),
    }

    meas_path = run_dir / "meas_log.csv"
    if meas_path.exists():
        md = pd.read_csv(meas_path)
        report["meas_rows"] = int(len(md))
        report["meas_pr_bias_m"] = _stats(md.get("pr_bias_m", pd.Series([], dtype=float)))
        report["meas_used_rate"] = float(pd.to_numeric(md.get("used_in_solution", 0), errors="coerce").fillna(0).mean())
        report["meas_cn0_dbhz"] = _stats(md.get("cn0_dbhz", pd.Series([], dtype=float)))

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate an offline validation report for a GNSS twin run.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    report = build_report(run_dir)
    out_path = Path(args.out) if args.out else (run_dir / "validation_report.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
