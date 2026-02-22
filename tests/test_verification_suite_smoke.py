"""Smoke test for the verification suite.

Intentionally light-weight: ensures runner executes and baseline metrics are finite.
"""

from pathlib import Path

from sim.validation.verification_suite import run_quick_verification


def test_quick_verification_runs(tmp_path: Path) -> None:
    report = run_quick_verification(run_root=tmp_path)
    assert "results" in report
    assert "normal" in report["results"]
    metrics = report["results"]["normal"]["metrics"]
    assert metrics["pos_err_rms_m"] == metrics["pos_err_rms_m"]  # not NaN
