from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_demo(tmp_path: Path, run_name: str) -> Path:
    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "sim.run_static_demo",
        "--duration-s",
        "5",
        "--out-dir",
        str(out_dir),
        "--run-name",
        run_name,
        "--use-ekf",
        "--no-plots",
        "--rng-seed",
        "123",
    ]
    subprocess.run(cmd, check=True)
    return out_dir / run_name / "epoch_logs.csv"


def test_rng_seed_cli_determinism(tmp_path: Path) -> None:
    first_path = _run_demo(tmp_path, "reproA")
    second_path = _run_demo(tmp_path, "reproB")

    assert first_path.read_bytes() == second_path.read_bytes()
