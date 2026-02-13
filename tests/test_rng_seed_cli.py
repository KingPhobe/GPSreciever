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
    return out_dir / run_name


def test_rng_seed_cli_determinism(tmp_path: Path) -> None:
    first_run_dir = _run_demo(tmp_path, "reproA")
    second_run_dir = _run_demo(tmp_path, "reproB")
    first_path = first_run_dir / "epoch_logs.csv"
    second_path = second_run_dir / "epoch_logs.csv"

    assert first_path.read_bytes() == second_path.read_bytes()


def test_rng_seed_cli_writes_nmea_and_metadata(tmp_path: Path) -> None:
    run_dir = _run_demo(tmp_path, "nmea")
    nmea_path = run_dir / "nmea_output.nmea"
    metadata_path = run_dir / "run_metadata.csv"

    assert nmea_path.exists()
    nmea_text = nmea_path.read_text(encoding="utf-8")
    assert "GNGGA" in nmea_text
    assert "GNRMC" in nmea_text
    assert metadata_path.exists()
