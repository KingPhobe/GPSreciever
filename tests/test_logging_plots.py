from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gnss_twin.config import SimConfig
from gnss_twin.logger import load_epochs_npz
from gnss_twin.plots import epochs_to_frame
from sim.run_static_demo import run_static_demo


def test_demo_outputs_and_log_reload(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    cfg = SimConfig(duration=5.0)
    epoch_log_path = run_static_demo(cfg, tmp_path / "pytest", save_figs=True)
    output_dir = epoch_log_path.parent
    expected_files = {
        "epoch_logs.csv",
        "epoch_logs.npz",
        "position_error.png",
        "clock_bias.png",
        "residual_rms.png",
        "dop.png",
        "satellites_used.png",
        "fix_status.png",
        "attack_telemetry.png",
        "conops_status_timeline.png",
        "integrity_residual_rms_timeline.png",
    }
    assert output_dir.exists()
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    epochs = load_epochs_npz(output_dir / "epoch_logs.npz")
    assert len(epochs) == 5
    assert "t" in epochs[0]
    assert "conops_status" in epochs[0]
    assert "integrity_p_value" in epochs[0]

    with (output_dir / "epoch_logs.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
    assert "conops_status" in columns
    assert "conops_mode5" in columns
    assert "conops_reason_codes" in columns
    assert "integrity_p_value" in columns
    assert "integrity_residual_rms" in columns
    assert "integrity_num_sats_used" in columns
    assert "integrity_excluded_sv_ids_count" in columns


def test_epochs_to_frame_includes_attack_columns(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    cfg = SimConfig(
        duration=3.0,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 0.0, "ramp_rate_mps": 10.0},
    )
    epoch_log_path = run_static_demo(cfg, tmp_path / "attack_frame", save_figs=False)
    epochs = load_epochs_npz(epoch_log_path)
    frame = epochs_to_frame(epochs)

    for column in [
        "attack_name",
        "attack_active",
        "attack_pr_bias_mean_m",
        "attack_prr_bias_mean_mps",
    ]:
        assert column in frame.columns

    assert frame["attack_name"].eq("spoof_clock_ramp").all()
    assert frame["attack_active"].any()
