from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gnss_twin.config import SimConfig
from gnss_twin.logger import load_epochs_npz
from gnss_twin.plots import epochs_to_frame, plot_update
from sim.run_static_demo import build_engine_with_truth, build_epoch_log, run_static_demo
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
    engine, truth = build_engine_with_truth(cfg)
    epochs = []
    for t_s in [0.0, 1.0, 2.0]:
        step = engine.step(t_s)
        epoch = build_epoch_log(
            t_s=t_s,
            step_out=step,
            receiver_truth_state=truth,
            integrity_checker=engine.integrity_checker,
            attack_name=cfg.attack_name or "none",
        )
        epochs.append(epoch)
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



def test_epochs_to_frame_includes_integrity_columns(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    cfg = SimConfig(duration=1.0)
    engine, truth = build_engine_with_truth(cfg)
    step = engine.step(0.0)
    epoch = build_epoch_log(
        t_s=0.0,
        step_out=step,
        receiver_truth_state=truth,
        integrity_checker=engine.integrity_checker,
        attack_name=cfg.attack_name or "none",
    )
    epoch = epoch.__class__(
        **{
            **epoch.__dict__,
            "nis": 1.25,
            "nis_alarm": True,
            "conops_reason_codes": ["excluded_svs", "nis_high"],
            "integrity_p_value": 0.0123,
            "integrity_num_sats_used": 5,
            "integrity_excluded_sv_ids_count": 2,
        }
    )

    frame = epochs_to_frame([epoch])

    assert frame.loc[0, "integrity_p_value"] == pytest.approx(0.0123)
    assert frame.loc[0, "integrity_num_sats_used"] == pytest.approx(5.0)
    assert frame.loc[0, "integrity_excluded_sv_ids_count"] == pytest.approx(2.0)
    assert frame.loc[0, "conops_reason_codes"] == "excluded_svs|nis_high"
    assert frame.loc[0, "nis"] == pytest.approx(1.25)
    assert bool(frame.loc[0, "nis_alarm"]) is True

def test_plot_update_writes_pngs_to_output_dir(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "t_s": [0.0, 1.0, 2.0],
            "pos_error_m": [1.0, 2.0, 3.0],
            "clk_bias_s": [0.0, 0.1, 0.2],
            "residual_rms_m": [0.5, 0.4, 0.3],
            "gdop": [2.0, 2.1, 2.2],
            "pdop": [1.5, 1.6, 1.7],
            "hdop": [1.2, 1.3, 1.4],
            "vdop": [1.1, 1.2, 1.3],
            "sats_used": [7, 8, 8],
            "fix_type": [2.0, 2.0, 2.0],
            "fix_valid": [1.0, 1.0, 1.0],
            "attack_active": [0.0, 1.0, 1.0],
            "attack_pr_bias_mean_m": [0.0, 3.0, 4.0],
            "attack_prr_bias_mean_mps": [0.0, 0.2, 0.4],
        }
    )

    plot_update(frame, out_dir=tmp_path, run_name=None)

    expected_files = {
        "position_error.png",
        "clock_bias.png",
        "residual_rms.png",
        "dop.png",
        "satellites_used.png",
        "fix_status.png",
        "attack_telemetry.png",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})
