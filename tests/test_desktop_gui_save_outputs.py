from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gnss_twin.config import SimConfig
from gnss_twin.runtime.factory import build_engine_with_truth, build_epoch_log


def _app():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    app = qt.QApplication.instance()
    if app is None:
        app = qt.QApplication([])
    return app


@pytest.mark.gui
def test_save_outputs_generates_full_plot_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _app()
    gui = pytest.importorskip("sim.desktop_gui")

    window = gui.MainWindow()
    cfg = SimConfig(duration=3.0, dt=1.0)
    engine, truth = build_engine_with_truth(cfg)
    window.cfg = cfg
    window.engine = engine
    window.receiver_truth_state = truth

    for t_s in [0.0, 1.0, 2.0]:
        step = engine.step(t_s)
        epoch = build_epoch_log(
            t_s=t_s,
            step_out=step,
            receiver_truth_state=truth,
            integrity_checker=engine.integrity_checker,
            attack_name=cfg.attack_name or "none",
        )
        window.epochs.append(epoch)

    monkeypatch.chdir(tmp_path)
    window.run_name_input.setText("gui run 001")
    window.current_run_name = "unused"
    monkeypatch.setattr(gui, "_date_folder_str", lambda: "02132026")

    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args, **kwargs: None)

    window.save_outputs()

    run_dir = tmp_path / "out" / "02132026" / "gui_run_001"
    assert run_dir.is_dir()

    expected_files = {
        "run_table.csv",
        "position_error.png",
        "clock_bias.png",
        "residual_rms.png",
        "dop.png",
        "satellites_used.png",
        "fix_status.png",
        "attack_telemetry.png",
    }
    assert expected_files.issubset({p.name for p in run_dir.iterdir()})
    with (run_dir / "run_table.csv").open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    for expected_column in [
        "nmea_enabled",
        "nmea_profile",
        "nmea_rate_hz",
        "nmea_msgs",
        "nmea_talker",
        "rx_lat_deg",
        "rx_lon_deg",
        "rx_alt_m",
        "integrity_p_value",
        "integrity_num_sats_used",
        "integrity_excluded_sv_ids_count",
        "conops_reason_codes",
        "nis",
        "nis_alarm",
    ]:
        assert expected_column in header


@pytest.mark.gui
def test_save_outputs_uses_unique_directory_on_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _app()
    gui = pytest.importorskip("sim.desktop_gui")

    window = gui.MainWindow()
    cfg = SimConfig(duration=2.0, dt=1.0)
    engine, truth = build_engine_with_truth(cfg)
    window.cfg = cfg
    window.engine = engine
    window.receiver_truth_state = truth

    for t_s in [0.0, 1.0]:
        step = engine.step(t_s)
        epoch = build_epoch_log(
            t_s=t_s,
            step_out=step,
            receiver_truth_state=truth,
            integrity_checker=engine.integrity_checker,
            attack_name=cfg.attack_name or "none",
        )
        window.epochs.append(epoch)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui, "_date_folder_str", lambda: "02132026")
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args, **kwargs: None)
    window.run_name_input.setText("gui_run_001")

    first_dir = tmp_path / "out" / "02132026" / "gui_run_001"
    first_dir.mkdir(parents=True)

    window.save_outputs()

    assert (tmp_path / "out" / "02132026" / "gui_run_001_01").is_dir()
