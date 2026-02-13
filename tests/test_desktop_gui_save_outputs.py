from __future__ import annotations

from pathlib import Path

import pytest

from gnss_twin.config import SimConfig
from sim.run_static_demo import build_engine_with_truth, build_epoch_log


def _app():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    app = qt.QApplication.instance()
    if app is None:
        app = qt.QApplication([])
    return app


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

    save_root = tmp_path / "saved"
    save_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(gui.QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(save_root))
    monkeypatch.setattr(gui.QMessageBox, "information", lambda *args, **kwargs: None)

    window.save_outputs()

    created = sorted([p for p in save_root.iterdir() if p.is_dir()])
    assert len(created) == 1
    run_dir = created[0]

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
