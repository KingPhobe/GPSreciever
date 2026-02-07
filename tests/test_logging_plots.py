from __future__ import annotations

from pathlib import Path

from gnss_twin.logger import load_epochs_npz
from sim.run_static_demo import run_demo


def test_demo_outputs_and_log_reload(tmp_path: Path) -> None:
    output_dir = run_demo(duration_s=5.0, out_dir=tmp_path, run_name="pytest")
    expected_files = {
        "epoch_logs.csv",
        "epoch_logs.npz",
        "position_error.png",
        "clock_bias.png",
        "residual_rms.png",
        "dop.png",
        "satellites_used.png",
        "fix_status.png",
    }
    assert output_dir.exists()
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    epochs = load_epochs_npz(output_dir / "epoch_logs.npz")
    assert len(epochs) == 5
    assert "t" in epochs[0]
