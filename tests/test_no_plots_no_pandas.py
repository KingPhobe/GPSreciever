from __future__ import annotations

import sys
from pathlib import Path

from gnss_twin.config import SimConfig
from sim.run_static_demo import run_static_demo


def test_no_plots_does_not_import_plot_module(tmp_path: Path) -> None:
    sys.modules.pop("gnss_twin.plots", None)
    cfg = SimConfig(duration=1.0)
    run_static_demo(cfg, tmp_path / "no_plots", save_figs=False)
    assert "gnss_twin.plots" not in sys.modules


def test_no_plots_has_quiet_stdout_unless_verbose(tmp_path: Path, capsys) -> None:
    cfg = SimConfig(duration=1.0)
    run_static_demo(cfg, tmp_path / "quiet", save_figs=False)
    quiet_out = capsys.readouterr().out
    assert "First-epoch pseudoranges" not in quiet_out

    run_static_demo(cfg, tmp_path / "verbose", save_figs=False, verbose=True)
    verbose_out = capsys.readouterr().out
    assert "First-epoch pseudoranges" in verbose_out
