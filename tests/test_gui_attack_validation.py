from __future__ import annotations

import pytest

from gnss_twin.config import SimConfig
from gnss_twin.runtime.factory import build_engine_with_truth


@pytest.mark.gui
def test_visible_svs_non_empty_and_supports_non_visible_choice() -> None:
    pytest.importorskip("PyQt6.QtWidgets")
    gui = pytest.importorskip("sim.desktop_gui")

    cfg = SimConfig(rng_seed=42)
    _, truth = build_engine_with_truth(cfg)

    visible = gui._visible_svs_at_time(10.0, truth, cfg.elev_mask_deg, cfg.rng_seed)
    assert visible

    all_sv_ids = [f"G{idx:02d}" for idx in range(1, 25)]
    non_visible = [sv for sv in all_sv_ids if sv not in visible]
    assert non_visible

    chosen, ok, msg = gui._resolve_target_sv(non_visible[0], visible, auto_select=True)
    assert ok
    assert chosen == visible[0]
    assert "auto_selected" in msg
