from __future__ import annotations

import pytest


def _app():
    qt = pytest.importorskip("PyQt6.QtWidgets")
    app = qt.QApplication.instance()
    if app is None:
        app = qt.QApplication([])
    return app


def test_main_window_defaults_to_flags_plot_only() -> None:
    _app()
    gui = pytest.importorskip("sim.desktop_gui")
    window = gui.MainWindow()

    assert hasattr(window, "ax_flags")
    assert window.diagnostics_window is None
    assert window.figure.axes == [window.ax_flags]


def test_diagnostics_window_created_on_demand() -> None:
    _app()
    gui = pytest.importorskip("sim.desktop_gui")
    window = gui.MainWindow()

    window.open_diagnostics()

    assert window.diagnostics_window is not None
    assert len(window.diagnostics_window.figure.axes) == 6
