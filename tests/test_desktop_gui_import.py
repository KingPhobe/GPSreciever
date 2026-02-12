import pytest


def test_desktop_gui_import() -> None:
    pytest.importorskip("PyQt6")
    import sim.desktop_gui  # noqa: F401
