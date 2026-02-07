"""Basic smoke tests."""

from gnss_twin.config import SimConfig


def test_imports() -> None:
    config = SimConfig()
    assert config.dt == 1.0
