"""Basic smoke tests."""

from gnss_twin.config import SimConfig
from gnss_twin.runtime.solver import DefaultPvtSolver


def test_imports() -> None:
    config = SimConfig()
    assert config.dt == 1.0


def test_active_runtime_solver_import() -> None:
    assert DefaultPvtSolver.__name__ == "DefaultPvtSolver"
