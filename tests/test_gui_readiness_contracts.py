"""GUI readiness contract tests."""

from gnss_twin.config import SimConfig
from gnss_twin.logger import EPOCH_CSV_COLUMNS, save_epochs_csv
from gnss_twin.runtime import Engine


def test_epoch_csv_contract(tmp_path) -> None:
    cfg = SimConfig(duration=5.0, dt=1.0)
    engine = Engine(cfg)
    engine.reset()

    epochs = engine.run(0.0, 5.0, 1.0)
    csv_path = tmp_path / "epoch_logs.csv"
    save_epochs_csv(csv_path, epochs)

    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == EPOCH_CSV_COLUMNS
