from __future__ import annotations

from pathlib import Path

from gnss_twin.config import SimConfig
from gnss_twin.logger import load_epochs_npz
from sim.run_static_demo import run_static_demo


def _run_demo(tmp_path: Path, *, attack_name: str, attack_params: dict[str, float | str]) -> list[dict]:
    cfg = SimConfig(
        duration=20.0,
        use_ekf=True,
        attack_name=attack_name,
        attack_params=attack_params,
    )
    epoch_log_path = run_static_demo(cfg, tmp_path / f"nis_{attack_name}", save_figs=False)
    return load_epochs_npz(epoch_log_path.parent / "epoch_logs.npz")


def test_nis_alarm_nominal_stays_quiet(tmp_path: Path) -> None:
    epochs = _run_demo(tmp_path, attack_name="none", attack_params={})
    alarm_count = sum(1 for epoch in epochs if epoch.get("nis_alarm"))
    assert alarm_count <= 1


def test_nis_alarm_trips_for_spoofing(tmp_path: Path) -> None:
    start_t = 0.0
    epochs = _run_demo(
        tmp_path,
        attack_name="spoof_pr_ramp",
        attack_params={"start_t": start_t, "ramp_rate_mps": 500.0, "target_sv": "G03"},
    )
    spoofed = [
        epoch for epoch in epochs if epoch.get("t", 0.0) >= start_t and epoch.get("nis_alarm")
    ]
    assert spoofed
