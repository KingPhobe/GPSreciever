from __future__ import annotations

from pathlib import Path

from gnss_twin.logger import load_epochs_npz
from sim.run_static_demo import run_demo


def _run_demo(tmp_path: Path, *, attack_name: str, attack_params: dict[str, float]) -> list[dict]:
    output_dir = run_demo(
        duration_s=20.0,
        out_dir=tmp_path,
        run_name=f"nis_{attack_name}",
        use_ekf=True,
        attack_name=attack_name,
        attack_params=attack_params,
    )
    return load_epochs_npz(output_dir / "epoch_logs.npz")


def test_nis_alarm_nominal_stays_quiet(tmp_path: Path) -> None:
    epochs = _run_demo(tmp_path, attack_name="none", attack_params={})
    alarm_count = sum(1 for epoch in epochs if epoch.get("nis_alarm"))
    assert alarm_count <= 1


def test_nis_alarm_trips_for_spoofing(tmp_path: Path) -> None:
    start_t = 2.0
    epochs = _run_demo(
        tmp_path,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": start_t, "ramp_rate_mps": 100.0},
    )
    spoofed = [
        epoch for epoch in epochs if epoch.get("t", 0.0) >= start_t and epoch.get("nis_alarm")
    ]
    assert spoofed
