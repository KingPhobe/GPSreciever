from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gnss_twin.config import SimConfig
from sim.run_static_demo import run_static_demo


def _read_nmea_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_nmea_csv_valid_is_one_when_raim_passes(tmp_path: Path) -> None:
    run_dir = tmp_path / "pass"
    cfg = SimConfig(duration=4.0, dt=1.0, attack_name="none", rng_seed=123)
    run_static_demo(cfg, run_dir, save_figs=False)

    rows = _read_nmea_csv(run_dir / "nmea_output.csv")
    assert rows
    valid_values = {int(row["valid"]) for row in rows}
    assert valid_values.issubset({0, 1})
    assert 1 in valid_values
    for row in rows:
        assert "\r" not in row["nmea_sentence"]
        assert "\n" not in row["nmea_sentence"]


def test_nmea_csv_valid_drops_to_zero_when_raim_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import gnss_twin.integrity.flags as flags_mod

    original_compute_raim = flags_mod.compute_raim

    def _forced_fail_raim(*args, **kwargs):
        t_stat, dof, threshold, _ = original_compute_raim(*args, **kwargs)
        return t_stat, dof, threshold, False

    monkeypatch.setattr(flags_mod, "compute_raim", _forced_fail_raim)

    run_dir = tmp_path / "fail"
    cfg = SimConfig(
        duration=6.0,
        dt=1.0,
        attack_name="spoof_clock_ramp",
        attack_params={"start_t": 1.0, "end_t": 5.0, "ramp_rate_mps": 200.0},
        rng_seed=123,
    )
    run_static_demo(cfg, run_dir, save_figs=False)

    rows = _read_nmea_csv(run_dir / "nmea_output.csv")
    assert rows
    valid_values = {int(row["valid"]) for row in rows}
    assert valid_values.issubset({0, 1})
    assert 0 in valid_values
    for row in rows:
        assert "\r" not in row["nmea_sentence"]
        assert "\n" not in row["nmea_sentence"]
