from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from gnss_twin.config import SimConfig

NMEA_RUN_TABLE_COLUMNS = {
    "nmea_enabled": True,
    "nmea_profile": "NEO-M8N",
    "nmea_rate_hz": 1.0,
    "nmea_msgs": "GGA,RMC",
    "nmea_talker": "GN",
}


def run_table_nmea_metadata(cfg: SimConfig) -> dict[str, Any]:
    return {
        **NMEA_RUN_TABLE_COLUMNS,
        "rx_lat_deg": cfg.rx_lat_deg,
        "rx_lon_deg": cfg.rx_lon_deg,
        "rx_alt_m": cfg.rx_alt_m,
    }


def run_table_sim_metadata(cfg: SimConfig) -> dict[str, Any]:
    """Configuration metadata duplicated onto every row of run_table.csv.

    Prefixed with `cfg_` to avoid collisions with per-epoch columns.
    """

    attack_name = cfg.attack_name or "none"
    params = cfg.attack_params or {}
    return {
        "cfg_duration_s": float(cfg.duration),
        "cfg_dt_s": float(cfg.dt),
        "cfg_rng_seed": int(cfg.rng_seed),
        "cfg_use_ekf": bool(cfg.use_ekf),
        "cfg_attack_name": attack_name,
        "cfg_attack_start_t_s": params.get("start_t", ""),
        "cfg_attack_end_t_s": params.get("end_t", ""),
        "cfg_attack_ramp_rate_mps": params.get("ramp_rate_mps", ""),
        "cfg_attack_target_sv": params.get("target_sv", "") if attack_name == "spoof_pr_ramp" else "",
        "cfg_attack_pos_north_m": params.get("north_m", ""),
        "cfg_attack_pos_east_m": params.get("east_m", ""),
        "cfg_attack_pos_up_m": params.get("up_m", ""),
        "cfg_attack_pos_ramp_time_s": params.get("ramp_time_s", ""),
        "cfg_jam_cn0_drop_db": params.get("cn0_drop_db", ""),
        "cfg_jam_sigma_pr_scale": params.get("sigma_pr_scale", ""),
        "cfg_jam_sigma_prr_scale": params.get("sigma_prr_scale", ""),
    }


def add_nmea_metadata_columns(frame: Any, cfg: SimConfig) -> Any:
    metadata = {**run_table_nmea_metadata(cfg), **run_table_sim_metadata(cfg)}
    for key, value in metadata.items():
        frame[key] = value
    return frame


def write_run_table_from_epoch_logs(epoch_logs_path: Path, run_table_path: Path, cfg: SimConfig) -> None:
    metadata = {**run_table_nmea_metadata(cfg), **run_table_sim_metadata(cfg)}
    with epoch_logs_path.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        base_fieldnames = list(reader.fieldnames or [])
        fieldnames = base_fieldnames + [k for k in metadata.keys() if k not in base_fieldnames]
        with run_table_path.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow({**row, **metadata})
