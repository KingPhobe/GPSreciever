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


def add_nmea_metadata_columns(frame: Any, cfg: SimConfig) -> Any:
    metadata = run_table_nmea_metadata(cfg)
    for key, value in metadata.items():
        frame[key] = value
    return frame


def write_run_table_from_epoch_logs(epoch_logs_path: Path, run_table_path: Path, cfg: SimConfig) -> None:
    metadata = run_table_nmea_metadata(cfg)
    with epoch_logs_path.open("r", encoding="utf-8", newline="") as in_file:
        reader = csv.DictReader(in_file)
        base_fieldnames = list(reader.fieldnames or [])
        fieldnames = base_fieldnames + list(metadata.keys())
        with run_table_path.open("w", encoding="utf-8", newline="") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow({**row, **metadata})
