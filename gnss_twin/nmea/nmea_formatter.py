"""Minimal NMEA 0183 formatter for GGA and RMC messages."""

from __future__ import annotations

from datetime import datetime


def nmea_checksum(payload: str) -> str:
    """Return the NMEA XOR checksum as a 2-digit uppercase hex string."""
    checksum = 0
    for char in payload:
        checksum ^= ord(char)
    return f"{checksum:02X}"


def wrap_sentence(payload: str) -> str:
    """Wrap a payload as a full NMEA sentence with checksum and CRLF."""
    return f"${payload}*{nmea_checksum(payload)}\r\n"


def format_lat(lat_deg: float) -> tuple[str, str]:
    """Format latitude as ddmm.mmmm and hemisphere indicator."""
    hemisphere = "N" if lat_deg >= 0 else "S"
    lat_abs = abs(lat_deg)
    degrees = int(lat_abs)
    minutes = (lat_abs - degrees) * 60.0
    return f"{degrees:02d}{minutes:07.4f}", hemisphere


def format_lon(lon_deg: float) -> tuple[str, str]:
    """Format longitude as dddmm.mmmm and hemisphere indicator."""
    hemisphere = "E" if lon_deg >= 0 else "W"
    lon_abs = abs(lon_deg)
    degrees = int(lon_abs)
    minutes = (lon_abs - degrees) * 60.0
    return f"{degrees:03d}{minutes:07.4f}", hemisphere


def format_time_hhmmss(t_utc: datetime) -> str:
    """Format UTC datetime as hhmmss.ss."""
    hundredths = round(t_utc.microsecond / 10000.0)
    second = t_utc.second
    minute = t_utc.minute
    hour = t_utc.hour

    if hundredths >= 100:
        hundredths = 0
        second += 1
        if second >= 60:
            second = 0
            minute += 1
            if minute >= 60:
                minute = 0
                hour = (hour + 1) % 24

    return f"{hour:02d}{minute:02d}{second:02d}.{hundredths:02d}"


def format_date_ddmmyy(t_utc: datetime) -> str:
    """Format UTC datetime as ddmmyy."""
    return t_utc.strftime("%d%m%y")


def build_gga(
    t_utc: datetime,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    valid: bool,
    num_sats: int = 8,
    hdop: float = 0.9,
    geoid_sep_m: float = 0.0,
    talker: str = "GN",
) -> str:
    """Build an NMEA GGA sentence with checksum and CRLF terminator."""
    lat_str, ns = format_lat(lat_deg)
    lon_str, ew = format_lon(lon_deg)
    time_str = format_time_hhmmss(t_utc)
    fix_quality = 1 if valid else 0

    payload = (
        f"{talker}GGA,{time_str},{lat_str},{ns},{lon_str},{ew},{fix_quality},"
        f"{num_sats:02d},{hdop:.1f},{alt_m:.1f},M,{geoid_sep_m:.1f},M,,"
    )
    return wrap_sentence(payload)


def build_rmc(
    t_utc: datetime,
    lat_deg: float,
    lon_deg: float,
    speed_knots: float,
    course_deg: float,
    valid: bool,
    talker: str = "GN",
) -> str:
    """Build an NMEA RMC sentence with checksum and CRLF terminator."""
    lat_str, ns = format_lat(lat_deg)
    lon_str, ew = format_lon(lon_deg)
    time_str = format_time_hhmmss(t_utc)
    date_str = format_date_ddmmyy(t_utc)
    status = "A" if valid else "V"
    mode = "A" if valid else "N"

    payload = (
        f"{talker}RMC,{time_str},{status},{lat_str},{ns},{lon_str},{ew},"
        f"{speed_knots:.2f},{course_deg:.2f},{date_str},,,{mode}"
    )
    return wrap_sentence(payload)
