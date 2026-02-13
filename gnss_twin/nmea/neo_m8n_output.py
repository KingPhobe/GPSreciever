"""Minimal NEO-M8N NMEA output driver."""

from __future__ import annotations

from datetime import datetime
import math

from .nmea_formatter import build_gga, build_rmc


class NeoM8nNmeaOutput:
    """Emit NMEA sentences at a fixed rate, approximating NEO-M8N output."""

    def __init__(
        self,
        rate_hz: float = 1.0,
        talker: str = "GN",
        enable_gga: bool = True,
        enable_rmc: bool = True,
    ) -> None:
        self.rate_hz = rate_hz
        self.talker = talker
        self.enable_gga = enable_gga
        self.enable_rmc = enable_rmc
        self.period_s = 1.0 / rate_hz
        self.next_emit_t_s = 0.0

    def reset(self) -> None:
        """Reset emission schedule to start from t=0."""
        self.next_emit_t_s = 0.0

    def step(
        self,
        t_s: float,
        *,
        t_utc: datetime,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        valid: bool,
        num_sats: int,
        hdop: float,
        speed_kn: float = 0.0,
        course_deg: float = 0.0,
    ) -> list[str]:
        """Return sentences for this step if the emit boundary is reached."""
        if t_s < self.next_emit_t_s:
            return []

        safe_num_sats = max(0, int(num_sats))
        safe_hdop = hdop if math.isfinite(hdop) else 99.9
        fix_quality_valid = bool(valid)

        sentences: list[str] = []

        if self.enable_gga:
            sentences.append(
                build_gga(
                    t_utc=t_utc,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    alt_m=alt_m,
                    valid=fix_quality_valid,
                    num_sats=safe_num_sats,
                    hdop=safe_hdop,
                    talker=self.talker,
                )
            )

        if self.enable_rmc:
            sentences.append(
                build_rmc(
                    t_utc=t_utc,
                    lat_deg=lat_deg,
                    lon_deg=lon_deg,
                    speed_knots=speed_kn,
                    course_deg=course_deg,
                    valid=fix_quality_valid,
                    talker=self.talker,
                )
            )

        self.next_emit_t_s += self.period_s
        return sentences
