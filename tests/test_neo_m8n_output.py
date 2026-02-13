from datetime import datetime, timedelta, timezone

from gnss_twin.nmea.neo_m8n_output import NeoM8nNmeaOutput


def test_neo_m8n_output_emits_once_per_second_including_t0():
    driver = NeoM8nNmeaOutput(rate_hz=1.0, talker="GN", enable_gga=True, enable_rmc=True)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

    emissions = []
    for second in range(4):
        emissions.extend(
            driver.step(
                float(second),
                t_utc=t0 + timedelta(seconds=second),
                lat_deg=37.0,
                lon_deg=-122.0,
                alt_m=10.0,
                raim_valid=True,
                num_sats=9,
                hdop=0.8,
                speed_kn=1.2,
                course_deg=45.0,
            )
        )

    assert len(emissions) == 8
    assert sum("$GNGGA" in emit.nmea_sentence for emit in emissions) == 4
    assert sum("$GNRMC" in emit.nmea_sentence for emit in emissions) == 4
    assert all(emit.valid == 1 for emit in emissions)


def test_neo_m8n_output_sanitizes_satellite_count_and_hdop():
    driver = NeoM8nNmeaOutput(enable_rmc=False)
    t_utc = datetime(2024, 1, 1, tzinfo=timezone.utc)

    lines = driver.step(
        0.0,
        t_utc=t_utc,
        lat_deg=0.0,
        lon_deg=0.0,
        alt_m=0.0,
        raim_valid=False,
        num_sats=-3,
        hdop=float("nan"),
    )

    assert len(lines) == 1
    gga = lines[0]
    assert "$GNGGA" in gga.nmea_sentence
    assert ",0,00,99.9," in gga.nmea_sentence
    assert gga.valid == 0
