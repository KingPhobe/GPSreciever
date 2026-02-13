from datetime import datetime, timezone

from gnss_twin.nmea.nmea_formatter import build_gga, build_rmc, nmea_checksum, wrap_sentence


def test_wrap_sentence_format():
    sentence = wrap_sentence("GNGGA,123519")
    assert sentence.startswith("$")
    assert "*" in sentence
    assert sentence.endswith("\r\n")


def test_checksum_is_two_hex_chars():
    checksum = nmea_checksum("GNRMC,123519")
    assert len(checksum) == 2
    assert checksum.upper() == checksum
    int(checksum, 16)


def test_build_gga_and_rmc_have_talker_and_checksum():
    t_utc = datetime(2024, 1, 2, 3, 4, 5, 670000, tzinfo=timezone.utc)

    gga = build_gga(t_utc, 37.7749, -122.4194, 15.2, True)
    rmc = build_rmc(t_utc, 37.7749, -122.4194, 0.5, 180.0, True)

    assert gga.startswith("$GN")
    assert rmc.startswith("$GN")
    assert "*" in gga
    assert "*" in rmc
    assert gga.endswith("\r\n")
    assert rmc.endswith("\r\n")
