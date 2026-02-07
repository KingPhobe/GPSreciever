import numpy as np

from gnss_twin.meas.pseudorange import LIGHT_SPEED_MPS, geometric_range_m, pseudorange_m


def test_pseudorange_matches_range_and_clock_terms() -> None:
    receiver = np.array([0.0, 0.0, 0.0])
    sv_pos = np.array([3.0e7, 4.0e7, 0.0])
    receiver_clock = 1.0e-6
    sv_clock = -2.0e-6

    expected_range = geometric_range_m(receiver, sv_pos)
    expected_clock = LIGHT_SPEED_MPS * (receiver_clock - sv_clock)

    assert np.isclose(expected_range, 5.0e7)
    assert np.isclose(pseudorange_m(receiver, receiver_clock, sv_pos, sv_clock), expected_range + expected_clock)
