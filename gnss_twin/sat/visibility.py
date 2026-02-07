"""Satellite visibility filtering utilities."""

from __future__ import annotations

import numpy as np

from gnss_twin.models import SvState
from gnss_twin.utils.angles import elev_az_from_rx_sv


def visible_sv_states(
    receiver_ecef_m: np.ndarray,
    sv_states: list[SvState],
    elevation_mask_deg: float = 10.0,
) -> list[SvState]:
    """Filter satellite states by elevation mask."""

    visible: list[SvState] = []
    for state in sv_states:
        elev_deg, _ = elev_az_from_rx_sv(receiver_ecef_m, state.pos_ecef_m)
        if elev_deg >= elevation_mask_deg:
            visible.append(state)
    return visible
