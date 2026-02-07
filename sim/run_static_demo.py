"""Run a minimal static GNSS twin demo."""

from __future__ import annotations

import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.visibility import apply_elevation_mask


def main() -> None:
    config = SimConfig(seed=7, dt=1.0, duration=10.0, elev_mask_deg=10.0)
    receiver_ecef = np.array([6_378_000.0, 0.0, 0.0])

    constellation = SimpleGpsConstellation(SimpleGpsConfig(num_sats=24, seed=config.seed))

    t = 0.0
    while t <= config.duration:
        sv_states = constellation.get_sv_states(t)
        visible = apply_elevation_mask(receiver_ecef, sv_states, config.elev_mask_deg)
        print(f"t={t:5.1f}s: {len(visible)} visible satellites")
        t += config.dt


if __name__ == "__main__":
    main()
