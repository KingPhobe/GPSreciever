"""Per-satellite tracking state placeholder."""

from __future__ import annotations

from dataclasses import dataclass

from gnss_twin.config import SimConfig
from gnss_twin.models import GnssMeasurement


@dataclass
class SvTrackState:
    locked: bool = False
    good_count: int = 0
    bad_count: int = 0
    last_cn0_dbhz: float | None = None


class TrackingState:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.states: dict[str, SvTrackState] = {}

    def update(self, measurements: list[GnssMeasurement]) -> dict[str, SvTrackState]:
        for meas in measurements:
            state = self.states.get(meas.sv_id)
            if state is None:
                state = SvTrackState()
                self.states[meas.sv_id] = state

            state.last_cn0_dbhz = float(meas.cn0_dbhz)
            if meas.cn0_dbhz >= self.cfg.cn0_lock_on_dbhz:
                state.good_count += 1
                state.bad_count = 0
            elif meas.cn0_dbhz < self.cfg.cn0_lock_off_dbhz:
                state.bad_count += 1
                state.good_count = 0

            if state.good_count >= self.cfg.n_good_to_lock:
                state.locked = True
            if state.bad_count >= self.cfg.n_bad_to_unlock:
                state.locked = False

        return self.states
