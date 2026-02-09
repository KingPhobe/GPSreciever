import numpy as np

from gnss_twin.config import SimConfig
from gnss_twin.models import GnssMeasurement
from gnss_twin.receiver.tracking_state import TrackingState


def _measurement(t: float, cn0_dbhz: float, sv_id: str = "G01") -> GnssMeasurement:
    return GnssMeasurement(
        sv_id=sv_id,
        t=t,
        pr_m=0.0,
        prr_mps=None,
        sigma_pr_m=1.0,
        cn0_dbhz=cn0_dbhz,
        elev_deg=0.0,
        az_deg=0.0,
        flags={},
    )


def test_tracking_locks_after_good_epochs() -> None:
    config = SimConfig()
    tracker = TrackingState(config)
    state = None
    for t in range(3):
        state = tracker.update([_measurement(float(t), 35.0)])
    assert state is not None
    assert state["G01"].locked is True


def test_tracking_unlocks_after_bad_epochs() -> None:
    config = SimConfig()
    tracker = TrackingState(config)
    for t in range(3):
        tracker.update([_measurement(float(t), 35.0)])
    state = None
    for t in range(3, 5):
        state = tracker.update([_measurement(float(t), 20.0)])
    assert state is not None
    assert state["G01"].locked is False


def test_tracking_deterministic_with_seed() -> None:
    def run_sequence(seed: int) -> list[bool]:
        config = SimConfig(rng_seed=seed)
        rng = np.random.default_rng(seed)
        tracker = TrackingState(config)
        locked_states: list[bool] = []
        for t, cn0_dbhz in enumerate(rng.uniform(20.0, 40.0, size=6)):
            states = tracker.update([_measurement(float(t), float(cn0_dbhz))])
            locked_states.append(states["G01"].locked)
        return locked_states

    assert run_sequence(123) == run_sequence(123)
