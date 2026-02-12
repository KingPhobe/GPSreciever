"""Smoke test for build_engine_with_truth helper used by desktop GUI."""

from gnss_twin.config import SimConfig
from sim.run_static_demo import build_engine_with_truth, build_epoch_log


def test_build_engine_with_truth_smoke() -> None:
    cfg = SimConfig(duration=2.0, dt=1.0, rng_seed=42, use_ekf=True)
    engine, receiver_truth_state = build_engine_with_truth(cfg)

    assert receiver_truth_state.pos_ecef_m.shape == (3,)

    for t_s in [0.0, 1.0]:
        step_out = engine.step(t_s)
        for key in ["sol", "integrity", "meas_attacked"]:
            assert key in step_out

        epoch = build_epoch_log(
            t_s=t_s,
            step_out=step_out,
            receiver_truth_state=receiver_truth_state,
            integrity_checker=engine.integrity_checker,
            attack_name=cfg.attack_name,
        )
        assert epoch.truth is receiver_truth_state
