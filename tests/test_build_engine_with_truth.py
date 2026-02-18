"""Smoke test for build_engine_with_truth helper used by desktop GUI."""

from gnss_twin.config import SimConfig
from gnss_twin.runtime.factory import build_engine_with_truth, build_epoch_log


def test_build_engine_with_truth_smoke() -> None:
    cfg = SimConfig(duration=2.0, dt=1.0, rng_seed=42, use_ekf=True)
    engine, receiver_truth_state = build_engine_with_truth(cfg)

    assert receiver_truth_state.pos_ecef_m.shape == (3,)

    epoch_truth_biases = []
    source_truth_biases = []

    for t_s in [0.0, 1.0]:
        step_out = engine.step(t_s)
        for key in ["sol", "integrity", "meas_attacked", "rx_truth"]:
            assert key in step_out

        epoch = build_epoch_log(
            t_s=t_s,
            step_out=step_out,
            integrity_checker=engine.integrity_checker,
            attack_name=cfg.attack_name,
        )
        assert epoch.truth is step_out["rx_truth"]
        assert epoch.truth is not receiver_truth_state
        assert epoch.truth is not None
        epoch_truth_biases.append(epoch.truth.clk_bias_s)
        source_truth_biases.append(step_out["rx_truth"].clk_bias_s)

    assert epoch_truth_biases == source_truth_biases
    assert epoch_truth_biases[1] != epoch_truth_biases[0]
