import numpy as np

from gnss_twin.models import GnssMeasurement, ReceiverTruth
from gnss_twin.runtime.engine import SimulationEngine


class _MeasSource:
    def __init__(self) -> None:
        self.receiver_truth = ReceiverTruth(
            pos_ecef_m=np.zeros(3),
            vel_ecef_mps=np.zeros(3),
            clk_bias_s=0.0,
            clk_drift_sps=0.0,
        )


class _LegacyPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[list[GnssMeasurement], ReceiverTruth]] = []

    def apply(self, measurements: list[GnssMeasurement], truth: ReceiverTruth):
        self.calls.append((measurements, truth))
        return list(measurements), {"legacy": True}


class _StepAwarePipeline:
    def __init__(self) -> None:
        self.step_times: list[float] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def step(self, t_s: float) -> None:
        self.step_times.append(t_s)

    def apply(self, measurements, sv_states, *, rx_truth):
        return measurements, None


def _make_measurement(t_s: float) -> GnssMeasurement:
    return GnssMeasurement(
        sv_id="G01",
        t=t_s,
        pr_m=20_000.0,
        prr_mps=0.0,
        sigma_pr_m=1.0,
        cn0_dbhz=45.0,
        elev_deg=30.0,
        az_deg=180.0,
        flags={"valid": True},
    )


def test_apply_attacks_supports_legacy_two_argument_pipeline_apply() -> None:
    pipeline = _LegacyPipeline()
    engine = SimulationEngine(
        meas_src=_MeasSource(),
        solver=None,
        integrity_checker=None,
        attack_pipeline=pipeline,
        conops_sm=None,
    )

    measurements = [_make_measurement(1.0)]
    attacked, report = engine._apply_attacks(measurements, sv_states=[])

    assert attacked == measurements
    assert report == {"legacy": True}
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0][0] == measurements
    assert pipeline.calls[0][1] is engine.meas_src.receiver_truth


def test_apply_attacks_steps_pipeline_and_resets_when_time_goes_back() -> None:
    pipeline = _StepAwarePipeline()
    engine = SimulationEngine(
        meas_src=_MeasSource(),
        solver=None,
        integrity_checker=None,
        attack_pipeline=pipeline,
        conops_sm=None,
    )

    engine._last_t_s = 5.0
    engine._apply_attacks([_make_measurement(3.0)], sv_states=[])
    engine._apply_attacks([_make_measurement(6.0)], sv_states=[])

    assert pipeline.reset_count == 1
    assert pipeline.step_times == [3.0, 6.0]
