import numpy as np

from gnss_twin.attacks import AttackPipeline, JamCn0DropAttack
from gnss_twin.models import GnssMeasurement, ReceiverTruth, SvState
from gnss_twin.runtime.engine import SimulationEngine


class _MeasSource:
    def __init__(self) -> None:
        self.receiver_truth = ReceiverTruth(
            pos_ecef_m=np.zeros(3),
            vel_ecef_mps=np.zeros(3),
            clk_bias_s=0.0,
            clk_drift_sps=0.0,
        )

    def get_batch(self, _t_s: float):
        return [], [], self.receiver_truth


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
        sigma_doppler_hz=1.0,
        flags={"valid": True},
    )


def _make_sv_state(t_s: float) -> SvState:
    return SvState(
        sv_id="G01",
        t=t_s,
        pos_ecef_m=np.array([15_600_000.0, 0.0, 21_400_000.0]),
        vel_ecef_mps=np.array([0.0, 2_500.0, 1_100.0]),
        clk_bias_s=0.0,
        clk_drift_sps=0.0,
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
    attacked, report = engine._apply_attacks(
        measurements,
        sv_states=[],
        rx_truth_snapshot=engine.meas_src.receiver_truth,
    )

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
    engine._apply_attacks(
        [_make_measurement(3.0)],
        sv_states=[],
        rx_truth_snapshot=engine.meas_src.receiver_truth,
    )
    engine._apply_attacks(
        [_make_measurement(6.0)],
        sv_states=[],
        rx_truth_snapshot=engine.meas_src.receiver_truth,
    )

    assert pipeline.reset_count == 1
    assert pipeline.step_times == [3.0, 6.0]


def test_step_uses_per_epoch_receiver_truth_snapshot_for_attacks() -> None:
    class _TruthRecordingPipeline:
        def __init__(self) -> None:
            self.truth_clk_bias_s: list[float] = []
            self.truth_clk_drift_sps: list[float] = []

        def apply(self, measurements, sv_states, *, rx_truth):
            self.truth_clk_bias_s.append(rx_truth.clk_bias_s)
            self.truth_clk_drift_sps.append(rx_truth.clk_drift_sps)
            return measurements, None

    class _EpochTruthSource(_MeasSource):
        def __init__(self) -> None:
            super().__init__()
            # Stored truth must be ignored by the attack path.
            self.receiver_truth = ReceiverTruth(
                pos_ecef_m=np.zeros(3),
                vel_ecef_mps=np.zeros(3),
                clk_bias_s=111.0,
                clk_drift_sps=222.0,
            )

        def get_batch(self, t_s: float):
            meas = [_make_measurement(t_s)]
            rx_truth = ReceiverTruth(
                pos_ecef_m=np.zeros(3),
                vel_ecef_mps=np.zeros(3),
                clk_bias_s=float(t_s),
                clk_drift_sps=float(t_s) * 0.1,
            )
            return meas, [], rx_truth

    pipeline = _TruthRecordingPipeline()
    engine = SimulationEngine(
        meas_src=_EpochTruthSource(),
        solver=None,
        integrity_checker=None,
        attack_pipeline=pipeline,
        conops_sm=None,
    )

    engine.step(1.0)
    engine.step(2.0)

    assert pipeline.truth_clk_bias_s == [1.0, 2.0]
    assert pipeline.truth_clk_drift_sps == [0.1, 0.2]


def test_step_jamming_attack_inflates_position_and_doppler_uncertainty() -> None:
    class _EpochSource(_MeasSource):
        def get_batch(self, t_s: float):
            measurements = [_make_measurement(t_s)]
            return measurements, [_make_sv_state(t_s)], self.receiver_truth

    attack_pipeline = AttackPipeline(
        attacks=[
            JamCn0DropAttack(
                start_t=0.0,
                cn0_drop_db=8.0,
                sigma_pr_scale=4.0,
                sigma_prr_scale=3.0,
            )
        ]
    )
    engine = SimulationEngine(
        meas_src=_EpochSource(),
        solver=None,
        integrity_checker=None,
        attack_pipeline=attack_pipeline,
        conops_sm=None,
    )

    output = engine.step(1.0)
    attacked = output["meas_attacked"][0]

    assert attacked.sigma_pr_m == 4.0
    assert attacked.sigma_doppler_hz == 3.0
