from dataclasses import fields

import numpy as np

from gnss_twin.models import (
    Constellation,
    DopMetrics,
    EpochLog,
    FixFlags,
    GnssMeasurement,
    MeasurementSource,
    NavSolver,
    PvtSolution,
    ReceiverTruth,
    ResidualStats,
    SvState,
)


def test_dataclass_fields() -> None:
    assert [field.name for field in fields(GnssMeasurement)] == [
        "sv_id",
        "t",
        "pr_m",
        "prr_mps",
        "sigma_pr_m",
        "cn0_dbhz",
        "elev_deg",
        "az_deg",
        "flags",
    ]
    assert [field.name for field in fields(SvState)] == [
        "sv_id",
        "t",
        "pos_ecef_m",
        "vel_ecef_mps",
        "clk_bias_s",
        "clk_drift_sps",
    ]
    assert [field.name for field in fields(ReceiverTruth)] == [
        "pos_ecef_m",
        "vel_ecef_mps",
        "clk_bias_s",
        "clk_drift_sps",
    ]
    assert [field.name for field in fields(PvtSolution)] == [
        "pos_ecef",
        "vel_ecef",
        "clk_bias_s",
        "clk_drift_sps",
        "dop",
        "residuals",
        "fix_flags",
    ]
    assert [field.name for field in fields(EpochLog)] == [
        "t",
        "meas",
        "solution",
        "truth",
        "per_sv_stats",
    ]


class DummyMeasurementSource(MeasurementSource):
    def get_measurements(self, t: float) -> list[GnssMeasurement]:
        return [
            GnssMeasurement(
                sv_id="G01",
                t=t,
                pr_m=0.0,
                prr_mps=None,
                sigma_pr_m=1.0,
                cn0_dbhz=0.0,
                elev_deg=0.0,
                az_deg=0.0,
                flags={},
            )
        ]


class DummyConstellation(Constellation):
    def get_sv_states(self, t: float) -> list[SvState]:
        return [
            SvState(
                sv_id="G01",
                t=t,
                pos_ecef_m=np.zeros(3),
                vel_ecef_mps=np.zeros(3),
                clk_bias_s=0.0,
                clk_drift_sps=0.0,
            )
        ]


class DummySolver(NavSolver):
    def solve(self, meas: list[GnssMeasurement], sv_states: list[SvState]) -> PvtSolution:
        return PvtSolution(
            pos_ecef=np.zeros(3),
            vel_ecef=None,
            clk_bias_s=0.0,
            clk_drift_sps=0.0,
            dop=DopMetrics(gdop=1.0, pdop=1.0, hdop=1.0, vdop=1.0),
            residuals=ResidualStats(rms_m=0.0, mean_m=0.0, max_m=0.0, chi_square=0.0),
            fix_flags=FixFlags(fix_type="NO_FIX", valid=False, sv_used=[], sv_rejected=[]),
        )


def test_interface_compliance() -> None:
    source = DummyMeasurementSource()
    constellation = DummyConstellation()
    solver = DummySolver()

    measurements = source.get_measurements(0.0)
    sv_states = constellation.get_sv_states(0.0)
    solution = solver.solve(measurements, sv_states)

    assert isinstance(measurements[0], GnssMeasurement)
    assert isinstance(sv_states[0], SvState)
    assert isinstance(solution, PvtSolution)
