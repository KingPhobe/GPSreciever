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
        "pr_model_corr_m",
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
        "t_s",
        "fix_valid",
        "fix_type",
        "sats_used",
        "pdop",
        "hdop",
        "vdop",
        "residual_rms_m",
        "pos_ecef",
        "vel_ecef",
        "clk_bias_s",
        "clk_drift_sps",
        "nis",
        "nis_alarm",
        "attack_name",
        "attack_active",
        "attack_pr_bias_mean_m",
        "attack_prr_bias_mean_mps",
        "innov_dim",
        "conops_status",
        "conops_mode5",
        "conops_reason_codes",
        "integrity_p_value",
        "integrity_residual_rms",
        "integrity_num_sats_used",
        "integrity_excluded_sv_ids_count",
        "per_sv_stats",
    ]
    assert [field.name for field in fields(FixFlags)] == [
        "fix_type",
        "valid",
        "sv_used",
        "sv_rejected",
        "sv_count",
        "sv_in_view",
        "mask_ok",
        "pdop",
        "gdop",
        "chi_square",
        "chi_square_threshold",
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
            fix_flags=FixFlags(
                fix_type="NO FIX",
                valid=False,
                sv_used=[],
                sv_rejected=[],
                sv_count=0,
                sv_in_view=0,
                mask_ok=False,
                pdop=float("nan"),
                gdop=float("nan"),
                chi_square=float("nan"),
                chi_square_threshold=float("inf"),
            ),
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
