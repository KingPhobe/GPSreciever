from __future__ import annotations

from types import SimpleNamespace

from sim.live_runner import epoch_to_dict


def test_epoch_to_dict_exports_nis_from_epoch_value() -> None:
    step = {
        "sol": SimpleNamespace(
            fix_flags=SimpleNamespace(valid=True, fix_type="3D", sv_count=7),
            dop=SimpleNamespace(pdop=1.5),
            residuals=SimpleNamespace(rms_m=2.0),
            clk_bias_s=1e-6,
            clk_drift_sps=0.0,
        ),
        "integrity": SimpleNamespace(
            is_suspect=False,
            is_invalid=False,
            p_value=0.9,
            excluded_sv_ids=[],
        ),
        "conops": None,
        "attack_report": None,
    }

    record = epoch_to_dict(12.0, step, attack_name="none", epoch_nis=3.25)

    assert record["nis"] == 3.25


def test_epoch_to_dict_omits_nis_when_unavailable() -> None:
    step = {
        "sol": SimpleNamespace(
            fix_flags=SimpleNamespace(valid=True, fix_type="3D", sv_count=7),
            dop=SimpleNamespace(pdop=1.5),
            residuals=SimpleNamespace(rms_m=2.0),
            clk_bias_s=1e-6,
            clk_drift_sps=0.0,
        ),
        "integrity": SimpleNamespace(
            is_suspect=False,
            is_invalid=False,
            p_value=0.9,
            excluded_sv_ids=[],
        ),
        "conops": None,
        "attack_report": None,
    }

    record = epoch_to_dict(12.0, step, attack_name="none", epoch_nis=None)

    assert "nis" not in record
