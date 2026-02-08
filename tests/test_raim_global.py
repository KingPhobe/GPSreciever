from __future__ import annotations

from gnss_twin.integrity.raim import compute_raim


def test_compute_raim_clean_passes() -> None:
    residuals_by_sv = {f"G{idx:02d}": 0.5 for idx in range(1, 7)}
    sigmas_by_sv = {sv_id: 1.0 for sv_id in residuals_by_sv}
    t_stat, dof, threshold, passed = compute_raim(residuals_by_sv, sigmas_by_sv)

    assert dof == 2
    assert threshold > 0.0
    assert t_stat < threshold
    assert passed is True


def test_compute_raim_outlier_fails() -> None:
    residuals_by_sv = {f"G{idx:02d}": 0.5 for idx in range(1, 7)}
    residuals_by_sv["G03"] = 10.0
    sigmas_by_sv = {sv_id: 1.0 for sv_id in residuals_by_sv}
    t_stat, _, threshold, passed = compute_raim(residuals_by_sv, sigmas_by_sv)

    assert t_stat > threshold
    assert passed is False
