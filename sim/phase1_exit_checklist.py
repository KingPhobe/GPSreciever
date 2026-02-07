"""Phase 1 exit checklist runs for GNSS twin."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from gnss_twin.integrity.flags import IntegrityConfig, SvTracker, integrity_pvt
from gnss_twin.meas.pseudorange import SyntheticMeasurementSource
from gnss_twin.models import ReceiverTruth
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.wgs84 import lla_to_ecef


@dataclass
class ScenarioConfig:
    name: str
    enable_multipath: bool
    pr_sigma_base_m: float
    prr_sigma_mps: float


def _run_scenario(
    config: ScenarioConfig,
    *,
    duration_s: float,
    seed: int,
    spike_threshold_m: float,
    low_elev_deg: float,
) -> None:
    receiver_lla = (37.4275, -122.1697, 30.0)
    receiver_truth_ecef = lla_to_ecef(*receiver_lla)
    receiver_clock = 4.2e-6
    receiver_truth = ReceiverTruth(
        pos_ecef_m=receiver_truth_ecef,
        vel_ecef_mps=np.zeros(3),
        clk_bias_s=receiver_clock,
        clk_drift_sps=0.0,
    )

    rng = np.random.default_rng(seed)
    constellation = SimpleGpsConstellation(SimpleGpsConfig(seed=seed))
    measurement_source = SyntheticMeasurementSource(
        constellation=constellation,
        receiver_truth=receiver_truth,
        pr_sigma_base_m=config.pr_sigma_base_m,
        prr_sigma_mps=config.prr_sigma_mps,
        enable_multipath=config.enable_multipath,
        rng=rng,
    )
    integrity_cfg = IntegrityConfig()
    tracker = SvTracker(integrity_cfg)

    times = np.arange(0.0, duration_s, 1.0)
    last_pos = receiver_truth_ecef + 100.0
    last_clk = receiver_clock
    pos_errors: list[float] = []
    spike_epochs = 0
    spike_low_elev = 0
    valid_epochs = 0

    for t in times:
        sv_states = constellation.get_sv_states(float(t))
        meas = measurement_source.get_measurements(float(t))
        solution, per_sv_stats = integrity_pvt(
            meas,
            sv_states,
            initial_pos_ecef_m=last_pos,
            initial_clk_bias_s=last_clk,
            config=integrity_cfg,
            tracker=tracker,
        )
        if solution.fix_flags.fix_type != "NO FIX" and np.isfinite(solution.pos_ecef).all():
            valid_epochs += 1
            pos_errors.append(float(np.linalg.norm(solution.pos_ecef - receiver_truth_ecef)))
            last_pos = solution.pos_ecef
            last_clk = solution.clk_bias_s

        residuals = [
            abs(stats["residual_m"])
            for stats in per_sv_stats.values()
            if np.isfinite(stats["residual_m"])
        ]
        if residuals:
            max_resid = max(residuals)
            min_elev = min(m.elev_deg for m in meas) if meas else float("inf")
            if max_resid >= spike_threshold_m:
                spike_epochs += 1
                if min_elev <= low_elev_deg:
                    spike_low_elev += 1

    pos_errors_np = np.array(pos_errors, dtype=float)
    max_error = float(np.nanmax(pos_errors_np)) if pos_errors_np.size else float("nan")
    rms_error = float(np.sqrt(np.nanmean(np.square(pos_errors_np)))) if pos_errors_np.size else float("nan")

    print(f"\nScenario: {config.name}")
    print(f"  Valid epochs: {valid_epochs}/{len(times)}")
    print(f"  Position error max (m): {max_error:.3f}")
    print(f"  Position error RMS (m): {rms_error:.3f}")
    if config.enable_multipath:
        spike_ratio = (spike_low_elev / spike_epochs) if spike_epochs else 0.0
        print(f"  Spike epochs: {spike_epochs} (threshold {spike_threshold_m:.2f} m)")
        print(
            "  Low-elevation spike ratio: "
            f"{spike_ratio:.2%} (<= {low_elev_deg:.1f} deg)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 exit checklist scenarios.")
    parser.add_argument("--duration-s", type=float, default=60.0, help="Duration per scenario in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic runs.")
    parser.add_argument(
        "--spike-threshold-m",
        type=float,
        default=2.0,
        help="Residual magnitude used to flag multipath spikes.",
    )
    parser.add_argument(
        "--low-elev-deg",
        type=float,
        default=20.0,
        help="Elevation threshold for low-elevation correlation checks.",
    )
    args = parser.parse_args()

    scenarios = [
        ScenarioConfig(
            name="No noise + no multipath (iono+tropo on)",
            enable_multipath=False,
            pr_sigma_base_m=0.0,
            prr_sigma_mps=0.0,
        ),
        ScenarioConfig(
            name="Multipath only (noise off)",
            enable_multipath=True,
            pr_sigma_base_m=0.0,
            prr_sigma_mps=0.0,
        ),
    ]

    for scenario in scenarios:
        _run_scenario(
            scenario,
            duration_s=args.duration_s,
            seed=args.seed,
            spike_threshold_m=args.spike_threshold_m,
            low_elev_deg=args.low_elev_deg,
        )


if __name__ == "__main__":
    main()
