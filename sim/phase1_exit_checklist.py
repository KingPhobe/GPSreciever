"""Phase 1 exit checklist runs for GNSS twin."""

from __future__ import annotations

import argparse

from sim.validation.phase1 import default_phase1_scenarios, run_phase1_scenario


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

    for scenario in default_phase1_scenarios():
        metrics = run_phase1_scenario(
            scenario,
            duration_s=args.duration_s,
            seed=args.seed,
            spike_threshold_m=args.spike_threshold_m,
            low_elev_deg=args.low_elev_deg,
        )
        print(f"\nScenario: {metrics.scenario_name}")
        print(f"  Valid epochs: {metrics.valid_epochs}/{metrics.total_epochs}")
        print(f"  Position error max (m): {metrics.max_error_m:.3f}")
        print(f"  Position error RMS (m): {metrics.rms_error_m:.3f}")
        if scenario.enable_multipath:
            print(f"  Spike epochs: {metrics.spike_epochs} (threshold {args.spike_threshold_m:.2f} m)")
            print(
                "  Low-elevation spike ratio: "
                f"{metrics.low_elev_spike_ratio:.2%} (<= {args.low_elev_deg:.1f} deg)"
            )


if __name__ == "__main__":
    main()
