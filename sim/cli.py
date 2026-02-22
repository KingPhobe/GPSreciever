"""Unified CLI entrypoint.

Project policy: expose only two run modes to users:
  1) GUI run (interactive)
  2) Scenario run (headless, produces CSV + plots)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _cmd_gui(_: argparse.Namespace) -> None:
    from sim.desktop_gui import main as gui_main

    gui_main()


def _cmd_scenario(args: argparse.Namespace) -> None:
    from sim.scenario_runner import run_scenarios

    scenarios = [Path(p) for p in (args.scenario or [])]
    if not scenarios:
        raise SystemExit("No scenarios provided. Use --scenario path.json (repeatable).")

    run_scenarios(
        scenarios,
        run_root=Path(args.run_root),
        save_figs=not args.no_plots,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gnss-twin", description="GNSS Twin unified runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gui = sub.add_parser("gui", help="Run the interactive desktop GUI")
    gui.set_defaults(func=_cmd_gui)

    scen = sub.add_parser("scenario", help="Run one or more JSON scenarios (headless)")
    scen.add_argument("--scenario", action="append", help="Path to a scenario JSON file (repeatable)")
    scen.add_argument("--run-root", type=str, default="runs", help="Root folder for scenario outputs")
    scen.add_argument("--no-plots", action="store_true", help="Skip saving plot PNGs")
    scen.set_defaults(func=_cmd_scenario)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
