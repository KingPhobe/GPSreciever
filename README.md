# GNSS Twin

Minimal GNSS digital twin package with demo scripts, a desktop GUI, and headless scenario runners for batch analysis.

## Overview

GNSS Twin simulates a simplified GNSS constellation, generates synthetic pseudorange / pseudorange-rate measurements, and runs a navigation
pipeline with integrity checks. The stack includes measurement modeling (iono/tropo/multipath/noise), WLS navigation with optional EKF
smoothing, RAIM-style integrity, logging/plotting utilities, and scenario runners for repeatable analysis.

For a deeper module-by-module description, see `docs/PROJECT_DOCUMENTATION.md`.

## Setup

Create a virtual environment (recommended) and install:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Quickstart

Run a single static demo:

```bash
python sim/run_static_demo.py --duration-s 120
```

Run scenarios from JSON configs:

```bash
python sim/scenario_runner.py --scenarios sim/scenarios/phase1_baseline.json
```

## GUI (native desktop)

Install with GUI extras (PyQt6 on Windows-friendly Qt bindings):

```bash
pip install -e ".[gui]"
```

Run the desktop GUI:

```bash
python -m sim.desktop_gui
```

Streamlit is no longer required for the interactive GUI flow.

Demo tips:

- Click **Initialize / Reset** after changing controls to rebuild the engine with the new scenario.
- Press **Run** for continuous stepping (**Step** for single-epoch updates, **Stop** to halt).
- Use **Save plots** to export standard PNG plots and a full `run_table.csv`.

## Live demo / interactive playback

Baseline run:

```bash
python -m sim.live_runner --duration-s 60 --use-ekf --attack-name none
```

Spoof ramp example:

```bash
python -m sim.live_runner --duration-s 60 --use-ekf --attack-name spoof_pr_ramp --attack-param target_sv=G12 --attack-param start_t=10 --attack-param ramp_rate_mps=2
```

Headless JSONL export:

```bash
python -m sim.live_runner --duration-s 60 --use-ekf --attack-name spoof_pr_ramp --attack-param target_sv=G12 --attack-param start_t=10 --attack-param ramp_rate_mps=2 --out-jsonl out/live.jsonl --no-plots
```

`slope_mps` is accepted as a deprecated alias for `ramp_rate_mps`.

## Outputs

Each run writes `epoch_logs.csv`/`epoch_logs.npz` plus standard plots (position error, clock bias, residual RMS, DOP, satellites
used, fix status) into a timestamped output directory under `out/`.

## Phase 1 exit checklist

Run the two acceptance scenarios that disable noise (keeping iono+tropo on) and validate
multipath behavior:

```bash
python sim/phase1_exit_checklist.py --duration-s 120
```

Expected outcomes:

- **No noise + no multipath:** position error max and RMS should remain below 1 m.
- **Multipath only:** residual spikes should mostly coincide with low-elevation satellites
  (check the printed low-elevation spike ratio).

## Tests

```bash
make test
```
