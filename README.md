# GNSS Twin

Minimal GNSS digital twin package with demo scripts and test coverage for Phase 1 (static GPS receiver) simulation.

## Overview

GNSS Twin simulates a simplified GPS constellation, generates synthetic pseudorange measurements, and runs a navigation
pipeline with integrity checks. The stack includes measurement modeling (iono/tropo/multipath/noise), WLS navigation with
optional EKF smoothing, RAIM-style integrity, logging/plotting utilities, and scenario runners for batch analysis.

For a detailed walkthrough of the current functionality and modules, see [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md).

## Setup

```bash
make install
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
