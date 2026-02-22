# GNSS Twin

A lightweight GNSS receiver *digital twin* for repeatable demos and experiments:
synthetic GPS-like constellation + measurement models + navigation (WLS / EKF) +
integrity monitoring (RAIM-style) + optional jamming/spoofing attacks.

This repo is intended to be **easy to run**, **easy to extend**, and **deterministic** (seeded).

---

## What you can do with it

- Run a **static receiver** simulation (truth + estimated PVT) and export logs/plots.
- Batch-run **JSON scenarios** (baseline, jamming, spoofing) and get summary metrics.
- Use the **desktop GUI** (PyQt6) for live stepping + plot saving.
- Run the **live runner** (Matplotlib interactive) for demo playback + optional JSONL export.

---

## Repo layout

- `gnss_twin/` – core library (constellation, measurement models, receiver solver, integrity, attacks, logging)
- `sim/` – runnable entrypoints (CLI, demo scripts, scenario runner, GUI, live runner)
- `sim/scenarios/` – ready-to-run JSON scenarios
- `tests/` – pytest suite (unit + integration/regression)

Docs:
- `ARCHITECTURE.md` – system-level architecture and interfaces
- `docs/PROJECT_DOCUMENTATION.md` – module-by-module walkthrough

---

## Requirements

- Python **3.9+**
- OS: Windows / Linux / macOS

---

## Install

Recommended (editable install in a virtual environment):

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[dev]"
```

GUI install (adds PyQt6):

```bash
pip install -e ".[dev,gui]"
```

---

## Quickstart (recommended path)

### 1) Run a baseline scenario (headless, saves CSV + plots)

```bash
gnss-twin scenario --scenario sim/scenarios/baseline.json
```

Outputs go under `runs/` by default (timestamped folder per scenario).

### 2) Run a single demo run (direct script)

```bash
python -m sim.run_static_demo --duration-s 60 --use-ekf
```

Outputs go under `out/<timestamp>/`.

---

## Scenarios (JSON)

Run multiple scenarios in one call:

```bash
gnss-twin scenario --scenario sim/scenarios/baseline.json --scenario sim/scenarios/jam_cn0_drop.json
```

The scenario runner writes:
- per-scenario `summary.json` inside each run directory
- aggregate `runs/summary.csv` (appended per run)

Scenario files live in `sim/scenarios/*.json` and must include:
`name`, `duration_s`, `rng_seed`, `use_ekf`, `attack_name`, `attack_params`.

You can also override any `SimConfig` field directly in the JSON
(e.g., `cn0_min_dbhz`, `sigma_pr_max_m`, `elev_mask_deg`, etc.).

---

## Attacks

Available attack names (see `gnss_twin/attacks/`):
- `none`
- `jam_cn0_drop` – CN0 reduction + measurement noise scaling
- `spoof_clock_ramp` – coherent pseudorange ramp across all SVs
- `spoof_pos_offset` – coherent position offset applied in measurement domain
- `spoof_pr_ramp` – ramp on a single satellite PR (can auto-select a visible SV)

Examples (script demo):

```bash
python -m sim.run_static_demo --duration-s 60 --use-ekf --attack-name spoof_clock_ramp --attack-param start_t=10 --attack-param ramp_rate_mps=50
```

```bash
python -m sim.run_static_demo --duration-s 60 --use-ekf --attack-name jam_cn0_drop --attack-param start_t=15 --attack-param cn0_drop_db=18
```

Note: `spoof_pr_ramp` supports auto-selection in JSON scenarios via
`auto_select_visible_sv=true` + `strict_target_sv=false`. The *demo CLI* currently
requires `--attack-param target_sv=G##` when using `--attack-name spoof_pr_ramp`.

---

## GUI (desktop)

Install GUI extras:

```bash
pip install -e ".[dev,gui]"
```

Run:

```bash
gnss-twin gui
# or:
python -m sim.desktop_gui
```

Workflow:
1. Adjust controls (scenario, attack params, thresholds, etc.)
2. Click **Initialize / Reset**
3. Use **Run** / **Step** / **Stop**
4. Click **Save outputs** / **Save plots** to export CSV + PNGs

---

## Live runner (interactive Matplotlib)

```bash
python -m sim.live_runner --duration-s 60 --use-ekf --attack-name none
```

Spoof ramp example:

```bash
python -m sim.live_runner --duration-s 60 --use-ekf --attack-name spoof_pr_ramp --attack-param target_sv=G12 --attack-param start_t=10 --attack-param ramp_rate_mps=2
```

Headless JSONL export (no plots):

```bash
python -m sim.live_runner --duration-s 60 --use-ekf --attack-name spoof_pr_ramp --attack-param target_sv=G12 --attack-param start_t=10 --attack-param ramp_rate_mps=2 --out-jsonl out/live.jsonl --no-plots
```

---

## Outputs (what to expect)

A typical run directory contains:
- `epoch_logs.csv` + `epoch_logs.npz` – per-epoch state/integrity/telemetry
- `run_table.csv` – flattened “analysis table” for quick plotting in Excel/Pandas
- `meas_log.csv` – per-SV raw vs attacked measurement audit (when available)
- `nmea_output.csv` + `nmea_output.nmea` – NMEA sentences (GGA/RMC)
- `mode5_auth.csv` – mode-5 auth/holdover timeline telemetry
- `run_manifest.json` + `run_metadata.csv` – config + environment metadata
- Plot PNGs (if enabled): `position_error.png`, `clock_bias.png`, `residual_rms.png`,
  `dop.png`, `satellites_used.png`, `fix_status.png`, `attack_telemetry.png`,
  plus ConOps plots like `conops_status_timeline.png`.

---

## Tests and verification

Run the unit test suite:

```bash
make test
# or:
pytest -q
```

Run the verification suite (quick):

```bash
make verify
```

Monte Carlo verification (slower):

```bash
make verify-mc
```

---

## Notes

- This is a simplified GPS-like constellation and receiver model intended for demos and rapid iteration.
- The simulation is deterministic given `rng_seed` (repeatable plots/logs).
