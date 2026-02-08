# GNSS Twin Project Documentation (Current State)

## Project Purpose

GNSS Twin is a lightweight, reproducible GNSS receiver digital twin that simulates a static GPS receiver, produces synthetic measurements, and runs navigation and integrity pipelines. The current implementation focuses on Phase 1 functionality: deterministic satellite orbits, pseudorange measurement modeling, weighted least-squares (WLS) positioning, optional EKF smoothing, integrity checks, and logging/plotting outputs. The repository includes scripts to run single demos and scenario batches, along with comprehensive tests that validate models and interfaces. 【F:README.md†L1-L20】【F:ARCHITECTURE.md†L1-L70】【F:gnss_twin/runtime/engine.py†L1-L238】

## Repository Layout

```
.
├── gnss_twin/          # Core package
├── sim/                # Simulation scripts and scenarios
├── tests/              # Pytest suite
├── docs/               # Project documentation (this file)
├── ARCHITECTURE.md     # Architecture/interfaces for Phase 1
├── README.md           # Setup + quickstart
└── Makefile            # install/test helpers
```

Key entry points and documentation live in `README.md`, `ARCHITECTURE.md`, and the `sim/` scripts. 【F:README.md†L1-L34】【F:ARCHITECTURE.md†L1-L122】【F:sim/run_static_demo.py†L1-L151】

## Core Simulation Pipeline

The runtime engine orchestrates the full pipeline. At each epoch it:

1. Generates satellite states using the simplified GPS constellation.
2. Synthesizes pseudorange (and pseudorange-rate) measurements with clock, iono, tropo, multipath, and noise models.
3. Applies attack models (if enabled).
4. Filters and gates measurements.
5. Solves navigation with WLS (or optional EKF) and runs integrity checks, including RAIM statistics.
6. Emits per-epoch logs and plotting outputs.

This flow is managed by `gnss_twin.runtime.Engine`. 【F:gnss_twin/runtime/engine.py†L1-L238】

## Package Guide (`gnss_twin/`)

### Configuration and Models

- **`config.SimConfig`** captures simulation controls: seeds, duration, elevation masks, CN0 thresholds, attack settings, and EKF enablement. 【F:gnss_twin/config.py†L1-L28】
- **`models`** defines data structures and interfaces: measurements, satellite states, receiver truth, solution summaries, and interface protocols. 【F:gnss_twin/models.py†L1-L156】

### Constellation and Visibility

- **`sat.simple_gps`** provides a deterministic, circular-orbit GPS-like constellation with configurable planes and clock bias/drift. 【F:gnss_twin/sat/simple_gps.py†L1-L139】
- **`sat.visibility`** filters visible satellites by elevation mask. 【F:gnss_twin/sat/visibility.py†L1-L51】

### Measurement Modeling

- **`meas.pseudorange`** synthesizes pseudorange and range-rate measurements, adding ionosphere (Klobuchar), troposphere (Saastamoinen), multipath, noise, and receiver clock dynamics. 【F:gnss_twin/meas/pseudorange.py†L1-L179】
- Supporting models live under `meas/`, including clock dynamics, atmospheric corrections, multipath, and noise tuning. 【F:gnss_twin/meas/clock_models.py†L1-L82】【F:gnss_twin/meas/iono_klobuchar.py†L1-L92】【F:gnss_twin/meas/tropo_saastamoinen.py†L1-L102】【F:gnss_twin/meas/multipath.py†L1-L63】【F:gnss_twin/meas/noise.py†L1-L74】

### Receiver Navigation

- **`receiver.wls_pvt`** implements iterative WLS position/clock solutions and optional velocity estimation from pseudorange rates. 【F:gnss_twin/receiver/wls_pvt.py†L1-L158】
- **`receiver.ekf_nav`** provides an EKF-based navigation filter (enabled via `SimConfig.use_ekf`). 【F:gnss_twin/receiver/ekf_nav.py†L1-L260】
- **`receiver.gating`** handles prefit and postfit gating, rejecting measurements that violate CN0 and residual thresholds. 【F:gnss_twin/receiver/gating.py†L1-L117】
- **`receiver.tracking_state`** maintains per-satellite tracking state based on CN0 and continuity thresholds. 【F:gnss_twin/receiver/tracking_state.py†L1-L115】

### Integrity Monitoring

- **`integrity.flags`** performs RAIM-style checks, FDE iterations, and builds `FixFlags`/`ResidualStats` outputs. 【F:gnss_twin/integrity/flags.py†L1-L221】
- **`integrity.raim`** provides chi-square computations and RAIM statistics used by the engine and integrity checks. 【F:gnss_twin/integrity/raim.py†L1-L105】

### Attack Models

- **`attacks`** contains optional disruptions for scenario testing:
  - CN0 drop (jamming),
  - clock ramp spoofing, and
  - single-satellite pseudorange ramp spoofing. 【F:gnss_twin/attacks/__init__.py†L1-L62】【F:gnss_twin/attacks/jamming.py†L1-L74】【F:gnss_twin/attacks/spoofing.py†L1-L83】

### Logging, Plotting, and Utilities

- **`logger`** writes epoch-level CSV/NPZ summaries. 【F:gnss_twin/logger.py†L1-L123】
- **`plots`** generates run plots (position error, clock bias, residuals, DOP, satellites used, fix status). 【F:gnss_twin/plots.py†L1-L170】
- **`utils`** includes WGS84 conversions and angle calculations used throughout the pipeline. 【F:gnss_twin/utils/wgs84.py†L1-L170】【F:gnss_twin/utils/angles.py†L1-L71】

## Simulation Scripts (`sim/`)

### Static Demo

- `sim/run_static_demo.py` runs a single static receiver simulation with configurable duration, EKF toggle, and attack parameters. It produces `epoch_logs.csv`, `epoch_logs.npz`, and plot images in a timestamped output directory. 【F:sim/run_static_demo.py†L1-L151】

Example:

```bash
python sim/run_static_demo.py --duration-s 120 --attack-name spoof_clock_ramp --attack-param ramp_rate_mps=1.0
```

### Scenario Runner

- `sim/scenario_runner.py` executes one or more JSON scenarios, writes per-run summaries, and aggregates metrics into a CSV. 【F:sim/scenario_runner.py†L1-L205】
- Scenarios live in `sim/scenarios/` and override `SimConfig` keys such as seed, duration, EKF usage, and attack settings. 【F:sim/scenario_runner.py†L32-L89】

Example:

```bash
python sim/scenario_runner.py --scenarios sim/scenarios/phase1_baseline.json sim/scenarios/phase1_spoofing.json
```

### Phase 1 Exit Checklist

- `sim/phase1_exit_checklist.py` runs two acceptance scenarios (no noise/no multipath, and multipath-only) to validate residual behavior and position error thresholds. 【F:sim/phase1_exit_checklist.py†L1-L127】

## Outputs

Each run produces:

- `epoch_logs.csv` / `epoch_logs.npz` with per-epoch summaries (PVT state, DOP, residuals, fix status, NIS alarms). 【F:gnss_twin/logger.py†L1-L123】
- Plot images: `position_error.png`, `clock_bias.png`, `residual_rms.png`, `dop.png`, `satellites_used.png`, `fix_status.png`. 【F:gnss_twin/plots.py†L1-L78】
- Scenario runs also create `summary.json` and append a global `summary.csv` for aggregate metrics. 【F:sim/scenario_runner.py†L20-L205】

## Testing

The `tests/` directory provides unit and integration tests for measurement models, solver correctness, integrity checks, attacks, tracking state, plotting/logging, and scenario runs. Use `make test` or `pytest` to run the suite. 【F:Makefile†L1-L6】【F:tests/test_smoke.py†L1-L36】

## Known Scope / Phase 1 Constraints

- GPS-only constellation with simplified circular orbits.
- Static receiver at a fixed location (Stanford-like LLA by default).
- Single-frequency pseudorange/range-rate measurements.
- Deterministic seeds for repeatable outputs.

These constraints are recorded as the current Phase 1 baseline in `ARCHITECTURE.md`. 【F:ARCHITECTURE.md†L72-L122】【F:gnss_twin/runtime/engine.py†L30-L43】
