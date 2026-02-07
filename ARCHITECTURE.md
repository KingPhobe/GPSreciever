# GNSS Receiver Digital Twin — Phase 1 Architecture & Interfaces

## High-level Architecture (Text Diagram)

```
+---------------------------+       +----------------------+       +-------------------------+
|   Scenario / Truth Model  | ----> |  Measurement Source | ----> |     Receiver Pipeline   |
| (static LLA/ECEF, time)   |       |  (interface)         |       |  (PVT, integrity)       |
+---------------------------+       +----------------------+       +-------------------------+
                                              |                               |
                                              v                               v
                                   +---------------------+       +-------------------------+
                                   | Synthetic Generator |       | Logging & Plotting      |
                                   | (Phase 1 impl.)     |       | (time series + figures) |
                                   +---------------------+       +-------------------------+

Receiver Pipeline (Phase 1 submodules)
  - sat/        : satellite orbit + clock models
  - meas/       : measurement models (iono/tropo/noise)
  - receiver/   : WLS solver + PVT + DOP + flags
  - integrity/  : residual checks, FDE, chi-square
  - sim/        : scenario runner + demo scripts
```

## Repo Folder Structure (Target)

```
/gnss_twin/
  __init__.py
  sat/
  meas/
  receiver/
  integrity/
  sim/
  tests/
```

## Non-Negotiable Interfaces (Definitions Only)

### 1) MeasurementSource Interface

Purpose: Produce per-epoch GNSS measurements for a fixed receiver scenario (Phase 1), with a future swap for tracking-based measurements.

```python
class MeasurementSource(Protocol):
    """Interface for GNSS measurement sources.

    Implementations must be deterministic given a seed and return
    GnssMeasurement objects consistent across sources.
    """

    def reset(self) -> None:
        """Reset internal state for a fresh simulation run."""

    def epoch(self, t_gps: float) -> list[GnssMeasurement]:
        """Return measurements for the given GPS time (seconds)."""
```

### 2) Measurement Data Structure

```python
@dataclass(frozen=True)
class GnssMeasurement:
    """Single-satellite measurement at a given epoch."""
    t_gps: float            # GPS time (s)
    sv_id: str              # e.g., "G05"
    sat_pos_ecef: np.ndarray  # (3,) meters
    sat_vel_ecef: np.ndarray  # (3,) m/s
    cn0_dbhz: float
    elevation_rad: float
    azimuth_rad: float

    # Required
    pseudorange_m: float

    # Preferred
    pseudorange_rate_mps: float | None

    # Optional (Phase 2+)
    carrier_phase_cycles: float | None
```

### 3) Navigation Solver Interface

```python
class NavSolver(Protocol):
    """Interface for navigation solution algorithms."""

    def solve(self, measurements: list[GnssMeasurement]) -> NavSolution:
        """Compute a navigation solution from measurements."""
```

### 4) Navigation Solution Structure

```python
@dataclass(frozen=True)
class NavSolution:
    t_gps: float
    pos_ecef_m: np.ndarray
    vel_ecef_mps: np.ndarray | None
    pos_lla_rad_m: tuple[float, float, float]  # (lat, lon in rad, alt in m)
    clock_bias_m: float
    clock_drift_mps: float | None

    # DOP metrics
    gdop: float
    pdop: float
    hdop: float
    vdop: float

    # Residual metrics
    residual_rms_m: float
    chi_square: float

    # Flags / metadata
    fix_type: str  # "NO_FIX", "2D", "3D"
    valid: bool
    sv_used: list[str]
    sv_rejected: list[str]
```

## Phase 1 Assumptions (Explicit)

- Constellation: GPS only (L1 C/A).
- Orbit model: **Simplified circular orbit model** (deterministic, no ephemeris parsing).
- Receiver scenario: static receiver, 60 seconds, 1 Hz epochs, elevation mask 10°.
- Error models: receiver clock bias + drift (Gauss-Markov), satellite clock (simple polynomial), ionosphere (Klobuchar), troposphere (Saastamoinen), elevation-weighted noise, optional multipath (simple elevation-dependent).
- Deterministic seeds for reproducibility.

## Next Steps (Implementation Plan)

1. Implement minimal runnable skeleton (package structure + stubs, demo runner, minimal tests).
2. Add synthetic orbit model and truth ranges.
3. Add clock, iono/tropo, and noise models.
4. Add WLS solver + DOP/residuals.
5. Add integrity checks, FDE, and validity flags.
6. Logging + plotting + unit tests.
