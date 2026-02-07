# GNSS Twin

Minimal GNSS digital twin package with demo scripts.

## Setup

```bash
make install
```

## Tests

```bash
make test
```

## Phase 1 exit checklist

Run the two acceptance scenarios that disable noise (keeping iono+tropo on) and validate
the multipath behavior:

```bash
python sim/phase1_exit_checklist.py --duration-s 120
```

Expected outcomes:

- **No noise + no multipath:** position error max and RMS should remain below 1 m.  
- **Multipath only:** residual spikes should mostly coincide with low-elevation satellites
  (check the printed low-elevation spike ratio).  
