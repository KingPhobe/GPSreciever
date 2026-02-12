"""Streamlit GUI for live GNSS simulation using the in-process SimulationEngine."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
import streamlit as st

from gnss_twin.config import SimConfig
from sim.run_static_demo import build_engine


def getv(obj: Any, *path: str, default: Any = None) -> Any:
    """Safely fetch nested values from dict-like or object-like structures."""
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
        if cur is default:
            return default
    return cur


def _init_state() -> None:
    defaults = {
        "engine": None,
        "cfg": None,
        "t_s_current": 0.0,
        "is_playing": False,
        "history": {k: [] for k in [
            "t_s",
            "residual_rms_m",
            "pdop",
            "clk_bias_s",
            "nis",
            "nis_alarm",
            "sats_used",
        ]},
        "last_status": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _to_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _record_step(step_out: dict[str, Any], t_s: float) -> None:
    sol = step_out.get("sol")
    integrity = step_out.get("integrity")
    conops = step_out.get("conops")
    attack_report = step_out.get("attack_report")

    applied_count = _to_float(getv(attack_report, "applied_count", default=0.0))
    pr_bias_sum_m = _to_float(getv(attack_report, "pr_bias_sum_m", default=0.0))
    attack_pr_bias_mean_m = pr_bias_sum_m / applied_count if applied_count > 0 else 0.0

    status = {
        "t_s_current": float(t_s),
        "fix_type": getv(sol, "fix_flags", "fix_type"),
        "fix_valid": getv(sol, "fix_flags", "valid"),
        "sats_used": getv(sol, "fix_flags", "sv_count"),
        "pdop": getv(sol, "dop", "pdop"),
        "residual_rms_m": getv(sol, "residuals", "rms_m"),
        "clk_bias_s": getv(sol, "clk_bias_s"),
        "clk_drift_sps": getv(sol, "clk_drift_sps"),
        "nis": getv(integrity, "chi2"),
        "nis_alarm": bool(getv(integrity, "is_suspect", default=False) or getv(integrity, "is_invalid", default=False)),
        "conops_status": getv(conops, "status", "value", default=getv(conops, "status")),
        "conops_mode5": getv(conops, "mode5", "value", default=getv(conops, "mode5")),
        "integrity_p_value": getv(integrity, "p_value"),
        "integrity_excluded_sv_ids_count": len(getv(integrity, "excluded_sv_ids", default=[]) or []),
        "attack_name": getv(attack_report, "attack_name", default=(st.session_state.cfg.attack_name if st.session_state.cfg else "none")),
        "attack_active": applied_count > 0,
        "attack_pr_bias_mean_m": attack_pr_bias_mean_m,
    }

    hist = st.session_state.history
    hist["t_s"].append(float(t_s))
    hist["residual_rms_m"].append(_to_float(status["residual_rms_m"]))
    hist["pdop"].append(_to_float(status["pdop"]))
    hist["clk_bias_s"].append(_to_float(status["clk_bias_s"]))
    hist["nis"].append(_to_float(status["nis"]))
    hist["nis_alarm"].append(1.0 if status["nis_alarm"] else 0.0)
    hist["sats_used"].append(_to_float(status["sats_used"]))

    st.session_state.last_status = status


def _step_once(duration_s: float, dt_s: float) -> None:
    if st.session_state.engine is None:
        return
    t_s = float(st.session_state.t_s_current)
    if t_s >= duration_s:
        st.session_state.is_playing = False
        return
    step_out = st.session_state.engine.step(t_s)
    _record_step(step_out, t_s)
    st.session_state.t_s_current = t_s + dt_s
    if st.session_state.t_s_current >= duration_s:
        st.session_state.is_playing = False


st.set_page_config(page_title="GNSS Twin Live GUI", layout="wide")
st.title("GNSS Twin Live GUI (engine-direct)")
_init_state()

with st.sidebar:
    st.header("Scenario")
    duration_s = st.slider("Duration (seconds)", min_value=5.0, max_value=600.0, value=60.0, step=1.0)
    default_dt = float(getattr(st.session_state.cfg, "dt", 1.0) if st.session_state.cfg else 1.0)
    dt_s = st.number_input("dt (seconds)", min_value=0.001, max_value=10.0, value=default_dt, step=0.1, format="%.3f")
    speed = st.slider("Speed (sim-time / wall-time)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    rng_seed = int(st.number_input("RNG seed", min_value=0, value=42, step=1))
    use_ekf = st.checkbox("use_ekf", value=True)

    attack_preset = st.selectbox("Attack preset", ["none", "spoof_pr_ramp"], index=0)
    attack_params: dict[str, float | str] = {}
    if attack_preset == "spoof_pr_ramp":
        attack_params["target_sv"] = st.text_input("target_sv", value="G12")
        attack_params["start_t"] = float(st.number_input("start_t", value=10.0, step=1.0))
        attack_params["slope_mps"] = float(st.number_input("slope_mps", value=2.0, step=0.1))

    initialize = st.button("Initialize / Reset", use_container_width=True)
    can_step = st.session_state.engine is not None
    col_a, col_b = st.columns(2)
    with col_a:
        do_step = st.button("Step", use_container_width=True, disabled=not can_step)
        do_play = st.button("Play", use_container_width=True, disabled=not can_step)
    with col_b:
        do_pause = st.button("Pause", use_container_width=True, disabled=not can_step)

if initialize:
    cfg = SimConfig(
        duration=float(duration_s),
        dt=float(dt_s),
        rng_seed=int(rng_seed),
        use_ekf=bool(use_ekf),
        attack_name=attack_preset,
        attack_params=attack_params,
    )
    st.session_state.cfg = cfg
    st.session_state.engine = build_engine(cfg)
    st.session_state.t_s_current = 0.0
    st.session_state.is_playing = False
    st.session_state.history = {k: [] for k in st.session_state.history.keys()}
    st.session_state.last_status = {}

if do_step:
    _step_once(duration_s=float(duration_s), dt_s=float(dt_s))
if do_play:
    st.session_state.is_playing = True
if do_pause:
    st.session_state.is_playing = False

if st.session_state.engine is None:
    st.info("Engine is not initialized. Use 'Initialize / Reset' in the sidebar.")

status = st.session_state.last_status
metric_cols = st.columns(8)
metric_cols[0].metric("t_s_current", f"{st.session_state.t_s_current:.2f}")
metric_cols[1].metric("fix_type", str(status.get("fix_type", "-")))
metric_cols[2].metric("sats_used", str(status.get("sats_used", "-")))
metric_cols[3].metric("pdop", f"{_to_float(status.get('pdop')):.3f}" if status else "-")
metric_cols[4].metric("residual_rms_m", f"{_to_float(status.get('residual_rms_m')):.3f}" if status else "-")
metric_cols[5].metric("nis_alarm", "YES" if status.get("nis_alarm") else "NO")
metric_cols[6].metric("conops_status", str(status.get("conops_status", "-")))
metric_cols[7].metric("attack_active", "YES" if status.get("attack_active") else "NO")

hist = st.session_state.history
if hist["t_s"]:
    df = pd.DataFrame(hist)
    st.subheader("Residual RMS (m)")
    st.line_chart(df.set_index("t_s")["residual_rms_m"])

    st.subheader("PDOP")
    st.line_chart(df.set_index("t_s")["pdop"])

    st.subheader("Clock Bias (s)")
    st.line_chart(df.set_index("t_s")["clk_bias_s"])

    st.subheader("NIS / Alarm")
    nis_df = df.set_index("t_s")[["nis", "nis_alarm"]]
    st.line_chart(nis_df)

if status:
    st.subheader("Latest Status")
    table_payload = asdict(status) if is_dataclass(status) else status
    st.table(pd.DataFrame([table_payload]))

if st.session_state.is_playing and st.session_state.engine is not None:
    batch_steps = max(1, min(5, int(0.25 / max(float(dt_s), 1e-9))))
    for _ in range(batch_steps):
        if not st.session_state.is_playing:
            break
        _step_once(duration_s=float(duration_s), dt_s=float(dt_s))
        if st.session_state.t_s_current >= float(duration_s):
            st.session_state.is_playing = False
            break
    wall_dt_target = float(dt_s) / max(float(speed), 1e-9)
    time.sleep(min(max(wall_dt_target, 0.0), 0.25))
    if st.session_state.is_playing:
        st.rerun()
