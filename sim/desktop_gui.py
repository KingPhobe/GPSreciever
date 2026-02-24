"""Native desktop GUI for live GNSS simulation."""

from __future__ import annotations

import sys
import time
import csv
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from gnss_twin.config import SimConfig
from gnss_twin.models import EpochLog, ReceiverTruth
from gnss_twin.nmea.neo_m8n_output import NmeaEmit, NeoM8nNmeaOutput
from gnss_twin.plots import epochs_to_frame, plot_update
from gnss_twin.sat.simple_gps import SimpleGpsConfig, SimpleGpsConstellation
from gnss_twin.utils.angles import elev_az_from_rx_sv
from gnss_twin.utils.wgs84 import ecef_to_lla
from sim.run_table import add_nmea_metadata_columns
from gnss_twin.runtime.factory import build_engine_with_truth, build_epoch_log


RESID_RMS_OK_M = 10.0
PDOP_OK = 6.0
DIAG_UPDATE_PERIOD_S = 0.3
NMEA_PREVIEW_UPDATE_PERIOD_S = 0.2
NMEA_PREVIEW_LINES = 10
LIVE_FRAME_REBUILD_EVERY_STEPS = 5
RUN_TABLE_INTEGRITY_COLUMNS: dict[str, object] = {
    "integrity_p_value": float("nan"),
    "integrity_num_sats_used": float("nan"),
    "integrity_excluded_sv_ids_count": float("nan"),
    "conops_reason_codes": "",
    "nis": float("nan"),
    "nis_alarm": False,
}


def _date_folder_str() -> str:
    return datetime.now().strftime("%m%d%Y")


def _time_str() -> str:
    return datetime.now().strftime("%H%M%S")


def _sanitize_run_name(name: str) -> str:
    stripped = name.strip().replace(" ", "_")
    safe = re.sub(r"[^A-Za-z0-9._-]", "", stripped)
    if not safe:
        return f"gui_run_{_time_str()}"
    return safe


def _ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for suffix in range(1, 100):
        candidate = path.parent / f"{path.name}_{suffix:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique directory for {path}")


def _visible_svs_at_time(
    t_s: float, receiver_truth: ReceiverTruth, elev_mask_deg: float, seed: int
) -> list[str]:
    const = SimpleGpsConstellation(SimpleGpsConfig(seed=int(seed)))
    svs: list[str] = []
    for st in const.get_sv_states(float(t_s)):
        elev_deg, _ = elev_az_from_rx_sv(receiver_truth.pos_ecef_m, st.pos_ecef_m)
        if elev_deg >= float(elev_mask_deg):
            svs.append(st.sv_id)
    return sorted(svs)


def _resolve_target_sv(chosen: str, visible: list[str], auto_select: bool) -> tuple[str, bool, str]:
    if chosen in visible:
        return chosen, True, ""
    if auto_select and visible:
        auto_choice = visible[0]
        msg = (
            f"target_sv_not_visible_at_start_t; requested={chosen}; "
            f"auto_selected={auto_choice}; visible={visible}"
        )
        return auto_choice, True, msg
    msg = f"target_sv_not_visible_at_start_t; requested={chosen}; visible={visible}"
    return chosen, False, msg


class DiagnosticsWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GNSS Diagnostics")
        self.resize(1000, 800)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._build_ui()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)

        self.ax_pos_err = self.figure.add_subplot(321)
        self.ax_pdop = self.figure.add_subplot(322)
        self.ax_residual = self.figure.add_subplot(323)
        self.ax_clock = self.figure.add_subplot(324)
        self.ax_sats = self.figure.add_subplot(325)
        self.ax_fix = self.figure.add_subplot(326)

        (self.line_pos_err,) = self.ax_pos_err.plot([], [], color="tab:purple")
        self.ax_pos_err.set_title("Position Error")
        self.ax_pos_err.set_ylabel("Error (m)")
        self.ax_pos_err.grid(True, alpha=0.3)

        (self.line_pdop,) = self.ax_pdop.plot([], [], color="tab:blue")
        self.ax_pdop.set_title("PDOP")
        self.ax_pdop.grid(True, alpha=0.3)

        (self.line_residual,) = self.ax_residual.plot([], [], color="tab:green")
        self.ax_residual.set_title("Residual RMS")
        self.ax_residual.set_ylabel("m")
        self.ax_residual.grid(True, alpha=0.3)

        (self.line_clock,) = self.ax_clock.plot([], [], color="tab:orange")
        self.ax_clock.set_title("Clock Bias")
        self.ax_clock.set_ylabel("s")
        self.ax_clock.grid(True, alpha=0.3)

        (self.line_sats,) = self.ax_sats.plot([], [], color="tab:cyan")
        self.ax_sats.set_title("Satellites Used")
        self.ax_sats.set_ylabel("count")
        self.ax_sats.set_xlabel("t_s")
        self.ax_sats.grid(True, alpha=0.3)

        (self.line_fix,) = self.ax_fix.step([], [], where="post", color="tab:red")
        self.ax_fix.set_title("Fix Valid")
        self.ax_fix.set_ylim(-0.1, 1.1)
        self.ax_fix.set_yticks([0, 1])
        self.ax_fix.set_xlabel("t_s")
        self.ax_fix.grid(True, alpha=0.3)

        self.figure.tight_layout()

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.canvas)
        tabs.addTab(page, "Core Diagnostics")

    def update_plots(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            for line in [
                self.line_pos_err,
                self.line_pdop,
                self.line_residual,
                self.line_clock,
                self.line_sats,
                self.line_fix,
            ]:
                line.set_data([], [])
            self.canvas.draw_idle()
            return

        t_vals = frame["t_s"].to_numpy(dtype=float)
        self.line_pos_err.set_data(t_vals, frame["pos_error_m"].to_numpy(dtype=float))
        self.line_pdop.set_data(t_vals, frame["pdop"].to_numpy(dtype=float))
        self.line_residual.set_data(t_vals, frame["residual_rms_m"].to_numpy(dtype=float))
        self.line_clock.set_data(t_vals, frame["clk_bias_s"].to_numpy(dtype=float))
        self.line_sats.set_data(t_vals, frame["sats_used"].to_numpy(dtype=float))
        self.line_fix.set_data(t_vals, frame["fix_valid"].to_numpy(dtype=float))

        for axis in [
            self.ax_pos_err,
            self.ax_pdop,
            self.ax_residual,
            self.ax_clock,
            self.ax_sats,
            self.ax_fix,
        ]:
            axis.relim()
            axis.autoscale_view(scaley=axis is not self.ax_fix)
        self.ax_fix.set_ylim(-0.1, 1.1)

        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GNSS Twin Desktop GUI")
        self.resize(1300, 800)

        self.engine = None
        self.cfg: SimConfig | None = None
        self.receiver_truth_state: ReceiverTruth | None = None
        self.t_s_current = 0.0
        self.is_running = False
        self.epochs: list[EpochLog] = []
        self.frame = pd.DataFrame()
        self._live_frame_dirty = False
        self._steps_since_live_frame_rebuild = 0
        self.diagnostics_window: DiagnosticsWindow | None = None
        self.last_diag_update_walltime = 0.0
        self.current_run_name = ""
        self.attack_config_ok = True
        self.attack_config_msg = ""
        self.attack_never_applied_warned = False
        self.nmea_enabled = True
        self.nmea = NeoM8nNmeaOutput(rate_hz=1.0, talker="GN")
        self.nmea_t0_utc = datetime.now(timezone.utc)
        self.nmea_buffer: list[NmeaEmit] = []
        self.nmea_recent: deque[NmeaEmit] = deque(maxlen=200)
        self.last_nmea_preview_update_walltime = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_once)

        self._build_ui()
        self._update_attack_controls()
        self._refresh_visible_svs()
        self._update_status_labels()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_controls_box())
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._build_status_box())
        right_layout.addWidget(self._build_plot_box(), stretch=1)

        layout.addWidget(left, stretch=0)
        layout.addWidget(right, stretch=1)
        self.setCentralWidget(root)

    def _build_controls_box(self) -> QGroupBox:
        group = QGroupBox("Scenario & Controls")
        form = QFormLayout(group)

        self.duration_s_input = QDoubleSpinBox()
        self.duration_s_input.setRange(1.0, 36000.0)
        self.duration_s_input.setValue(60.0)

        self.dt_s_input = QDoubleSpinBox()
        self.dt_s_input.setDecimals(3)
        self.dt_s_input.setRange(0.001, 10.0)
        self.dt_s_input.setSingleStep(0.1)
        self.dt_s_input.setValue(1.0)

        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.1, 100.0)
        self.speed_input.setSingleStep(0.1)
        self.speed_input.setValue(1.0)

        self.rng_seed_input = QSpinBox()
        self.rng_seed_input.setRange(0, 999999)
        self.rng_seed_input.setValue(42)

        self.use_ekf_input = QCheckBox()
        self.use_ekf_input.setChecked(True)

        self.attack_preset_input = QComboBox()
        self.attack_preset_input.addItems([
            "none",
            "jam_cn0_drop",
            "spoof_clock_ramp",
            "spoof_pr_ramp",
            "spoof_pos_offset",
        ])
        self.attack_preset_input.currentTextChanged.connect(self._update_attack_controls)
        self.attack_preset_input.currentTextChanged.connect(self._refresh_visible_svs)

        self.target_sv_dropdown = QComboBox()
        self.refresh_svs_btn = QPushButton("Refresh SVs")
        self.refresh_svs_btn.clicked.connect(self._refresh_visible_svs)
        self.auto_select_sv_checkbox = QCheckBox("Auto-select visible SV if target not visible")
        self.auto_select_sv_checkbox.setChecked(True)
        self.run_name_input = QLineEdit("")
        self.start_t_input = QDoubleSpinBox()
        self.start_t_input.setRange(0.0, 36000.0)
        self.start_t_input.setValue(10.0)
        self.start_t_input.setDecimals(2)
        self.t_end_input = QDoubleSpinBox()
        self.t_end_input.setRange(-1.0, 1_000_000_000.0)
        self.t_end_input.setValue(-1.0)
        self.t_end_input.setDecimals(2)
        self.slope_mps_input = QDoubleSpinBox()
        self.slope_mps_input.setRange(-1000.0, 1000.0)
        self.slope_mps_input.setValue(2.0)

        # Position spoof controls (local N/E/U offset)
        self.pos_north_m_input = QDoubleSpinBox()
        self.pos_north_m_input.setRange(-1_000_000.0, 1_000_000.0)
        self.pos_north_m_input.setDecimals(2)
        self.pos_north_m_input.setValue(0.0)
        self.pos_east_m_input = QDoubleSpinBox()
        self.pos_east_m_input.setRange(-1_000_000.0, 1_000_000.0)
        self.pos_east_m_input.setDecimals(2)
        self.pos_east_m_input.setValue(0.0)
        self.pos_up_m_input = QDoubleSpinBox()
        self.pos_up_m_input.setRange(-1_000_000.0, 1_000_000.0)
        self.pos_up_m_input.setDecimals(2)
        self.pos_up_m_input.setValue(0.0)
        self.pos_ramp_time_s_input = QDoubleSpinBox()
        self.pos_ramp_time_s_input.setRange(0.0, 36000.0)
        self.pos_ramp_time_s_input.setDecimals(2)
        self.pos_ramp_time_s_input.setValue(0.0)

        # Jamming controls
        self.jam_cn0_drop_db_input = QDoubleSpinBox()
        self.jam_cn0_drop_db_input.setRange(0.0, 60.0)
        self.jam_cn0_drop_db_input.setDecimals(1)
        self.jam_cn0_drop_db_input.setValue(15.0)
        self.jam_sigma_pr_scale_input = QDoubleSpinBox()
        self.jam_sigma_pr_scale_input.setRange(1.0, 100.0)
        self.jam_sigma_pr_scale_input.setDecimals(2)
        self.jam_sigma_pr_scale_input.setValue(5.0)
        self.jam_sigma_prr_scale_input = QDoubleSpinBox()
        self.jam_sigma_prr_scale_input.setRange(1.0, 100.0)
        self.jam_sigma_prr_scale_input.setDecimals(2)
        self.jam_sigma_prr_scale_input.setValue(5.0)
        self.start_t_input.valueChanged.connect(self._refresh_visible_svs)
        self.rng_seed_input.valueChanged.connect(self._refresh_visible_svs)

        target_sv_widget = QWidget()
        target_sv_layout = QHBoxLayout(target_sv_widget)
        target_sv_layout.setContentsMargins(0, 0, 0, 0)
        target_sv_layout.addWidget(self.target_sv_dropdown, stretch=1)
        target_sv_layout.addWidget(self.refresh_svs_btn)

        self.target_sv_label = QLabel("target_sv")
        self.attack_window_label = QLabel("attack window")
        self.ramp_rate_label = QLabel("ramp_rate_mps")
        self.auto_select_label = QLabel("target behavior")

        time_window = QWidget()
        time_window_layout = QHBoxLayout(time_window)
        time_window_layout.setContentsMargins(0, 0, 0, 0)
        time_window_layout.addWidget(QLabel("start_t (s)"))
        time_window_layout.addWidget(self.start_t_input)
        time_window_layout.addSpacing(8)
        time_window_layout.addWidget(QLabel("t_end (s)"))
        time_window_layout.addWidget(self.t_end_input)

        form.addRow("duration_s", self.duration_s_input)
        form.addRow("dt_s", self.dt_s_input)
        form.addRow("speed", self.speed_input)
        form.addRow("rng_seed", self.rng_seed_input)
        form.addRow("use_ekf", self.use_ekf_input)
        self.nmea_enable_checkbox = QCheckBox("Enable NMEA (NEO-M8N: GGA+RMC @1Hz)")
        self.nmea_enable_checkbox.setChecked(True)
        self.nmea_enable_checkbox.toggled.connect(self._on_nmea_toggled)
        form.addRow("nmea", self.nmea_enable_checkbox)
        form.addRow("attack preset", self.attack_preset_input)
        form.addRow("run name", self.run_name_input)
        form.addRow(self.target_sv_label, target_sv_widget)
        form.addRow(self.auto_select_label, self.auto_select_sv_checkbox)
        form.addRow(self.attack_window_label, time_window)
        form.addRow(self.ramp_rate_label, self.slope_mps_input)
        form.addRow("pos_north_m", self.pos_north_m_input)
        form.addRow("pos_east_m", self.pos_east_m_input)
        form.addRow("pos_up_m", self.pos_up_m_input)
        form.addRow("pos_ramp_time_s", self.pos_ramp_time_s_input)
        form.addRow("jam_cn0_drop_db", self.jam_cn0_drop_db_input)
        form.addRow("jam_sigma_pr_scale", self.jam_sigma_pr_scale_input)
        form.addRow("jam_sigma_prr_scale", self.jam_sigma_prr_scale_input)

        self.init_button = QPushButton("Initialize / Reset")
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        self.step_button = QPushButton("Step")
        self.save_button = QPushButton("Save plots")
        self.open_plots_button = QPushButton("Open Plots")

        self.init_button.clicked.connect(self.initialize_reset)
        self.run_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(self.stop_run)
        self.step_button.clicked.connect(self.step_once)
        self.save_button.clicked.connect(self.save_outputs)
        self.open_plots_button.clicked.connect(self.open_diagnostics)

        btns = QGridLayout()
        btns.addWidget(self.init_button, 0, 0, 1, 2)
        btns.addWidget(self.run_button, 1, 0)
        btns.addWidget(self.stop_button, 1, 1)
        btns.addWidget(self.step_button, 2, 0)
        btns.addWidget(self.save_button, 2, 1)
        btns.addWidget(self.open_plots_button, 3, 0, 1, 2)
        form.addRow(btns)
        return group

    def _build_status_box(self) -> QGroupBox:
        group = QGroupBox("Status")
        layout = QGridLayout(group)
        keys = [
            "t_s_current",
            "fix_type",
            "sats_used",
            "pdop",
            "residual_rms_m",
            "nis_alarm",
            "conops_status",
            "attack_active",
        ]
        self.status_labels: dict[str, QLabel] = {}
        for idx, key in enumerate(keys):
            layout.addWidget(QLabel(f"{key}:"), idx // 4, (idx % 4) * 2)
            value = QLabel("-")
            self.status_labels[key] = value
            layout.addWidget(value, idx // 4, (idx % 4) * 2 + 1)
        return group

    def _build_plot_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax_flags = self.figure.add_subplot(111)

        (self.line_attack_active,) = self.ax_flags.step([], [], where="post", label="attack_active")
        (self.line_gnss_valid,) = self.ax_flags.step([], [], where="post", label="gnss_valid")
        (self.line_receiver_health,) = self.ax_flags.step(
            [], [], where="post", label="receiver_health"
        )

        self.ax_flags.set_title("Flags / Health")
        self.ax_flags.set_ylabel("state")
        self.ax_flags.set_xlabel("t_s")
        self.ax_flags.set_ylim(-0.1, 1.1)
        self.ax_flags.set_yticks([0, 1])
        self.ax_flags.grid(True, alpha=0.3)
        self.ax_flags.legend(loc="upper right")

        self.figure.tight_layout()
        layout.addWidget(self.canvas)

        self.nmea_preview = QPlainTextEdit()
        self.nmea_preview.setReadOnly(True)
        self.nmea_preview.setPlaceholderText("NMEA preview appears here while running.")
        self.nmea_preview.setMaximumBlockCount(200)
        layout.addWidget(QLabel("Live NMEA Preview"))
        layout.addWidget(self.nmea_preview)
        return box

    def _on_nmea_toggled(self, checked: bool) -> None:
        self.nmea_enabled = bool(checked)

    def _update_attack_controls(self) -> None:
        attack_name = self.attack_preset_input.currentText()
        is_ramp = attack_name in {"spoof_pr_ramp", "spoof_clock_ramp"}
        is_position = attack_name in {"spoof_pos_offset"}
        is_jam = attack_name in {"jam_cn0_drop"}
        has_time_window = attack_name != "none" and not is_jam
        has_start_only = is_jam
        for widget in [
            self.target_sv_label,
            self.target_sv_dropdown,
            self.refresh_svs_btn,
            self.auto_select_label,
            self.auto_select_sv_checkbox,
        ]:
            widget.setVisible(attack_name == "spoof_pr_ramp")
        self.refresh_svs_btn.setEnabled(attack_name == "spoof_pr_ramp")
        self.attack_window_label.setVisible(has_time_window or has_start_only)
        self.start_t_input.setEnabled(attack_name != "none")
        self.t_end_input.setEnabled(attack_name in {"spoof_pr_ramp", "spoof_clock_ramp", "spoof_pos_offset"})
        self.t_end_input.setVisible(not has_start_only)

        self.slope_mps_input.setEnabled(is_ramp)
        self.ramp_rate_label.setVisible(is_ramp)
        self.slope_mps_input.setVisible(is_ramp)

        for widget in [
            self.pos_north_m_input,
            self.pos_east_m_input,
            self.pos_up_m_input,
            self.pos_ramp_time_s_input,
        ]:
            widget.setVisible(is_position)
        for widget in [
            self.jam_cn0_drop_db_input,
            self.jam_sigma_pr_scale_input,
            self.jam_sigma_prr_scale_input,
        ]:
            widget.setVisible(is_jam)

    def _refresh_visible_svs(self) -> None:
        cfg = SimConfig(
            duration=float(self.duration_s_input.value()),
            dt=float(self.dt_s_input.value()),
            rng_seed=int(self.rng_seed_input.value()),
            use_ekf=bool(self.use_ekf_input.isChecked()),
        )
        rx_truth = self.receiver_truth_state
        if rx_truth is None:
            _, rx_truth = build_engine_with_truth(cfg)
            self.receiver_truth_state = rx_truth
        start_t = float(self.start_t_input.value())
        visible = _visible_svs_at_time(start_t, rx_truth, cfg.elev_mask_deg, cfg.rng_seed)
        self.target_sv_dropdown.clear()
        if visible:
            self.target_sv_dropdown.addItems(visible)
        else:
            self.target_sv_dropdown.addItem("NONE_VISIBLE")

    def _build_config(self) -> SimConfig:
        attack_name = self.attack_preset_input.currentText()
        attack_params: dict[str, float | str] = {}
        self.attack_config_ok = True
        self.attack_config_msg = ""
        if attack_name in {"spoof_pr_ramp", "spoof_clock_ramp"}:
            start_t = float(self.start_t_input.value())
            ramp_rate_mps = float(self.slope_mps_input.value())
            t_end_val = float(self.t_end_input.value())
            end_t = None if t_end_val < 0 else t_end_val
            attack_params = {
                "start_t": start_t,
                "ramp_rate_mps": ramp_rate_mps,
            }
            if end_t is not None:
                attack_params["end_t"] = end_t
            if attack_name == "spoof_pr_ramp":
                partial_cfg = SimConfig(rng_seed=int(self.rng_seed_input.value()))
                rx_truth = self.receiver_truth_state
                if rx_truth is None:
                    _, rx_truth = build_engine_with_truth(partial_cfg)
                    self.receiver_truth_state = rx_truth
                visible = _visible_svs_at_time(
                    start_t,
                    rx_truth,
                    partial_cfg.elev_mask_deg,
                    partial_cfg.rng_seed,
                )
                chosen = self.target_sv_dropdown.currentText().strip()
                resolved, ok, msg = _resolve_target_sv(
                    chosen,
                    visible,
                    self.auto_select_sv_checkbox.isChecked(),
                )
                if not ok:
                    self.attack_config_ok = False
                    self.attack_config_msg = msg
                    raise ValueError(
                        f"Target SV {chosen} is not visible at t={start_t:.2f}s. Visible: {visible}"
                    )
                if msg:
                    self.attack_config_msg = msg
                if resolved != chosen:
                    self.target_sv_dropdown.setCurrentText(resolved)
                attack_params["target_sv"] = resolved
                attack_params["auto_select_visible_sv"] = bool(self.auto_select_sv_checkbox.isChecked())
                attack_params["strict_target_sv"] = False
        elif attack_name == "spoof_pos_offset":
            start_t = float(self.start_t_input.value())
            t_end_val = float(self.t_end_input.value())
            end_t = None if t_end_val < 0 else t_end_val
            attack_params = {
                "start_t": start_t,
                "north_m": float(self.pos_north_m_input.value()),
                "east_m": float(self.pos_east_m_input.value()),
                "up_m": float(self.pos_up_m_input.value()),
                "ramp_time_s": float(self.pos_ramp_time_s_input.value()),
            }
            if end_t is not None:
                attack_params["end_t"] = end_t
        elif attack_name == "jam_cn0_drop":
            start_t = float(self.start_t_input.value())
            attack_params = {
                "start_t": start_t,
                "cn0_drop_db": float(self.jam_cn0_drop_db_input.value()),
                "sigma_pr_scale": float(self.jam_sigma_pr_scale_input.value()),
                "sigma_prr_scale": float(self.jam_sigma_prr_scale_input.value()),
            }
        return SimConfig(
            duration=float(self.duration_s_input.value()),
            dt=float(self.dt_s_input.value()),
            rng_seed=int(self.rng_seed_input.value()),
            use_ekf=bool(self.use_ekf_input.isChecked()),
            attack_name=attack_name,
            attack_params=attack_params,
        )

    def initialize_reset(self) -> None:
        self.stop_run()
        try:
            self.cfg = self._build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid attack configuration", str(exc))
            return
        self.engine, self.receiver_truth_state = build_engine_with_truth(self.cfg)
        self.t_s_current = 0.0
        self.epochs = []
        self.meas_log_rows = []  # per-SV measurement audit log for post-analysis
        self.frame = pd.DataFrame()
        self._live_frame_dirty = False
        self._steps_since_live_frame_rebuild = 0
        self.nmea.reset()
        self.nmea_t0_utc = datetime.now(timezone.utc)
        self.nmea_buffer.clear()
        self.nmea_recent.clear()
        self.last_nmea_preview_update_walltime = 0.0
        self.nmea_preview.clear()
        self.attack_never_applied_warned = False
        self._refresh_flags_plot()
        if self.diagnostics_window is not None:
            self.diagnostics_window.update_plots(self.frame)
        self._update_status_labels()

    def _compute_timer_ms(self) -> int:
        dt = float(self.dt_s_input.value())
        speed = max(float(self.speed_input.value()), 1e-6)
        return max(10, int(1000.0 * dt / speed))

    def start_run(self) -> None:
        if self.engine is None:
            self.initialize_reset()
        if self.engine is None:
            return
        if not self.epochs and self.t_s_current == 0.0:
            self.nmea.reset()
            self.nmea_t0_utc = datetime.now(timezone.utc)
            self.nmea_buffer.clear()
            self.nmea_recent.clear()
            self._maybe_update_nmea_preview(force=True)
        self.is_running = True
        self.timer.start(self._compute_timer_ms())

    def stop_run(self) -> None:
        self.is_running = False
        self.timer.stop()
        self._refresh_live_frame_if_needed(force=True)

    def step_once(self) -> None:
        if self.engine is None or self.cfg is None or self.receiver_truth_state is None:
            return
        if self.t_s_current >= float(self.cfg.duration):
            self.stop_run()
            self._warn_if_attack_never_applied()
            return

        step_out = self.engine.step(float(self.t_s_current))
        epoch = build_epoch_log(
            t_s=float(self.t_s_current),
            step_out=step_out,
            integrity_checker=self.engine.integrity_checker,
            attack_name=self.cfg.attack_name or "none",
        )
        self.epochs.append(epoch)
        self._append_meas_log(step_out, epoch)
        self._live_frame_dirty = True
        self._steps_since_live_frame_rebuild += 1
        self._step_nmea(float(self.t_s_current), epoch, step_out)

        self.t_s_current += float(self.cfg.dt)
        self._update_status_labels(epoch)
        if self._refresh_live_frame_if_needed():
            self._refresh_flags_plot()
            self._maybe_update_diagnostics()
        self._maybe_update_nmea_preview()

        if self.t_s_current >= float(self.cfg.duration):
            self.stop_run()
            self._refresh_live_frame_if_needed(force=True)
            self._refresh_flags_plot()
            self._maybe_update_diagnostics(force=True)
            self._warn_if_attack_never_applied()

    def _refresh_live_frame_if_needed(self, *, force: bool = False) -> bool:
        if not self._live_frame_dirty:
            return False
        if not force and self._steps_since_live_frame_rebuild < LIVE_FRAME_REBUILD_EVERY_STEPS:
            return False
        self.frame = epochs_to_frame(self.epochs)
        self._live_frame_dirty = False
        self._steps_since_live_frame_rebuild = 0
        return True

    def _step_nmea(self, t_s: float, epoch: EpochLog, step_out: dict) -> None:
        if not self.nmea_enabled:
            return
        sol = step_out.get("sol")
        if sol is None:
            return
        lat_deg, lon_deg, alt_m = ecef_to_lla(*sol.pos_ecef)
        raim_valid = bool(sol.fix_flags.raim_passed)
        num_sats = int(epoch.sats_used or 0)
        hdop = epoch.hdop
        if hdop is None and epoch.pdop is not None:
            hdop = float(epoch.pdop)
        elif hdop is None:
            hdop = float("nan")
        t_utc = self.nmea_t0_utc + timedelta(seconds=t_s)
        lines = self.nmea.step(
            t_s,
            t_utc=t_utc,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            alt_m=alt_m,
            raim_valid=raim_valid,
            num_sats=num_sats,
            hdop=float(hdop),
        )
        if not lines:
            return
        self.nmea_buffer.extend(lines)
        self.nmea_recent.extend(lines)

    def _maybe_update_nmea_preview(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_nmea_preview_update_walltime < NMEA_PREVIEW_UPDATE_PERIOD_S:
            return
        self.last_nmea_preview_update_walltime = now
        recent_lines = list(self.nmea_recent)[-NMEA_PREVIEW_LINES:]
        self.nmea_preview.setPlainText("".join(f"{emit.nmea_sentence}\n" for emit in recent_lines))

    def _update_status_labels(self, epoch: EpochLog | None = None) -> None:
        values = {
            "t_s_current": f"{self.t_s_current:.2f}",
            "fix_type": "-",
            "sats_used": "-",
            "pdop": "-",
            "residual_rms_m": "-",
            "nis_alarm": "-",
            "conops_status": "-",
            "attack_active": "-",
        }
        if epoch is not None:
            values.update(
                {
                    "fix_type": str(epoch.solution.fix_flags.fix_type if epoch.solution else "-"),
                    "sats_used": str(epoch.sats_used),
                    "pdop": f"{epoch.pdop:.3f}" if epoch.pdop is not None else "-",
                    "residual_rms_m": (
                        f"{epoch.residual_rms_m:.3f}" if epoch.residual_rms_m is not None else "-"
                    ),
                    "nis_alarm": "YES" if epoch.nis_alarm else "NO",
                    "conops_status": str(epoch.conops_status),
                    "attack_active": "YES" if epoch.attack_active else "NO",
                }
            )
        for key, label in self.status_labels.items():
            label.setText(values[key])

    def _derive_binary_flags(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        gnss_valid = (
            (frame["sats_used"] >= 4)
            & (frame["residual_rms_m"] <= RESID_RMS_OK_M)
            & (frame["pdop"] <= PDOP_OK)
        ).fillna(False)
        receiver_health = (
            gnss_valid
            & frame["residual_rms_m"].notna()
            & frame[["pos_ecef_x", "pos_ecef_y", "pos_ecef_z"]].notna().all(axis=1)
        ).fillna(False)
        return gnss_valid.astype(float), receiver_health.astype(float)

    def _refresh_flags_plot(self) -> None:
        if self.frame.empty:
            for line in [self.line_attack_active, self.line_gnss_valid, self.line_receiver_health]:
                line.set_data([], [])
            self.canvas.draw_idle()
            return

        t_vals = self.frame["t_s"].to_numpy(dtype=float)
        gnss_valid, receiver_health = self._derive_binary_flags(self.frame)

        self.line_attack_active.set_data(t_vals, self.frame["attack_active"].astype(float).to_numpy())
        self.line_gnss_valid.set_data(t_vals, gnss_valid.to_numpy(dtype=float))
        self.line_receiver_health.set_data(t_vals, receiver_health.to_numpy(dtype=float))

        self.ax_flags.relim()
        self.ax_flags.autoscale_view(scaley=False)
        self.ax_flags.set_ylim(-0.1, 1.1)

        self.canvas.draw_idle()

    def open_diagnostics(self) -> None:
        if self.diagnostics_window is None:
            self.diagnostics_window = DiagnosticsWindow()
            self.diagnostics_window.destroyed.connect(self._on_diagnostics_closed)
        self.diagnostics_window.show()
        self.diagnostics_window.raise_()
        self.diagnostics_window.activateWindow()
        self.last_diag_update_walltime = 0.0
        self._maybe_update_diagnostics(force=True)

    def _on_diagnostics_closed(self) -> None:
        self.diagnostics_window = None

    def _maybe_update_diagnostics(self, force: bool = False) -> None:
        if self.diagnostics_window is None:
            return
        now = time.monotonic()
        if not force and now - self.last_diag_update_walltime < DIAG_UPDATE_PERIOD_S:
            return
        self.last_diag_update_walltime = now
        self.diagnostics_window.update_plots(self.frame)

    def save_outputs(self) -> None:
        if not self.epochs:
            QMessageBox.information(self, "No data", "Run the simulation before saving outputs.")
            return

        base_out = Path("out")
        date_dir = base_out / _date_folder_str()
        run_name = _sanitize_run_name(
            self.run_name_input.text() or self.current_run_name or ""
        )
        run_dir = _ensure_unique_dir(date_dir / run_name)
        run_dir.mkdir(parents=True, exist_ok=False)

        df = epochs_to_frame(self.epochs)
        for column, default_value in RUN_TABLE_INTEGRITY_COLUMNS.items():
            if column not in df.columns:
                df[column] = default_value
        if self.cfg is not None:
            attack_name = self.cfg.attack_name or "none"
            attack_params = self.cfg.attack_params
            df["attack_name"] = attack_name
            df["attack_start_t_s"] = attack_params.get("start_t", "")
            df["attack_end_t_s"] = attack_params.get("end_t", "")
            df["attack_ramp_rate_mps"] = attack_params.get("ramp_rate_mps", "")
            target_sv = attack_params.get("target_sv", "") if attack_name == "spoof_pr_ramp" else ""
            df["attack_target_sv"] = target_sv
            df["attack_pos_north_m"] = attack_params.get("north_m", "")
            df["attack_pos_east_m"] = attack_params.get("east_m", "")
            df["attack_pos_up_m"] = attack_params.get("up_m", "")
            df["attack_pos_ramp_time_s"] = attack_params.get("ramp_time_s", "")
            df["attack_jam_cn0_drop_db"] = attack_params.get("cn0_drop_db", "")
            df["attack_jam_sigma_pr_scale"] = attack_params.get("sigma_pr_scale", "")
            df["attack_jam_sigma_prr_scale"] = attack_params.get("sigma_prr_scale", "")
            df["attack_config_ok"] = bool(self.attack_config_ok)
            df["attack_config_msg"] = self.attack_config_msg
            add_nmea_metadata_columns(df, self.cfg)
        else:
            add_nmea_metadata_columns(df, SimConfig())
        self.frame = df
        df.to_csv(run_dir / "run_table.csv", index=False)
        if getattr(self, "meas_log_rows", None):
            meas_path = run_dir / "meas_log.csv"
            pd.DataFrame(self.meas_log_rows).to_csv(meas_path, index=False)
        plot_update(df, out_dir=run_dir, run_name=None)
        (run_dir / "nmea_output.nmea").write_text(
            "".join(f"{emit.nmea_sentence}\r\n" for emit in self.nmea_buffer),
            encoding="utf-8",
        )
        with (run_dir / "nmea_output.csv").open("w", encoding="utf-8", newline="") as nmea_csv:
            writer = csv.DictWriter(
                nmea_csv,
                fieldnames=["t_s", "t_utc_iso", "valid", "sentence_type", "talker", "nmea_sentence"],
            )
            writer.writeheader()
            for emit in self.nmea_buffer:
                writer.writerow({
                    "t_s": emit.t_s,
                    "t_utc_iso": emit.t_utc_iso,
                    "valid": emit.valid,
                    "sentence_type": emit.sentence_type,
                    "talker": emit.talker,
                    "nmea_sentence": emit.nmea_sentence,
                })
        with (run_dir / "run_metadata.csv").open("w", encoding="utf-8", newline="") as metadata_file:
            writer = csv.DictWriter(
                metadata_file,
                fieldnames=[
                    "rx_lat_deg",
                    "rx_lon_deg",
                    "rx_alt_m",
                    "nmea_profile",
                    "nmea_rate_hz",
                    "nmea_msgs",
                    "nmea_talker",
                ],
            )
            writer.writeheader()
            cfg = self.cfg or SimConfig()
            writer.writerow(
                {
                    "rx_lat_deg": cfg.rx_lat_deg,
                    "rx_lon_deg": cfg.rx_lon_deg,
                    "rx_alt_m": cfg.rx_alt_m,
                    "nmea_profile": "NEO-M8N",
                    "nmea_rate_hz": 1.0,
                    "nmea_msgs": "GGA,RMC",
                    "nmea_talker": "GN",
                }
            )
        self._warn_if_attack_never_applied()

        QMessageBox.information(self, "Saved", f"Saved to: {run_dir}")

    def _warn_if_attack_never_applied(self) -> None:
        if self.attack_never_applied_warned or self.cfg is None or self.frame.empty:
            return
        attack_name = self.cfg.attack_name or "none"
        if attack_name == "none":
            return
        if bool(self.frame["attack_active"].astype(bool).max()):
            return
        self.attack_never_applied_warned = True
        QMessageBox.warning(
            self,
            "Attack not applied",
            (
                "Attack was configured but never applied. For spoof_pr_ramp, "
                "confirm target SV is visible at start_t."
            ),
        )

    def _append_meas_log(self, step_out: dict, epoch) -> None:
        """Append per-SV measurement telemetry for debugging/validation."""

        try:
            raw = {m.sv_id: m for m in step_out.get("meas_raw", [])}
            attacked = {m.sv_id: m for m in step_out.get("meas_attacked", [])}
            sol = step_out.get("sol")
            used = set(sol.fix_flags.sv_used) if sol is not None else set()
            rejected = set(sol.fix_flags.sv_rejected) if sol is not None else set()
            t_s = epoch.t_s if getattr(epoch, "t_s", None) is not None else epoch.t
            for sv_id, meas_a in attacked.items():
                meas_r = raw.get(sv_id)
                pr_raw = meas_r.pr_m if meas_r is not None else float("nan")
                prr_raw = meas_r.prr_mps if meas_r is not None else float("nan")
                pr_bias = meas_a.pr_m - pr_raw if meas_r is not None else float("nan")
                prr_bias = (
                    meas_a.prr_mps - prr_raw
                    if (meas_r is not None and meas_a.prr_mps is not None)
                    else float("nan")
                )
                self.meas_log_rows.append(
                    {
                        "t_s": t_s,
                        "sv_id": sv_id,
                        "elev_deg": meas_a.elev_deg,
                        "az_deg": meas_a.az_deg,
                        "cn0_dbhz": meas_a.cn0_dbhz,
                        "sigma_pr_m": meas_a.sigma_pr_m,
                        "pr_raw_m": pr_raw,
                        "pr_attacked_m": meas_a.pr_m,
                        "pr_bias_m": pr_bias,
                        "prr_raw_mps": prr_raw,
                        "prr_attacked_mps": (
                            meas_a.prr_mps if meas_a.prr_mps is not None else float("nan")
                        ),
                        "prr_bias_mps": prr_bias,
                        "used_in_solution": int(sv_id in used),
                        "rejected": int(sv_id in rejected),
                    }
                )
        except Exception:
            return


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
