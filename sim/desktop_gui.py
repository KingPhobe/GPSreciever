"""Native desktop GUI for live GNSS simulation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from gnss_twin.config import SimConfig
from gnss_twin.models import EpochLog, ReceiverTruth
from gnss_twin.plots import epochs_to_frame, plot_update
from sim.run_static_demo import build_engine_with_truth, build_epoch_log


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

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step_once)

        self._build_ui()
        self._update_attack_controls()
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
        self.attack_preset_input.addItems(["none", "spoof_pr_ramp"])
        self.attack_preset_input.currentTextChanged.connect(self._update_attack_controls)

        self.target_sv_input = QLineEdit("G12")
        self.start_t_input = QDoubleSpinBox()
        self.start_t_input.setRange(0.0, 36000.0)
        self.start_t_input.setValue(10.0)
        self.slope_mps_input = QDoubleSpinBox()
        self.slope_mps_input.setRange(-1000.0, 1000.0)
        self.slope_mps_input.setValue(2.0)

        form.addRow("duration_s", self.duration_s_input)
        form.addRow("dt_s", self.dt_s_input)
        form.addRow("speed", self.speed_input)
        form.addRow("rng_seed", self.rng_seed_input)
        form.addRow("use_ekf", self.use_ekf_input)
        form.addRow("attack preset", self.attack_preset_input)
        form.addRow("target_sv", self.target_sv_input)
        form.addRow("start_t", self.start_t_input)
        form.addRow("slope_mps", self.slope_mps_input)

        self.init_button = QPushButton("Initialize / Reset")
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        self.step_button = QPushButton("Step")
        self.save_button = QPushButton("Save plots")

        self.init_button.clicked.connect(self.initialize_reset)
        self.run_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(self.stop_run)
        self.step_button.clicked.connect(self.step_once)
        self.save_button.clicked.connect(self.save_outputs)

        btns = QGridLayout()
        btns.addWidget(self.init_button, 0, 0, 1, 2)
        btns.addWidget(self.run_button, 1, 0)
        btns.addWidget(self.stop_button, 1, 1)
        btns.addWidget(self.step_button, 2, 0)
        btns.addWidget(self.save_button, 2, 1)
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
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax_pdop = self.figure.add_subplot(311)
        self.ax_residual = self.figure.add_subplot(312)
        self.ax_clock = self.figure.add_subplot(313)

        (self.line_pdop,) = self.ax_pdop.plot([], [], color="tab:blue")
        self.ax_pdop.set_ylabel("PDOP")
        self.ax_pdop.grid(True, alpha=0.3)

        (self.line_residual,) = self.ax_residual.plot([], [], color="tab:green")
        self.ax_residual.set_ylabel("Residual RMS (m)")
        self.ax_residual.grid(True, alpha=0.3)

        (self.line_clock,) = self.ax_clock.plot([], [], color="tab:orange")
        self.ax_clock.set_ylabel("Clock bias (s)")
        self.ax_clock.set_xlabel("t_s")
        self.ax_clock.grid(True, alpha=0.3)

        self.figure.tight_layout()
        layout.addWidget(self.canvas)
        return box

    def _update_attack_controls(self) -> None:
        is_spoof = self.attack_preset_input.currentText() == "spoof_pr_ramp"
        self.target_sv_input.setEnabled(is_spoof)
        self.start_t_input.setEnabled(is_spoof)
        self.slope_mps_input.setEnabled(is_spoof)

    def _build_config(self) -> SimConfig:
        attack_name = self.attack_preset_input.currentText()
        attack_params: dict[str, float | str] = {}
        if attack_name == "spoof_pr_ramp":
            attack_params = {
                "target_sv": self.target_sv_input.text().strip(),
                "start_t": float(self.start_t_input.value()),
                "ramp_rate_mps": float(self.slope_mps_input.value()),
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
        self.cfg = self._build_config()
        self.engine, self.receiver_truth_state = build_engine_with_truth(self.cfg)
        self.t_s_current = 0.0
        self.epochs = []
        self.frame = pd.DataFrame()
        self._refresh_plots()
        self._update_status_labels()

    def _compute_timer_ms(self) -> int:
        dt = float(self.dt_s_input.value())
        speed = max(float(self.speed_input.value()), 1e-6)
        return max(10, int(1000.0 * dt / speed))

    def start_run(self) -> None:
        if self.engine is None:
            self.initialize_reset()
        self.is_running = True
        self.timer.start(self._compute_timer_ms())

    def stop_run(self) -> None:
        self.is_running = False
        self.timer.stop()

    def step_once(self) -> None:
        if self.engine is None or self.cfg is None or self.receiver_truth_state is None:
            return
        if self.t_s_current >= float(self.cfg.duration):
            self.stop_run()
            return

        step_out = self.engine.step(float(self.t_s_current))
        epoch = build_epoch_log(
            t_s=float(self.t_s_current),
            step_out=step_out,
            receiver_truth_state=self.receiver_truth_state,
            integrity_checker=self.engine.integrity_checker,
            attack_name=self.cfg.attack_name or "none",
        )
        self.epochs.append(epoch)
        self.frame = epochs_to_frame(self.epochs)

        self.t_s_current += float(self.cfg.dt)
        self._update_status_labels(epoch)
        self._refresh_plots()

        if self.t_s_current >= float(self.cfg.duration):
            self.stop_run()

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

    def _refresh_plots(self) -> None:
        if self.frame.empty:
            for line in [self.line_pdop, self.line_residual, self.line_clock]:
                line.set_data([], [])
            self.canvas.draw_idle()
            return

        t_vals = self.frame["t_s"].to_numpy(dtype=float)
        pdop_vals = self.frame["pdop"].to_numpy(dtype=float)
        residual_vals = self.frame["residual_rms_m"].to_numpy(dtype=float)
        clock_vals = self.frame["clk_bias_s"].to_numpy(dtype=float)

        self.line_pdop.set_data(t_vals, pdop_vals)
        self.line_residual.set_data(t_vals, residual_vals)
        self.line_clock.set_data(t_vals, clock_vals)

        for axis in [self.ax_pdop, self.ax_residual, self.ax_clock]:
            axis.relim()
            axis.autoscale_view()

        self.canvas.draw_idle()

    def save_outputs(self) -> None:
        if not self.epochs:
            QMessageBox.information(self, "No data", "Run the simulation before saving outputs.")
            return

        default_dir = Path("out")
        default_dir.mkdir(parents=True, exist_ok=True)
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder", str(default_dir))
        if not out_dir:
            return

        run_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(out_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        df = epochs_to_frame(self.epochs)
        self.frame = df
        df.to_csv(run_dir / "run_table.csv", index=False)
        plot_update(df, out_dir=str(run_dir), run_name=None)

        QMessageBox.information(self, "Saved", f"Saved PNG plots and CSV files to {run_dir}")


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
