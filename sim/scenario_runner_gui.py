from __future__ import annotations

import json
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
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
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

ATTACK_DEFAULTS: dict[str, dict[str, Any]] = {
    "none": {},
    "jam_cn0_drop": {
        "start_t": 20.0,
        "cn0_drop_db": 15.0,
        "sigma_pr_scale": 5.0,
        "sigma_prr_scale": 5.0,
    },
    "spoof_clock_ramp": {
        "start_t": 20.0,
        "end_t": "",
        "ramp_rate_mps": 1.0,
    },
    "spoof_pr_ramp": {
        "start_t": 20.0,
        "end_t": "",
        "ramp_rate_mps": 1.0,
        "target_sv": "G01",
        "auto_select_visible_sv": False,
        "strict_target_sv": True,
    },
    "spoof_pos_offset": {
        "start_t": 20.0,
        "end_t": "",
        "north_m": 100.0,
        "east_m": 0.0,
        "up_m": 0.0,
        "ramp_time_s": 0.0,
    },
}


@dataclass
class ScenarioDraft:
    name: str
    duration_s: float
    rng_seed: int
    use_ekf: bool
    attack_name: str
    attack_params: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "duration_s": self.duration_s,
            "rng_seed": self.rng_seed,
            "use_ekf": self.use_ekf,
            "attack_name": self.attack_name,
            "attack_params": self.attack_params,
        }
        payload.update(self.overrides)
        return payload


class RunWorker(QObject):
    log = pyqtSignal(str)
    finished = pyqtSignal(bool, str)

    def __init__(
        self,
        *,
        scenarios: list[dict[str, Any]],
        run_root: str,
        save_plots: bool,
        monte_carlo_n: int,
        mc_seed_mode: str,
        mc_seed_start: int,
        mc_seed_step: int,
        mc_per_run_plots: bool,
        mc_aggregate_plots: bool,
    ) -> None:
        super().__init__()
        self.scenarios = scenarios
        self.run_root = run_root
        self.save_plots = save_plots
        self.monte_carlo_n = monte_carlo_n
        self.mc_seed_mode = mc_seed_mode
        self.mc_seed_start = mc_seed_start
        self.mc_seed_step = mc_seed_step
        self.mc_per_run_plots = mc_per_run_plots
        self.mc_aggregate_plots = mc_aggregate_plots

    def run(self) -> None:
        try:
            from sim.scenario_runner import run_scenarios
            from sim.validation.monte_carlo import run_monte_carlo

            run_root = Path(self.run_root).expanduser()
            run_root.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Run root: {run_root}")

            with tempfile.TemporaryDirectory(prefix="gnss_scenarios_") as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                scenario_paths: list[Path] = []
                for idx, payload in enumerate(self.scenarios, start=1):
                    name = str(payload.get("name") or f"scenario_{idx}")
                    path = tmp_dir / f"{idx:02d}_{_slugify(name)}.json"
                    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    scenario_paths.append(path)
                    self.log.emit(f"Prepared: {path.name}")

                if self.monte_carlo_n > 0:
                    for scenario_path in scenario_paths:
                        self.log.emit(
                            f"MC start: {scenario_path.name} (n={self.monte_carlo_n}, mode={self.mc_seed_mode})"
                        )
                        report = run_monte_carlo(
                            scenario_path,
                            n=int(self.monte_carlo_n),
                            seed_mode=str(self.mc_seed_mode),
                            seed_start=int(self.mc_seed_start),
                            seed_step=int(self.mc_seed_step),
                            run_root=run_root,
                            per_run_plots=bool(self.mc_per_run_plots),
                            aggregate_plots=bool(self.mc_aggregate_plots),
                        )
                        self.log.emit(
                            f"MC done: {report.get('scenario')} -> {report.get('mc_dir')}"
                        )
                    self.finished.emit(True, "Monte Carlo runs completed.")
                    return

                self.log.emit(f"Running {len(scenario_paths)} scenario(s) headless...")
                summaries = run_scenarios(
                    scenario_paths,
                    run_root=run_root,
                    save_figs=bool(self.save_plots),
                )
                for summary in summaries:
                    self.log.emit(
                        f"Done: {summary.get('scenario')} | pos_err_rms={summary.get('pos_err_rms')} | {summary.get('run_dir')}"
                    )
                self.finished.emit(True, f"Completed {len(summaries)} scenario run(s).")
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=20)
            self.log.emit(tb)
            self.finished.emit(False, f"Run failed: {exc}")


class ScenarioRunnerGui(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GNSS Twin Scenario Runner (Headless GUI)")
        self.resize(1200, 820)

        self._queue: list[dict[str, Any]] = []
        self._thread: QThread | None = None
        self._worker: RunWorker | None = None

        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setSizes([700, 500])

        # ---------- Scenario editor ----------
        editor_group = QGroupBox("Scenario Builder")
        left_layout.addWidget(editor_group)
        editor_layout = QVBoxLayout(editor_group)

        form_box = QWidget()
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        editor_layout.addWidget(form_box)

        self.name_edit = QLineEdit("baseline_gui")
        self.duration_edit = QLineEdit("60")
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)
        self.use_ekf_chk = QCheckBox("Enable EKF")
        self.use_ekf_chk.setChecked(True)
        self.attack_combo = QComboBox()
        self.attack_combo.addItems(list(ATTACK_DEFAULTS.keys()))
        self.attack_combo.currentTextChanged.connect(self._rebuild_attack_param_form)

        form.addRow("Name", self.name_edit)
        form.addRow("Duration [s]", self.duration_edit)
        form.addRow("RNG seed", self.seed_spin)
        form.addRow("Use EKF", self.use_ekf_chk)
        form.addRow("Attack", self.attack_combo)

        self.attack_param_group = QGroupBox("Attack Parameters")
        self.attack_param_layout = QFormLayout(self.attack_param_group)
        self.attack_param_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        editor_layout.addWidget(self.attack_param_group)
        self._attack_param_widgets: dict[str, QWidget] = {}

        self.overrides_text = QPlainTextEdit()
        self.overrides_text.setPlaceholderText(
            '{\n  "dt": 1.0,\n  "rx_lat_deg": 36.597383,\n  "rx_lon_deg": -121.8743\n}\n\nOptional extra SimConfig overrides (JSON object).'
        )
        self.overrides_text.setFixedHeight(150)
        editor_layout.addWidget(QLabel("Advanced SimConfig overrides (JSON object, optional):"))
        editor_layout.addWidget(self.overrides_text)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add to Queue")
        self.update_btn = QPushButton("Update Selected")
        self.clear_form_btn = QPushButton("Reset Form")
        self.save_json_btn = QPushButton("Save JSON...")
        self.load_json_btn = QPushButton("Load JSON...")
        for b in [self.add_btn, self.update_btn, self.clear_form_btn, self.save_json_btn, self.load_json_btn]:
            btn_row.addWidget(b)
        editor_layout.addLayout(btn_row)

        self.add_btn.clicked.connect(self._add_to_queue)
        self.update_btn.clicked.connect(self._update_selected_queue_item)
        self.clear_form_btn.clicked.connect(self._reset_form)
        self.save_json_btn.clicked.connect(self._save_form_as_json)
        self.load_json_btn.clicked.connect(self._load_json_into_form)

        # ---------- Queue ----------
        queue_group = QGroupBox("Scenario Queue")
        left_layout.addWidget(queue_group, 1)
        queue_layout = QVBoxLayout(queue_group)

        self.queue_list = QListWidget()
        self.queue_list.itemSelectionChanged.connect(self._load_selected_queue_item_into_form)
        queue_layout.addWidget(self.queue_list, 1)

        queue_btns = QHBoxLayout()
        self.remove_btn = QPushButton("Remove")
        self.duplicate_btn = QPushButton("Duplicate")
        self.import_btn = QPushButton("Import JSONs...")
        self.export_btn = QPushButton("Export Selected JSON...")
        self.clear_queue_btn = QPushButton("Clear Queue")
        for b in [self.remove_btn, self.duplicate_btn, self.import_btn, self.export_btn, self.clear_queue_btn]:
            queue_btns.addWidget(b)
        queue_layout.addLayout(queue_btns)

        self.remove_btn.clicked.connect(self._remove_selected)
        self.duplicate_btn.clicked.connect(self._duplicate_selected)
        self.import_btn.clicked.connect(self._import_jsons_to_queue)
        self.export_btn.clicked.connect(self._export_selected_json)
        self.clear_queue_btn.clicked.connect(self._clear_queue)

        # ---------- Runner settings ----------
        runner_group = QGroupBox("Runner Settings")
        right_layout.addWidget(runner_group)
        runner_grid = QGridLayout(runner_group)

        self.run_root_edit = QLineEdit(str(Path("runs").resolve()))
        self.browse_run_root_btn = QPushButton("Browse...")
        self.save_plots_chk = QCheckBox("Save plots")
        self.save_plots_chk.setChecked(True)

        runner_grid.addWidget(QLabel("Run root"), 0, 0)
        runner_grid.addWidget(self.run_root_edit, 0, 1)
        runner_grid.addWidget(self.browse_run_root_btn, 0, 2)
        runner_grid.addWidget(self.save_plots_chk, 1, 0, 1, 2)

        self.browse_run_root_btn.clicked.connect(self._browse_run_root)

        # Monte Carlo settings
        mc_group = QGroupBox("Monte Carlo (optional)")
        right_layout.addWidget(mc_group)
        mc_grid = QGridLayout(mc_group)

        self.mc_enable_chk = QCheckBox("Enable Monte Carlo mode")
        self.mc_n_spin = QSpinBox()
        self.mc_n_spin.setRange(0, 100000)
        self.mc_n_spin.setValue(20)
        self.mc_seed_mode_combo = QComboBox()
        self.mc_seed_mode_combo.addItems(["offset", "absolute"])
        self.mc_seed_start_spin = QSpinBox()
        self.mc_seed_start_spin.setRange(-2_147_483_648, 2_147_483_647)
        self.mc_seed_start_spin.setValue(0)
        self.mc_seed_step_spin = QSpinBox()
        self.mc_seed_step_spin.setRange(1, 1_000_000)
        self.mc_seed_step_spin.setValue(1)
        self.mc_per_run_plots_chk = QCheckBox("MC per-run plots (heavy)")
        self.mc_agg_plots_chk = QCheckBox("MC aggregate plots")
        self.mc_agg_plots_chk.setChecked(True)

        mc_grid.addWidget(self.mc_enable_chk, 0, 0, 1, 3)
        mc_grid.addWidget(QLabel("N runs"), 1, 0)
        mc_grid.addWidget(self.mc_n_spin, 1, 1)
        mc_grid.addWidget(QLabel("Seed mode"), 2, 0)
        mc_grid.addWidget(self.mc_seed_mode_combo, 2, 1)
        mc_grid.addWidget(QLabel("Seed start"), 3, 0)
        mc_grid.addWidget(self.mc_seed_start_spin, 3, 1)
        mc_grid.addWidget(QLabel("Seed step"), 4, 0)
        mc_grid.addWidget(self.mc_seed_step_spin, 4, 1)
        mc_grid.addWidget(self.mc_per_run_plots_chk, 5, 0, 1, 3)
        mc_grid.addWidget(self.mc_agg_plots_chk, 6, 0, 1, 3)

        # Run controls
        controls_group = QGroupBox("Run Controls")
        right_layout.addWidget(controls_group)
        controls_layout = QHBoxLayout(controls_group)
        self.run_btn = QPushButton("Run Queue")
        self.stop_note = QLabel("Runs execute in background; close window only after completion.")
        self.stop_note.setWordWrap(True)
        controls_layout.addWidget(self.run_btn)
        controls_layout.addWidget(self.stop_note, 1)
        self.run_btn.clicked.connect(self._run_queue)

        # Log panel
        log_group = QGroupBox("Execution Log")
        right_layout.addWidget(log_group, 1)
        log_layout = QVBoxLayout(log_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text, 1)

        self._rebuild_attack_param_form(self.attack_combo.currentText())

    # ---------------- form helpers ----------------
    def _log(self, msg: str) -> None:
        self.log_text.appendPlainText(msg)

    def _clear_attack_param_widgets(self) -> None:
        while self.attack_param_layout.rowCount() > 0:
            self.attack_param_layout.removeRow(0)
        self._attack_param_widgets.clear()

    def _rebuild_attack_param_form(self, attack_name: str) -> None:
        self._clear_attack_param_widgets()
        defaults = ATTACK_DEFAULTS.get(attack_name, {})
        if not defaults:
            self.attack_param_layout.addRow(QLabel("No parameters for selected attack."))
            return
        for key, default in defaults.items():
            widget: QWidget
            if isinstance(default, bool):
                chk = QCheckBox()
                chk.setChecked(bool(default))
                widget = chk
            else:
                edit = QLineEdit(str(default))
                widget = edit
            self._attack_param_widgets[key] = widget
            self.attack_param_layout.addRow(key, widget)

    def _attack_params_from_form(self) -> dict[str, Any]:
        attack = self.attack_combo.currentText()
        params: dict[str, Any] = {}
        for key, widget in self._attack_param_widgets.items():
            if isinstance(widget, QCheckBox):
                params[key] = bool(widget.isChecked())
                continue
            if not isinstance(widget, QLineEdit):
                continue
            raw = widget.text().strip()
            if raw == "":
                continue
            default = ATTACK_DEFAULTS.get(attack, {}).get(key)
            if isinstance(default, str):
                params[key] = raw
            else:
                params[key] = _coerce_scalar(raw)
        return params

    def _overrides_from_form(self) -> dict[str, Any]:
        raw = self.overrides_text.toPlainText().strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid overrides JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Overrides must be a JSON object")
        reserved = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
        overlap = reserved & set(parsed.keys())
        if overlap:
            raise ValueError(f"Overrides cannot include reserved keys: {sorted(overlap)}")
        return parsed

    def _draft_from_form(self) -> ScenarioDraft:
        name = self.name_edit.text().strip()
        if not name:
            raise ValueError("Scenario name is required")
        try:
            duration_s = float(self.duration_edit.text().strip())
        except ValueError as exc:
            raise ValueError("Duration must be numeric") from exc
        if duration_s <= 0:
            raise ValueError("Duration must be > 0")
        attack_name = self.attack_combo.currentText().strip() or "none"
        return ScenarioDraft(
            name=name,
            duration_s=duration_s,
            rng_seed=int(self.seed_spin.value()),
            use_ekf=bool(self.use_ekf_chk.isChecked()),
            attack_name=attack_name,
            attack_params=self._attack_params_from_form(),
            overrides=self._overrides_from_form(),
        )

    def _reset_form(self) -> None:
        self.name_edit.setText("baseline_gui")
        self.duration_edit.setText("60")
        self.seed_spin.setValue(42)
        self.use_ekf_chk.setChecked(True)
        self.attack_combo.setCurrentText("none")
        self.overrides_text.clear()
        self._rebuild_attack_param_form(self.attack_combo.currentText())

    def _load_payload_into_form(self, payload: dict[str, Any]) -> None:
        self.name_edit.setText(str(payload.get("name", "baseline_gui")))
        self.duration_edit.setText(str(payload.get("duration_s", 60)))
        self.seed_spin.setValue(int(payload.get("rng_seed", 42)))
        self.use_ekf_chk.setChecked(bool(payload.get("use_ekf", True)))

        attack_name = str(payload.get("attack_name", "none"))
        if self.attack_combo.findText(attack_name) < 0:
            self.attack_combo.addItem(attack_name)
        self.attack_combo.setCurrentText(attack_name)

        self._rebuild_attack_param_form(attack_name)
        attack_params = payload.get("attack_params", {}) or {}
        if isinstance(attack_params, dict):
            for key, widget in self._attack_param_widgets.items():
                if key not in attack_params:
                    continue
                value = attack_params[key]
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText("" if value is None else str(value))

        reserved = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
        overrides = {k: v for k, v in payload.items() if k not in reserved}
        self.overrides_text.setPlainText(json.dumps(overrides, indent=2) if overrides else "")

    def _show_form_error(self, exc: Exception) -> None:
        QMessageBox.critical(self, "Invalid scenario", str(exc))

    def _validate_queue_payload(self, payload: Any, *, source: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: scenario JSON must be an object")

        payload_copy = json.loads(json.dumps(payload))
        required = {"name", "duration_s", "rng_seed", "use_ekf", "attack_name", "attack_params"}
        missing = sorted(required - set(payload_copy.keys()))
        if missing:
            raise ValueError(f"{source}: missing required keys: {missing}")

        name = str(payload_copy.get("name", "")).strip()
        if not name:
            raise ValueError(f"{source}: name must be a non-empty string")
        payload_copy["name"] = name

        # Reuse backend parser/build logic for schema/type validation so GUI and CLI
        # reject the same malformed payloads.
        from sim.scenario_runner import _build_sim_config

        _build_sim_config(payload_copy)
        return payload_copy

    def _unique_queue_name(self, base_name: str, *, exclude_idx: int | None = None) -> str:
        base = str(base_name).strip() or "scenario"
        used: set[str] = set()
        for idx, payload in enumerate(self._queue):
            if exclude_idx is not None and idx == exclude_idx:
                continue
            used.add(str(payload.get("name", "")).strip())
        if base not in used:
            return base
        suffix = 2
        while True:
            candidate = f"{base}_{suffix}"
            if candidate not in used:
                return candidate
            suffix += 1

    # ---------------- queue actions ----------------
    def _queue_label(self, payload: dict[str, Any]) -> str:
        return (
            f"{payload.get('name', 'unnamed')} | "
            f"dur={payload.get('duration_s')}s | "
            f"seed={payload.get('rng_seed')} | "
            f"ekf={payload.get('use_ekf')} | "
            f"attack={payload.get('attack_name')}"
        )

    def _refresh_queue_list(self) -> None:
        self.queue_list.clear()
        for payload in self._queue:
            item = QListWidgetItem(self._queue_label(payload))
            self.queue_list.addItem(item)

    def _add_to_queue(self) -> None:
        try:
            payload = self._validate_queue_payload(self._draft_from_form().to_json_dict(), source="form")
        except Exception as exc:  # noqa: BLE001
            self._show_form_error(exc)
            return
        unique_name = self._unique_queue_name(str(payload.get("name", "scenario")))
        if unique_name != payload.get("name"):
            self._log(f"Renamed duplicate scenario '{payload.get('name')}' -> '{unique_name}'")
            payload["name"] = unique_name
        self._queue.append(payload)
        self._refresh_queue_list()
        self.queue_list.setCurrentRow(len(self._queue) - 1)
        self._log(f"Queued: {payload.get('name')}")

    def _update_selected_queue_item(self) -> None:
        idx = self.queue_list.currentRow()
        if idx < 0 or idx >= len(self._queue):
            self._show_error("No queue item selected")
            return
        try:
            payload = self._validate_queue_payload(self._draft_from_form().to_json_dict(), source=f"queue item #{idx + 1}")
        except Exception as exc:  # noqa: BLE001
            self._show_form_error(exc)
            return
        unique_name = self._unique_queue_name(str(payload.get("name", "scenario")), exclude_idx=idx)
        if unique_name != payload.get("name"):
            self._log(f"Renamed duplicate scenario '{payload.get('name')}' -> '{unique_name}'")
            payload["name"] = unique_name
        self._queue[idx] = payload
        self._refresh_queue_list()
        self.queue_list.setCurrentRow(idx)
        self._log(f"Updated queue item #{idx + 1}: {payload.get('name')}")

    def _load_selected_queue_item_into_form(self) -> None:
        idx = self.queue_list.currentRow()
        if idx < 0 or idx >= len(self._queue):
            return
        self._load_payload_into_form(self._queue[idx])

    def _remove_selected(self) -> None:
        idx = self.queue_list.currentRow()
        if idx < 0 or idx >= len(self._queue):
            self._show_error("No queue item selected")
            return
        removed = self._queue.pop(idx)
        self._refresh_queue_list()
        self._log(f"Removed: {removed.get('name')}")

    def _duplicate_selected(self) -> None:
        idx = self.queue_list.currentRow()
        if idx < 0 or idx >= len(self._queue):
            self._show_error("No queue item selected")
            return
        dup = json.loads(json.dumps(self._queue[idx]))
        dup["name"] = self._unique_queue_name(f"{dup.get('name', 'scenario')}_copy")
        self._queue.append(dup)
        self._refresh_queue_list()
        self.queue_list.setCurrentRow(len(self._queue) - 1)
        self._log(f"Duplicated: {dup.get('name')}")

    def _clear_queue(self) -> None:
        if not self._queue:
            return
        self._queue.clear()
        self._refresh_queue_list()
        self._log("Queue cleared.")

    # ---------------- JSON import/export ----------------
    def _save_form_as_json(self) -> None:
        try:
            payload = self._draft_from_form().to_json_dict()
        except Exception as exc:  # noqa: BLE001
            self._show_form_error(exc)
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scenario JSON",
            f"{payload.get('name', 'scenario')}.json",
            "JSON (*.json)",
        )
        if not path_str:
            return
        path = Path(path_str)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"Saved form JSON: {path}")

    def _load_json_into_form(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load Scenario JSON", "", "JSON (*.json)")
        if not path_str:
            return
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Scenario JSON must be an object")
            self._load_payload_into_form(payload)
        except Exception as exc:  # noqa: BLE001
            self._show_form_error(exc)
            return
        self._log(f"Loaded form JSON: {path}")

    def _import_jsons_to_queue(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Import Scenario JSONs", "", "JSON (*.json)")
        if not paths:
            return
        count = 0
        for p_str in paths:
            path = Path(p_str)
            try:
                payload = self._validate_queue_payload(
                    json.loads(path.read_text(encoding="utf-8")),
                    source=str(path),
                )
                unique_name = self._unique_queue_name(str(payload.get("name", "scenario")))
                if unique_name != payload.get("name"):
                    self._log(f"Import rename (duplicate): '{payload.get('name')}' -> '{unique_name}'")
                    payload["name"] = unique_name
                self._queue.append(payload)
                count += 1
            except Exception as exc:  # noqa: BLE001
                self._log(f"Import failed for {path}: {exc}")
        self._refresh_queue_list()
        self._log(f"Imported {count} scenario JSON(s) into queue.")

    def _export_selected_json(self) -> None:
        idx = self.queue_list.currentRow()
        if idx < 0 or idx >= len(self._queue):
            self._show_error("No queue item selected")
            return
        payload = self._queue[idx]
        default_name = f"{payload.get('name', 'scenario')}.json"
        path_str, _ = QFileDialog.getSaveFileName(self, "Export Selected Scenario", default_name, "JSON (*.json)")
        if not path_str:
            return
        path = Path(path_str)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log(f"Exported queue item to: {path}")

    # ---------------- runner actions ----------------
    def _browse_run_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select run root", self.run_root_edit.text())
        if folder:
            self.run_root_edit.setText(folder)

    def _set_running_state(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        for widget in [
            self.add_btn,
            self.update_btn,
            self.clear_form_btn,
            self.save_json_btn,
            self.load_json_btn,
            self.remove_btn,
            self.duplicate_btn,
            self.import_btn,
            self.export_btn,
            self.clear_queue_btn,
        ]:
            widget.setEnabled(not running)

    def _run_queue(self) -> None:
        if self._thread is not None:
            self._show_error("A run is already in progress")
            return
        if not self._queue:
            self._show_error("Queue is empty")
            return
        run_root = self.run_root_edit.text().strip()
        if not run_root:
            self._show_error("Run root is required")
            return
        preflight_queue: list[dict[str, Any]] = []
        try:
            for idx, payload in enumerate(self._queue, start=1):
                name = str(payload.get("name", "scenario")) if isinstance(payload, dict) else "scenario"
                preflight_queue.append(
                    self._validate_queue_payload(payload, source=f"queue item #{idx} ({name})")
                )
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Preflight validation failed: {exc}")
            return
        mc_n = int(self.mc_n_spin.value()) if self.mc_enable_chk.isChecked() else 0
        self._log("=" * 70)
        self._log(f"Starting run... (preflight OK for {len(preflight_queue)} scenario(s))")
        self._set_running_state(True)

        self._thread = QThread(self)
        self._worker = RunWorker(
            scenarios=preflight_queue,
            run_root=run_root,
            save_plots=bool(self.save_plots_chk.isChecked()),
            monte_carlo_n=mc_n,
            mc_seed_mode=self.mc_seed_mode_combo.currentText(),
            mc_seed_start=int(self.mc_seed_start_spin.value()),
            mc_seed_step=int(self.mc_seed_step_spin.value()),
            mc_per_run_plots=bool(self.mc_per_run_plots_chk.isChecked()),
            mc_aggregate_plots=bool(self.mc_agg_plots_chk.isChecked()),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _on_worker_finished(self, ok: bool, msg: str) -> None:
        self._log(msg)
        if ok:
            QMessageBox.information(self, "Scenario Runner", msg)
        else:
            QMessageBox.critical(self, "Scenario Runner", msg)

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._set_running_state(False)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.warning(
                self,
                "Run in progress",
                "A run is still in progress. Wait for completion before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    # ---------------- misc ----------------
    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Error", message)


def _coerce_scalar(raw: str) -> Any:
    s = raw.strip()
    lowered = s.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in s for ch in [".", "e", "E"]):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _slugify(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return out or "scenario"


def main() -> None:
    app = QApplication([])
    w = ScenarioRunnerGui()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
