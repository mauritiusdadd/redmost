# Copyright (c) 2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
#
# This file is part of Redmost.
#
# Redmost is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Redmost is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Redmost.  If not, see <https://www.gnu.org/licenses/>.

"""Redrock backend plugin providing UI components and execution handling."""

from __future__ import annotations

import os
import sys
import uuid
import pickle
from pathlib import Path
import traceback
import numpy as np

from typing import Optional, Any

from redmost.backends.base import BackendPlugin
from . import rrock as rrock_backend
from redmost.qt_compat import qt_api

QtWidgets = qt_api.QtWidgets
QtCore = qt_api.QtCore


class RedrockPlugin(BackendPlugin):
    """Plugin implementation for the Redrock redshift estimation backend."""

    backend_id: str = "rrock"
    backend_name: str = "Redrock"

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        """Initialize the Redrock plugin instance."""
        super().__init__(parent)
        self.qapp = qt_api.QtWidgets.QApplication.instance()
        self._target_dump_file = "rr.targets"
        self._result_dump_file = "rr.result"
        self._t_dict: dict[str, uuid.UUID] = {}
        self._stderr_buffer: list[str] = []
        self._total_progress = 0
        self._total_maximum = 0

        # Subprocess handler
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self._process_output)
        self.process.readyReadStandardError.connect(self._process_error)
        self.process.finished.connect(self._on_process_finished)

        # Main UI widgets
        self._main_widget: Optional[QtWidgets.QWidget] = None
        self.current_radio: Optional[QtWidgets.QRadioButton] = None
        self.selected_radio: Optional[QtWidgets.QRadioButton] = None
        self.all_radio: Optional[QtWidgets.QRadioButton] = None
        self.log_text_edit: Optional[QtWidgets.QTextEdit] = None
        self.run_button: Optional[QtWidgets.QPushButton] = None
        self.templates_table_widget: Optional[QtWidgets.QTableWidget] = None
        self.preview_button: Optional[QtWidgets.QPushButton] = None
        self.use_z_button: Optional[QtWidgets.QPushButton] = None

        # Redrock results store: target_uuid -> {template_name: dict_with_z_and_info}
        self.results_per_object: dict[uuid.UUID, dict[str, Any]] = {}

        # Settings UI widgets
        self._settings_widget: Optional[QtWidgets.QWidget] = None
        self.template_dir_edit: Optional[QtWidgets.QLineEdit] = None
        self.select_dir_button: Optional[QtWidgets.QPushButton] = None
        self.template_files_list: Optional[QtWidgets.QListWidget] = None

    def is_available(self) -> bool:
        """Return True if the redrock Python module is available."""
        return rrock_backend.HAS_REDROCK

    def get_run_button(self) -> Optional[QtWidgets.QPushButton]:
        """Return the main Run/Cancel button."""
        return self.run_button

    def is_running(self) -> bool:
        """Return True if the redrock subprocess is currently running."""
        return (
            self.process.state()
            != QtCore.QProcess.ProcessState.NotRunning
        )

    def clear(self) -> None:
        """Clear cached redrock results."""
        self.results_per_object.clear()
        if self.templates_table_widget is not None:
            self.templates_table_widget.setRowCount(0)

    def cancel(self) -> None:
        """Kill the running redrock subprocess."""
        if self.is_running():
            self.process.kill()
        self.restore_controls()
        self.cancelled.emit()

    def disable_controls(self) -> None:
        """Disable options in the main widget while a job is running."""
        if self.current_radio is not None:
            self.current_radio.setEnabled(False)
        if self.selected_radio is not None:
            self.selected_radio.setEnabled(False)
        if self.all_radio is not None:
            self.all_radio.setEnabled(False)
        if self.run_button is not None:
            self.run_button.setText(
                self.qapp.tr("Cancel") if self.qapp else "Cancel"
            )

    def restore_controls(self) -> None:
        """Restore options in the main widget after job completion."""
        if self.current_radio is not None:
            self.current_radio.setEnabled(True)
        if self.selected_radio is not None:
            self.selected_radio.setEnabled(True)
        if self.all_radio is not None:
            self.all_radio.setEnabled(True)
        if self.run_button is not None:
            self.run_button.setText(
                self.qapp.tr("Run") if self.qapp else "Run"
            )

    def get_main_widget(
        self, parent: Optional[QtWidgets.QWidget] = None
    ) -> QtWidgets.QWidget:
        """Load or return the main widget containing Redrock controls."""
        if self._main_widget is not None:
            return self._main_widget

        ui_path = Path(__file__).resolve().parent / "rrock_tab.ui"
        widget = qt_api.loadUiWidget(str(ui_path), parent)

        self.current_radio = getattr(widget, "redrock_current_radio", None)
        self.selected_radio = getattr(widget, "redrock_selected_radio", None)
        self.all_radio = getattr(widget, "redrock_all_radio", None)
        self.log_text_edit = getattr(widget, "redrock_text_edit", None)
        self.run_button = getattr(widget, "redrock_run_button", None)
        self.templates_table_widget = getattr(
            widget, "redrock_templates_table_widget", None
        )
        self.preview_button = getattr(widget, "redrock_preview_button", None)
        self.use_z_button = getattr(widget, "redrock_use_z_button", None)

        if not self.is_available():
            if self.run_button is not None:
                self.run_button.setEnabled(False)
            if self.log_text_edit is not None:
                self.log_text_edit.setText(
                    "Cannot load redrock python module.\n"
                    "Please check if redrock is correctly installed!\n"
                )

        self._main_widget = widget
        return widget

    def run(self, spectra: dict[uuid.UUID, Any], **kwargs: Any) -> None:
        """Build targets and start Redrock execution in a subprocess."""
        if not self.is_available():
            return

        self._stderr_buffer.clear()
        if self.log_text_edit is not None:
            self.log_text_edit.clear()

        self.started.emit()
        self.maximum.emit(len(spectra))
        self.message.emit(
            self.qapp.tr("Building targets")
            if self.qapp else "Building targets"
        )

        targets, self._t_dict = rrock_backend.build_redrock_targets(
            spectra, progress_callback=self.progress.emit
        )

        rrock_backend.dump_targets(targets, self._target_dump_file)

        self._total_maximum = 22
        self._total_progress = 0
        self.maximum.emit(self._total_maximum)
        self.message.emit(
            self.qapp.tr("Running redrock backend")
            if self.qapp else "Running redrock backend"
        )

        env = QtCore.QProcessEnvironment.systemEnvironment()
        src_path = str(
            Path(__file__).resolve().parent.parent.parent.parent
        )
        existing_ppath = env.value("PYTHONPATH", "")

        if existing_ppath:
            env.insert("PYTHONPATH", f"{src_path}{os.pathsep}{existing_ppath}")
        else:
            env.insert("PYTHONPATH", src_path)
        self.process.setProcessEnvironment(env)

        self.process.start(
            sys.executable,
            [
                rrock_backend.__file__,
                self._target_dump_file,
                self._result_dump_file,
            ],
        )

    def _process_output(self) -> None:
        data_bytes = self.process.readAllStandardOutput()
        text = bytes(data_bytes).decode("utf-8", errors="replace")
        for line in text.splitlines():
            simplified = " ".join(line.lower().strip().split())
            if "finished in" in simplified:
                self._total_progress += 1
                self.progress.emit(self._total_progress)
            elif "computing redshifts took:" in simplified:
                self.progress.emit(self._total_maximum)

        if self.log_text_edit is not None:
            self.log_text_edit.append(text.rstrip())
        self.message.emit(text)

    def _process_error(self) -> None:
        data_bytes = self.process.readAllStandardError()
        text = bytes(data_bytes).decode("utf-8", errors="replace")
        if text:
            self._stderr_buffer.append(text)
            if self.log_text_edit is not None:
                self.log_text_edit.append(text.rstrip())
            self.message.emit(text)

    def _on_process_finished(self, exit_status: int) -> None:
        rem_err_bytes = self.process.readAllStandardError()
        rem_err = bytes(rem_err_bytes).decode("utf-8", errors="replace")
        if rem_err:
            self._stderr_buffer.append(rem_err)
            if self.log_text_edit is not None:
                self.log_text_edit.append(rem_err.rstrip())
            self.message.emit(rem_err)

        results = None
        if exit_status != 0:
            err_details = "".join(self._stderr_buffer).strip()
            msg = (
                f"\n*** REDROCK EXECUTION FAILED "
                f"(exit code {exit_status}) ***\n{err_details}\n"
            )
            if self.log_text_edit is not None:
                self.log_text_edit.append(msg)
            self.message.emit(msg)
        else:
            try:
                with open(self._result_dump_file, "rb") as f:
                    results = pickle.load(f)
            except Exception:
                tb = traceback.format_exc()
                msg = (
                    f"\n*** REDROCK EXECUTION FAILED: "
                    f"failed to load results ***\n{tb}\n"
                )
                if self.log_text_edit is not None:
                    self.log_text_edit.append(msg)
                self.message.emit(msg)
            else:
                if results is not None:
                    scandata, zfit = results
                    for target_id, target_uuid in self._t_dict.items():
                        obj_dict: dict[str, Any] = {}
                        if scandata is not None and target_id in scandata:
                            for tname, val in scandata[target_id].items():
                                z_grid = val['redshifts']
                                tot_chi2 = val['zchi2'] + val['penalty']
                                idx = int(tot_chi2.argmin())
                                best_z = float(z_grid[idx])
                                best_coeff = np.array(val['zcoeff'][idx])
                                parts = tname.split(':::')
                                spectype = parts[0]
                                subtype = parts[1] if len(parts) > 1 else ''
                                obj_dict[tname] = {
                                    'name': tname,
                                    'z': best_z,
                                    'chi2': float(tot_chi2[idx]),
                                    'coeff': best_coeff,
                                    'spectype': spectype,
                                    'subtype': subtype,
                                    'redshifts': np.array(
                                        z_grid, dtype=float
                                    ),
                                    'chi2_curve': np.array(
                                        tot_chi2, dtype=float
                                    ),
                                }
                        elif zfit is not None:
                            t_rows = zfit[zfit['targetid'] == target_id]
                            for row in t_rows:
                                tname = row['spectype'] + (
                                    ':::' + row['subtype'] if row['subtype']
                                    else ''
                                )
                                obj_dict[tname] = {
                                    'name': tname,
                                    'z': float(row['z']),
                                    'chi2': float(row['chi2']),
                                    'coeff': np.array(row['coeff']),
                                    'spectype': row['spectype'],
                                    'subtype': row['subtype'],
                                }
                        self.results_per_object[target_uuid] = obj_dict

        self._stderr_buffer.clear()
        self.restore_controls()
        self.finished.emit((results, self._t_dict))

    def get_settings_widget(
        self, parent: Optional[QtWidgets.QWidget] = None
    ) -> Optional[QtWidgets.QWidget]:
        """Load and return the settings widget for Redrock options."""
        if self._settings_widget is not None:
            return self._settings_widget

        ui_path = Path(__file__).resolve().parent / "rrock_settings_tab.ui"
        widget = qt_api.loadUiWidget(str(ui_path), parent)

        self.select_dir_button = getattr(
            widget, "select_template_dir_rrock", None
        )
        self.template_dir_edit = getattr(
            widget, "linet_edit_template_dir_rrock", None
        )
        self.template_files_list = getattr(
            widget, "template_files_list_widget_rrock", None
        )

        if self.select_dir_button is not None:
            self.select_dir_button.clicked.connect(
                self._on_choose_template_dir
            )
        if self.template_dir_edit is not None:
            self.template_dir_edit.textChanged.connect(
                self._update_template_files_list
            )

        if not self.is_available():
            widget.setEnabled(False)

        self._settings_widget = widget
        return widget

    def get_template_dir(self) -> str:
        """Return the currently configured Redrock template directory."""
        if self.template_dir_edit is not None:
            return self.template_dir_edit.text()
        return os.environ.get("RR_TEMPLATE_DIR", "")

    def set_template_dir(self, dir_path: str) -> None:
        """Set the Redrock template directory and update environment."""
        if not dir_path:
            return
        resolved = str(Path(dir_path).resolve())
        if self.template_dir_edit is not None:
            self.template_dir_edit.setText(resolved)
        os.environ["RR_TEMPLATE_DIR"] = resolved
        self._update_template_files_list(resolved)

    def _update_template_files_list(self, dir_path: str = "") -> None:
        """Refresh the files listed in the template list widget."""
        if self.template_files_list is None:
            return

        if not dir_path:
            if self.template_dir_edit is not None:
                dir_path = self.template_dir_edit.text()

        self.template_files_list.clear()
        if not dir_path:
            return

        resolved = Path(dir_path)
        if not resolved.is_dir():
            return

        for entry in sorted(resolved.iterdir()):
            if entry.is_file():
                self.template_files_list.addItem(entry.name)


    def _on_choose_template_dir(self) -> None:
        """Handle template folder selection dialog and validation."""
        if not self.is_available():
            return

        from redrock.templates import find_templates

        parent = self._settings_widget
        url = QtWidgets.QFileDialog.getExistingDirectoryUrl(
            parent,
            self.qapp.tr("Select Redrock templates directory")
            if self.qapp else "Select Redrock templates directory",
        )
        if not url.isValid():
            return

        selected_dir = url.path()
        try:
            tmpl_files = list(find_templates(selected_dir))
        except Exception:
            tmpl_files = []

        if tmpl_files:
            self.set_template_dir(selected_dir)
            return

        # Prompt download if no templates found
        msg = QtWidgets.QMessageBox(parent=parent)
        msg.setWindowTitle(
            self.qapp.tr("No templates found")
            if self.qapp else "No templates found"
        )
        msg.setText(
            (
                self.qapp.tr(
                    "No valid Redrock templates were found in the selected "
                    "folder:\n{}\n\nWould you like to download the templates "
                    "into that folder?"
                ) if self.qapp else (
                    "No valid Redrock templates were found in:\n{}\n\n"
                    "Download them?"
                )
            ).format(selected_dir)
        )
        msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)

        btn_dl = msg.addButton(
            self.qapp.tr("Download") if self.qapp else "Download",
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        msg.addButton(
            self.qapp.tr("Cancel") if self.qapp else "Cancel",
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        msg.setDefaultButton(btn_dl)
        msg.exec()

        if msg.clickedButton() != btn_dl:
            return

        clone_dir = str(Path(selected_dir) / "redrock-templates")
        if self._download_templates(clone_dir, parent=parent):
            self.set_template_dir(clone_dir)

    def _download_templates(
        self, dest_dir: str, parent: Optional[QtWidgets.QWidget] = None
    ) -> bool:
        """Download redrock templates repository via git subprocess."""
        from redmost.gui import ProcessOutputDialog

        repo_url = "https://github.com/desihub/redrock-templates"
        command = [
            "git", "clone", "--progress", "--depth", "1",
            repo_url, dest_dir
        ]
        title = (
            self.qapp.tr("Downloading Redrock templates")
            if self.qapp else "Downloading Redrock templates"
        )
        dlg = ProcessOutputDialog(command=command, title=title, parent=parent)
        result = dlg.exec()

        if result != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        return dlg.exit_code() == 0

    def check_templates_on_startup(
        self, parent_window: Optional[QtWidgets.QWidget] = None
    ) -> None:
        """Check template availability and prompt the user if missing."""
        if not self.is_available():
            return

        from redrock.templates import find_templates

        configured_dir = self.get_template_dir()
        if configured_dir and Path(configured_dir).is_dir():
            try:
                tmpl_files = list(find_templates(configured_dir))
            except Exception:
                tmpl_files = []
            if tmpl_files:
                os.environ["RR_TEMPLATE_DIR"] = configured_dir
                self._update_template_files_list(configured_dir)
                return

        try:
            tmpl_files = list(find_templates(None))
        except Exception:
            tmpl_files = []

        if tmpl_files:
            self._update_template_files_list()
            return

        # Prompt user
        msg = QtWidgets.QMessageBox(parent=parent_window)
        msg.setWindowTitle(
            self.qapp.tr("Redrock templates not found")
            if self.qapp else "Redrock templates not found"
        )
        msg.setText(
            self.qapp.tr(
                "Redrock templates could not be found.\n\n"
                "What would you like to do?"
            ) if self.qapp else "Redrock templates could not be found."
        )
        msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)

        btn_download = msg.addButton(
            self.qapp.tr("Download") if self.qapp else "Download",
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        btn_select = msg.addButton(
            self.qapp.tr("Select folder") if self.qapp else "Select folder",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        msg.addButton(
            self.qapp.tr("Ignore") if self.qapp else "Ignore",
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        msg.setDefaultButton(btn_download)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_download:
            dest_url = QtWidgets.QFileDialog.getExistingDirectoryUrl(
                parent_window,
                self.qapp.tr("Select destination folder for templates")
                if self.qapp else "Select destination folder",
            )
            if not dest_url.isValid():
                return
            dest_dir = dest_url.path()
            clone_dir = str(Path(dest_dir) / "redrock-templates")
            if self._download_templates(clone_dir, parent=parent_window):
                self.set_template_dir(clone_dir)

        elif clicked == btn_select:
            url = QtWidgets.QFileDialog.getExistingDirectoryUrl(
                parent_window,
                self.qapp.tr("Select Redrock templates directory")
                if self.qapp else "Select Redrock templates directory",
            )
            if not url.isValid():
                return
            self.set_template_dir(url.path())

    def load_settings(self, settings: QtCore.QSettings) -> None:
        """Load Redrock settings from persistent storage."""
        template_dir = settings.value("redrock/template_dir", type=str)
        if not template_dir:
            template_dir = settings.value("rrock_template_dir", type=str)
        if template_dir:
            self.set_template_dir(template_dir)

    def apply_settings(self, settings: QtCore.QSettings) -> None:
        """Save Redrock settings to persistent storage."""
        template_dir = self.get_template_dir()
        if template_dir:
            settings.setValue("redrock/template_dir", template_dir)

    def setup_gui(self, app: Any) -> None:
        """Register Redrock tabs and connect signals to GuiApp."""
        app.redrock_plugin = self

        rr_widget = self.get_main_widget(app.main_wnd.tab_widget)
        app.main_wnd.tab_widget.insertTab(
            1, rr_widget, self.backend_name
        )

        rr_settings = self.get_settings_widget(
            app.settings_wnd.settings_tab_widget
        )
        if rr_settings is not None:
            app.settings_wnd.settings_tab_widget.addTab(
                rr_settings, self.backend_name
            )

        self.started.connect(app._on_redrock_started)
        self.finished.connect(app._on_redrock_finished)
        self.cancelled.connect(app._on_redrock_cancelled)
        self.progress.connect(app.pbar.setValue)
        self.maximum.connect(app.pbar.setMaximum)
        self.message.connect(app.statusbar.showMessage)

        rr_btn = self.get_run_button()
        if rr_btn is not None:
            rr_btn.clicked.connect(app.doStartRedrock)

        if self.templates_table_widget is not None:
            self.templates_table_widget.itemDoubleClicked.connect(
                app.setCurrentObjectRedshiftFromRedrock
            )
            self.templates_table_widget.itemDoubleClicked.connect(
                app.plotRedrockTemplate
            )
            self.templates_table_widget.itemDoubleClicked.connect(
                app.plotRedrockChi2Curve
            )
            self.templates_table_widget.itemClicked.connect(
                app.plotRedrockChi2Curve
            )

        if self.use_z_button is not None:
            self.use_z_button.clicked.connect(
                app.doSetCurrentObjectRedshiftFromRedrock
            )
            self.use_z_button.clicked.connect(
                app.doPlotRedrockTemplate
            )
            self.use_z_button.clicked.connect(
                app.doPlotRedrockChi2Curve
            )

        if self.preview_button is not None:
            self.preview_button.clicked.connect(
                app.doPreviewRedrockTemplate
            )
