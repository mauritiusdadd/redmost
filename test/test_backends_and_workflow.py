# Copyright (c) 2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Unit tests for Redmost backend plugins, projects, masks, and spectra."""

from __future__ import annotations

import glob
import os
import uuid
import pytest
from pytestqt.qtbot import QtBot
from pytestqt.qt_compat import qt_api

from test import TEST_DATA_PATH
from redmost import gui
from redmost.backends import xcorr, HAS_XCORR


XCORR_TEMPLATES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "src", "redmost", "spectral_templates"
    )
)
REDROCK_CLONE_DEST = os.path.join(TEST_DATA_PATH, "redrock-templates")
PROJECT_3_PATH = os.path.join(TEST_DATA_PATH, "test_project_3.json")
TIMEOUT_MS = 60000
DOWNLOAD_TIMEOUT_MS = 180000


@pytest.fixture(scope="session", autouse=True)
def setup_redrock_templates():
    """Ensure redrock templates exist in test/data for the test suite."""
    if not os.path.isdir(REDROCK_CLONE_DEST):
        qapp = (
            qt_api.QtWidgets.QApplication.instance()
            or qt_api.QtWidgets.QApplication([])
        )
        plugin = gui.backends.RedrockPlugin()

        # Connect a polling timer to click OK on ProcessOutputDialog
        timer = qt_api.QtCore.QTimer()

        def _auto_accept():
            for widget in qapp.topLevelWidgets():
                if isinstance(widget, gui.ProcessOutputDialog):
                    if widget._ok_button.isEnabled():
                        widget._ok_button.click()
                        timer.stop()

        timer.timeout.connect(_auto_accept)
        timer.start(100)

        # Download via the app method with a 3-minute timeout
        success = plugin._download_templates(REDROCK_CLONE_DEST)
        timer.stop()
        assert success, "Failed to download redrock templates via app method"
    yield
    if os.path.islink(REDROCK_CLONE_DEST):
        try:
            os.unlink(REDROCK_CLONE_DEST)
        except OSError:
            pass


@pytest.fixture
def app(qtbot: QtBot) -> gui.GuiApp:
    """Create a headless GuiApp fixture."""
    main_app = gui.GuiApp()
    setattr(main_app.main_wnd, "closeEvent", lambda event: None)
    qtbot.addWidget(main_app.main_wnd)
    qtbot.addWidget(main_app.settings_wnd)
    return main_app


def _select_only_three_templates(app: gui.GuiApp) -> int:
    """Select only the three specified templates in the xcorr tree widget."""
    targets = {"etg_muse.dat", "NEW_E_extended.dat", "NEW_Im_extended.dat"}
    count = 0
    for item in app.xcor_tree_leaf_items:
        if item.text(0) in targets:
            item.setCheckState(0, qt_api.QtCore.Qt.CheckState.Checked)
            count += 1
        else:
            item.setCheckState(0, qt_api.QtCore.Qt.CheckState.Unchecked)
    return count


def test_template_loading(app: gui.GuiApp):
    """Verify loading of both X-Corr and Redrock spectral templates."""
    # Test X-Corr template loading
    if xcorr is not None:
        xcorr_templates = xcorr.load_templates(XCORR_TEMPLATES_DIR)
        assert len(xcorr_templates) > 0, "No X-Corr templates loaded"
        loaded_filenames = [
            os.path.basename(k) for k in xcorr_templates.keys()
        ]
        assert "etg_muse.dat" in loaded_filenames
        assert "NEW_E_extended.dat" in loaded_filenames
        assert "NEW_Im_extended.dat" in loaded_filenames


    # Test Redrock template detection
    assert os.path.isdir(REDROCK_CLONE_DEST)
    app.redrock_plugin.set_template_dir(REDROCK_CLONE_DEST)
    assert os.path.realpath(app.redrock_plugin.get_template_dir()) == (
        os.path.realpath(REDROCK_CLONE_DEST)
    )
    assert app.redrock_plugin.template_files_list is not None
    assert app.redrock_plugin.template_files_list.count() > 0


def test_import_spectra(app: gui.GuiApp):
    """Verify importing spec_*.fits files located in test/data."""
    spec_files = sorted(glob.glob(os.path.join(TEST_DATA_PATH, "spec_*.fits")))
    assert len(spec_files) > 0, "No spec_*.fits files found in test/data"

    app.importSpectra(spec_files)
    assert len(app.open_spectra) == len(spec_files)
    assert app.main_wnd.spec_list_widget.count() == len(spec_files)


def test_add_spectral_masks(app: gui.GuiApp):
    """Verify adding, configuring, and deleting spectral masks."""
    spec_files = [os.path.join(TEST_DATA_PATH, "spec_95.fits")]
    app.importSpectra(spec_files)
    assert len(app.open_spectra) == 1
    spec_uuid = list(app.open_spectra.keys())[0]

    mask_props = {
        "name": "telluric_band",
        "w_min": 7590.0,
        "w_max": 7700.0,
        "enabled": True,
        "global": False,
        "comment": "Oxygen A band mask",
    }
    mask_id = app.addMask(spec_uuid, mask_props)
    assert isinstance(mask_id, uuid.UUID)
    assert spec_uuid in app.spectral_masks
    assert mask_id in app.spectral_masks[spec_uuid]
    assert app.spectral_masks[spec_uuid][mask_id]["name"] == "telluric_band"

    # Test mask deletion
    app.deleteMask(spec_uuid, mask_id)
    assert mask_id not in app.spectral_masks[spec_uuid]


@pytest.mark.parametrize(
    "project_name",
    [
        "test_project_1.json",
        "test_project_2.json",
        "test_project_3.json",
        "test_project_4.json",
        "test_project_5.json",
    ]
)
def test_load_projects(app: gui.GuiApp, project_name: str):
    """Verify loading test projects."""
    proj_path = os.path.join(TEST_DATA_PATH, project_name)
    assert os.path.isfile(proj_path), f"Project file not found: {proj_path}"
    app.openProject(proj_path)
    assert len(app.open_spectra) > 0
    assert app.main_wnd.spec_list_widget.count() == len(app.open_spectra)


def test_xcorr_execution_cpu_and_gpu(qtbot: QtBot, app: gui.GuiApp):
    """Run X-Corr on all, selected, and single spectrum (CPU and GPU)."""
    if not hasattr(app, "xcorr_plugin") or app.xcorr_plugin is None:
        pytest.skip("X-Corr backend is not installed.")

    app.openProject(PROJECT_3_PATH)

    total_specs = len(app.open_spectra)
    assert total_specs >= 4

    # Select the 3 templates requested
    n_sel = _select_only_three_templates(app)
    assert n_sel == 3
    selected_templates = app.getXcorrSelectedTemplates()
    assert len(selected_templates) == 3

    # Configure scan settings: range 0.0 - 1.0, step 0.01
    app.main_wnd.xcorr_z_min_dspinbox.setValue(0.0)
    app.main_wnd.xcorr_z_max_dspinbox.setValue(1.0)
    app.main_wnd.xcorr_z_step_dspinbox.setValue(0.01)

    modes = ["all", "selected", "single"]
    for mode in modes:
        if mode == "all":
            app.main_wnd.xcorr_all_radio.setChecked(True)
            expected_count = total_specs
        elif mode == "selected":
            app.main_wnd.xcorr_selected_radio.setChecked(True)
            # Select 2 spectra in spec_list_widget
            for row in range(app.main_wnd.spec_list_widget.count()):
                item = app.main_wnd.spec_list_widget.item(row)
                if item is not None:
                    state = (
                        qt_api.QtCore.Qt.CheckState.Checked
                        if row in (0, 1)
                        else qt_api.QtCore.Qt.CheckState.Unchecked
                    )
                    item.setCheckState(state)
            expected_count = 2
        else:
            app.main_wnd.xcorr_current_radio.setChecked(True)
            app._safe_set_spec_index(0)
            expected_count = 1

        # Test CPU execution
        app.xcorr_plugin.clear()
        app.main_wnd.xcorr_gpu_checkbox.setChecked(False)
        with qtbot.waitSignal(
            app.xcorr_plugin.finished, timeout=TIMEOUT_MS
        ):
            app.doStartXCorr()
        assert len(app.xcorr_plugin.best_z) == expected_count

        # Test GPU execution if available
        if app.main_wnd.xcorr_gpu_checkbox.isEnabled():
            app.xcorr_plugin.clear()
            app.main_wnd.xcorr_gpu_checkbox.setChecked(True)
            with qtbot.waitSignal(
                app.xcorr_plugin.finished, timeout=TIMEOUT_MS
            ):
                app.doStartXCorr()
            assert len(app.xcorr_plugin.best_z) == expected_count


def test_redrock_execution(qtbot: QtBot, app: gui.GuiApp):
    """Run Redrock on all, selected, and single spectrum."""
    if not app.redrock_plugin.is_available():
        pytest.skip("Redrock module is not available in the environment")

    app.openProject(PROJECT_3_PATH)
    total_specs = len(app.open_spectra)
    assert total_specs >= 4

    app.redrock_plugin.set_template_dir(REDROCK_CLONE_DEST)

    modes = ["all", "selected", "single"]
    for mode in modes:
        if mode == "all":
            app.redrock_plugin.all_radio.setChecked(True)
            expected_count = total_specs
        elif mode == "selected":
            app.redrock_plugin.selected_radio.setChecked(True)
            for row in range(app.main_wnd.spec_list_widget.count()):
                item = app.main_wnd.spec_list_widget.item(row)
                if item is not None:
                    state = (
                        qt_api.QtCore.Qt.CheckState.Checked
                        if row in (0, 1)
                        else qt_api.QtCore.Qt.CheckState.Unchecked
                    )
                    item.setCheckState(state)
            expected_count = 2
        else:
            app.redrock_plugin.current_radio.setChecked(True)
            app._safe_set_spec_index(0)
            expected_count = 1

        with qtbot.waitSignal(
            app.redrock_plugin.finished, timeout=TIMEOUT_MS
        ) as blocker:
            app.doStartRedrock()

        # Check emitted results
        results, t_dict = blocker.args[0]
        assert results is not None
        assert len(t_dict) == expected_count
