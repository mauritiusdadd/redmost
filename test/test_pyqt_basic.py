#!/usr/bin/env python3
# Copyright (c) 2022-2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
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
"""
Redmost - PyQt basic UI and export tests.
"""


from pathlib import Path
import shutil

import pytest
from pytestqt.qtbot import QtBot
from pytestqt.qt_compat import qt_api

from test import TEST_DATA_PATH
from redmost import gui

TEST_DATA_DIR = Path(TEST_DATA_PATH)
TEST_FILE_95 = str(TEST_DATA_DIR / "spec_95.fits")
PROJECT_1_FILE = str(TEST_DATA_DIR / "test_project_1.json")
ZCAT_FILE = str(TEST_DATA_DIR / "test_zcat.fits")
SCREEN_OUT_DIR = TEST_DATA_DIR / "screenshots"

if not SCREEN_OUT_DIR.is_dir():
    SCREEN_OUT_DIR.mkdir(parents=True, exist_ok=True)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(counter=0)
def save_screen(qtbot: QtBot, widget: qt_api.QtWidgets.QWidget):
    input_file = Path(qtbot.screenshot(widget))
    dest_file_name = f"{input_file.stem}_{save_screen.counter:04d}{input_file.suffix}"
    shutil.copy(
        str(input_file),
        str(SCREEN_OUT_DIR / dest_file_name)
    )
    save_screen.counter += 1



@pytest.fixture
def main_app(qtbot: QtBot) -> gui.GuiApp:
    main_app = gui.GuiApp()
    setattr(main_app.main_wnd, "closeEvent", lambda x: None)
    qtbot.addWidget(main_app.main_wnd)
    qtbot.addWidget(main_app.about_wnd)
    qtbot.addWidget(main_app.msgBox)
    return main_app


def test_new_project(qtbot: QtBot, main_app: gui.GuiApp):
    """Test creating a new project from UI action."""
    main_app.main_wnd.action_new_project.trigger()
    save_screen(qtbot, main_app.main_wnd)

def test_load_project_save_project(qtbot: QtBot, main_app: gui.GuiApp):
    """Test loading and saving a project."""
    main_app.openProject(PROJECT_1_FILE)
    main_app.saveProject(PROJECT_1_FILE)

def test_load_project_import_zcat(qtbot: QtBot, main_app: gui.GuiApp):
    """Test importing zcat catalog and mapping columns."""
    main_app.openProject(PROJECT_1_FILE)
    main_app._doImportZcat(ZCAT_FILE)
    qtbot.wait_exposed(main_app.zcat_mapping_dialog)
    main_app.zcat_mapping_dialog.id_combo_box.setCurrentIndex(1)
    main_app.zcat_mapping_dialog.z_combo_box.setCurrentIndex(2)
    main_app.zcat_mapping_dialog.qf_combo_box.setCurrentIndex(4)
    save_screen(qtbot, main_app.zcat_mapping_dialog)
    main_app.zcat_mapping_dialog.accept()
    save_screen(qtbot, main_app.main_wnd)


def test_save_picture(qtbot: QtBot, main_app: gui.GuiApp):
    """Test spectrum plotting options and picture export."""
    main_app.importSpectra([TEST_FILE_95, ])
    main_app._safe_set_spec_index(0)

    main_app.main_wnd.smoothing_1d_check_box.setCheckState(
        qt_api.QtCore.Qt.CheckState.Checked
    )
    main_app.main_wnd.smoothing_1d_dspinbox.setValue(8.5)
    main_app.main_wnd.show_lines_check_box.setCheckState(
        qt_api.QtCore.Qt.CheckState.Checked
    )

    main_app.main_wnd.z_dspinbox.setValue(0.3998)

    save_screen(qtbot, main_app.main_wnd)
    out_pic = SCREEN_OUT_DIR / "spec_95.png"
    main_app.exportAsPicture(str(out_pic))
    assert out_pic.is_file()


def test_import_spectra_with_associated_2d(qtbot: QtBot, main_app: gui.GuiApp):
    """Test importing 1D spectra with associated 2D spectra."""
    # Test 1: R2129 with SPEC_2D card in header
    r2129_1d = str(TEST_DATA_DIR / "R2129_LRb_p1_M1_Q1_10_1_1d.fits")
    main_app.importSpectra([r2129_1d])

    # 1D child + 2D parent = 2 spectra in open_spectra
    assert len(main_app.open_spectra) == 2
    assert main_app.main_wnd.spec_list_widget.topLevelItemCount() == 1
    parent_item = main_app.main_wnd.spec_list_widget.topLevelItem(0)
    assert parent_item is not None
    assert parent_item.text(0).startswith('[2D]')
    assert parent_item.childCount() == 1

    child_item = parent_item.child(0)
    assert child_item is not None
    assert child_item.text(0).startswith('[1D]')
    child_uuid = child_item.data(0, qt_api.QtCore.Qt.ItemDataRole.UserRole)
    sp_1d = main_app.open_spectra[child_uuid]
    assert 'linked' in sp_1d.meta
    assert '2D' in sp_1d.meta['linked']

    # Clear open spectra
    main_app.open_spectra.clear()
    main_app.open_spectra_items.clear()
    main_app.main_wnd.spec_list_widget.clear()

    # Test 2: M0416 with ESO Phase 3 ASSOC1 / ASSON1 cards
    m0416_1d = str(TEST_DATA_DIR / "M0416_LRb_p1_M1_Q1_100_1_1d.fits")
    main_app.importSpectra([m0416_1d])

    assert len(main_app.open_spectra) == 2
    assert main_app.main_wnd.spec_list_widget.topLevelItemCount() == 1
    parent_m = main_app.main_wnd.spec_list_widget.topLevelItem(0)
    assert parent_m is not None
    assert parent_m.text(0).startswith('[2D]')
    assert parent_m.childCount() == 1

    child_m = parent_m.child(0)
    assert child_m is not None
    assert child_m.text(0).startswith('[1D]')
    child_uuid_m = child_m.data(0, qt_api.QtCore.Qt.ItemDataRole.UserRole)
    sp_1d_m = main_app.open_spectra[child_uuid_m]
    assert 'linked' in sp_1d_m.meta
    assert '2D' in sp_1d_m.meta['linked']

    # Test 3: Verify dummy spectra are excluded when preparing for backends
    for_backend = {
        u: sp
        for u, sp in main_app.open_spectra.items()
        if not sp.meta.get('is_dummy', False)
    }
    assert len(for_backend) == 1
    assert child_uuid_m in for_backend


