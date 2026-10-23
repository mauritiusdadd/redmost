#!/usr/bin/env python3
"""
Redmost - PyQt basic UI and export tests.

Copyright (C) 2022-2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
"""

import os
import shutil

import pytest
from pytestqt.qtbot import QtBot
from pytestqt.qt_compat import qt_api

from test import TEST_DATA_PATH
from redmost import gui

TEST_FILE_95 = os.path.join(TEST_DATA_PATH, "spec_95.fits")
PROJECT_1_FILE = os.path.join(TEST_DATA_PATH, "test_project_1.json")
ZCAT_FILE = os.path.join(TEST_DATA_PATH, "test_zcat.fits")
SCREEN_OUT_DIR = os.path.join(TEST_DATA_PATH, "screenshots")

if not os.path.isdir(SCREEN_OUT_DIR):
    os.makedirs(SCREEN_OUT_DIR)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(counter=0)
def save_screen(qtbot: QtBot, widget: qt_api.QtWidgets.QWidget):
    input_file = qtbot.screenshot(widget)
    file, ext = os.path.splitext(os.path.basename(input_file))
    dest_file_name = f"{file}_{save_screen.counter:04d}{ext}"
    shutil.copy(
        input_file,
        os.path.join(SCREEN_OUT_DIR, dest_file_name)
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
    out_pic = os.path.join(SCREEN_OUT_DIR, "spec_95.png")
    main_app.exportAsPicture(out_pic)
    assert os.path.isfile(out_pic)
