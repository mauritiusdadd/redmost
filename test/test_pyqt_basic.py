#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPEX - SPectra EXtractor.

Extract spectra from spectral data cubes.

Copyright (C) 2022  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""
from __future__ import absolute_import, division, print_function

import os
import time
import shutil

import pytest
from pytestqt.qtbot import QtBot
from pytestqt.qt_compat import qt_api

from test import TEST_DATA_PATH
from redmost import gui

Z_FTOL = 0.01

PROJECT_1_FILE = os.path.join(TEST_DATA_PATH, 'test_project_1.json')
PROJECT_2_FILE = os.path.join(TEST_DATA_PATH, 'test_project_2.json')
ZCAT_FILE = os.path.join(TEST_DATA_PATH, 'test_zcat.fits')
SCREEN_OUT_DIR = os.path.join(TEST_DATA_PATH, 'screenshots')

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


def handle_message_box(button_type, qtbot: QtBot, main_app: gui.GuiApp):
    qtbot.wait_exposed(main_app.msgBox, timeout=1000)
    button = main_app.msgBox.button(button_type)
    shutil.copy(
        qtbot.screenshot(main_app.msgBox),
        SCREEN_OUT_DIR
    )
    assert button is not None
    qtbot.mouseClick(
        button,
        qt_api.QtCore.Qt.MouseButton.LeftButton,
        delay=1
    )

@pytest.fixture
def main_app(qtbot: QtBot) -> gui.GuiApp:
    print("Creating main window")
    main_app = gui.GuiApp()
    setattr(main_app.main_wnd, "closeEvent", lambda x: None)
    qtbot.addWidget(main_app.main_wnd)
    qtbot.addWidget(main_app.about_wnd)
    qtbot.addWidget(main_app.msgBox)
    return main_app


def test_new_project(qtbot: QtBot, main_app: gui.GuiApp):
    print("Creating a new project")
    main_app.main_wnd.action_new_project.trigger()
    save_screen(qtbot, main_app.main_wnd)


def test_load_project(qtbot: QtBot, main_app: gui.GuiApp):
    print("Loading a project")
    main_app.openProject(PROJECT_2_FILE)
    save_screen(qtbot, main_app.main_wnd)


def test_load_project_import_zcat(qtbot: QtBot, main_app: gui.GuiApp):
    print("Loading a project")
    main_app.openProject(PROJECT_1_FILE)
    main_app._doImportZcat(ZCAT_FILE)
    qtbot.wait_exposed(main_app.zcat_mapping_dialog)
    main_app.zcat_mapping_dialog.id_combo_box.setCurrentIndex(1)
    main_app.zcat_mapping_dialog.z_combo_box.setCurrentIndex(2)
    main_app.zcat_mapping_dialog.qf_combo_box.setCurrentIndex(4)
    save_screen(qtbot, main_app.zcat_mapping_dialog)
    main_app.zcat_mapping_dialog.accept()
    save_screen(qtbot, main_app.main_wnd)

