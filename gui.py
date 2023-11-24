#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: daddona
"""
import os
import sys
import typing
import logging


from utils import loadSpectrum
import backends

try:
    from PySide6 import QtCore, QtGui, QtUiTools, QtWidgets
except (ImportError, ModuleNotFoundError):
    from PyQt6 import QtCore, QtGui, QtWidgets, uic
    QT_BACKEND = 'PyQt6'
else:
    QT_BACKEND = 'PySide6'


class GuiApp():
    """General class for the main GUI."""

    def __init__(self, qt_backend: str):
        self.qapp = QtWidgets.QApplication.instance()
        if self.qapp is None:
            # if it does not exist then a QApplication is created
            self.qapp = QtWidgets.QApplication(sys.argv)

        self.open_spectra = {}

        self.main_wnd = loadUiWidget("main_window.ui", qt_backend=qt_backend)

        self.main_wnd.importSpecPushButton.clicked.connect(
            self.doImportSpectra
        )


    def doImportSpectra(self, *args, **kwargs):
        """
        Use QFileDialog to get spectra files.

        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        file_list, files_type = QtWidgets.QFileDialog.getOpenFileNames(
            self.main_wnd,
            self.qapp.tr("Import Spectra"),
            '.',
            (
                f"{self.qapp.tr('FITS')} (*.fit *.fits);;"
                f"{self.qapp.tr('ASCII')} (*.txt *.dat *.cat);;"
                f"{self.qapp.tr('All Files')} (*.*)"
            )
        )

        for file in file_list:
            sp = loadSpectrum(file)

    def run(self):
        """
        Run the main Qt Application.

        Returns
        -------
        None.

        """
        self.main_wnd.show()
        sys.exit(self.qapp.exec())


class GuiAppPyside(GuiApp):
    """Main App GUI in PySide."""

    def __init__(self):
        super().__init__(QT_BACKEND)


def loadUiWidget(uifilename: str,
                 parent: typing.Optional[QtWidgets.QWidget] = None,
                 qt_backend: typing.Optional[str] = 'PySide6'
                 ) -> QtWidgets.QWidget:
    """
    Load an UI file.

    Parameters
    ----------
    uifilename : str
        Path of the UI file to load.
    parent : Qwidget, optional
        Parent widget. The default is None.

    Returns
    -------
    ui : Qwidget
        The widget loaded from the UI file.

    """
    logging.debug(f"Loading UI using Qt backend '{qt_backend}'")
    ui_file_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'ui',
        uifilename
    )

    if qt_backend == 'PySide6':
        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile(ui_file_path)
        uifile.open(QtCore.QFile.ReadOnly)
        ui = loader.load(uifile, parent)
        uifile.close()
    elif qt_backend == 'PyQt6':
        ui = uic.loadUi(ui_file_path)

    return ui


def main():
    """
    Run the main GUI application using PySide.

    Returns
    -------
    None.

    """
    myapp = GuiAppPyside()
    myapp.run()


if __name__ == '__main__':
    main()
