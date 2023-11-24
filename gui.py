#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: daddona
"""
import sys
import typing
from PySide6 import QtCore, QtGui, QtUiTools, QtWidgets


class GuiApp():
    """General class for the main GUI."""

    def __init__(self):
        self.main_wnd = loadUiWidget("main_window.ui")

    def run(self):
        """
        Run the main Qt Application.

        Returns
        -------
        None.

        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            # if it does not exist then a QApplication is created
            app = QtWidgets.QApplication(sys.argv)
        self.main_wnd.show()
        sys.exit(app.exec())


class GuiAppPyside(GuiApp):
    """Main App GUI in PySide."""

    def __init__(self):
        super().__init__()


def loadUiWidget(uifilename: str,
                 parent: typing.Optional[QtWidgets.QWidget] = None
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
    loader = QtUiTools.QUiLoader()
    uifile = QtCore.QFile(uifilename)
    uifile.open(QtCore.QFile.ReadOnly)
    ui = loader.load(uifile, parent)
    uifile.close()
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
