#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: daddona
"""
import os
import sys
import uuid
import typing
import logging

from astropy.nddata import VarianceUncertainty, StdDevUncertainty

from utils import loadSpectrum
import backends

try:
    from PySide6 import QtCore, QtGui, QtUiTools, QtWidgets, QtCharts
except (ImportError, ModuleNotFoundError):
    from PyQt6 import QtCore, QtGui, QtWidgets, QtCharts, uic
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
        self.main_wnd.specListWidget.currentItemChanged.connect(
            self.currentSpecItemChanged
        )

        self.fluxQChartView = QtCharts.QChartView(
            self.main_wnd.fluxGroupBox
        )
        self.fluxQChartView.setObjectName("fluxQChartView")
        self.fluxQChartView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.fluxQChartView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.main_wnd.fluxGroupBoxGridLayout.addWidget(
            self.fluxQChartView, 0, 0, 1, 1
        )

        self.varQChartView = QtCharts.QChartView(
            self.main_wnd.varianceGroupBox
        )
        self.varQChartView.setObjectName("varQChartView")
        self.varQChartView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.varQChartView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.main_wnd.varGroupBoxGridLayout.addWidget(
            self.varQChartView, 0, 0, 1, 1
        )

    def currentSpecItemChanged(self, new_item, *args, **kwargs):
        """
        Update widgets when the current object changes.

        Parameters
        ----------
        new_item : TYPE
            DESCRIPTION.
        *args : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        spec_uuid = new_item.uuid
        sp = self.open_spectra[spec_uuid]

        wav = sp.spectral_axis.value
        wav_unit = sp.spectral_axis.unit

        flux = sp.flux.value
        flux_unit = sp.flux.unit

        var = None
        var_unit = None

        if isinstance(sp.uncertainty, VarianceUncertainty):
            var = sp.uncertainty.array
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = sp.uncertainty.array ** 2

        flux_chart = self.fluxQChartView.chart()
        var_chart = self.varQChartView.chart()

        flux_axis_x = QtCharts.QValueAxis()
        flux_axis_x.setTickCount(10)
        flux_axis_x.setLabelFormat("%.2f")
        flux_axis_x.setTitleText("Flux")

        flux_axis_y = QtCharts.QValueAxis()
        flux_axis_y.setTickCount(10)
        flux_axis_y.setLabelFormat("%.2f")
        flux_axis_y.setTitleText("Flux")

        flux_series = values2series(wav, flux, "Flux", )
        flux_series.attachAxis(flux_axis_x)
        flux_series.attachAxis(flux_axis_y)
        flux_chart.addSeries(flux_series)
        flux_chart.addAxis(flux_axis_x, QtCore.Qt.AlignBottom)
        flux_chart.addAxis(flux_axis_y, QtCore.Qt.AlignLeft)

        if var is None:
            self.main_wnd.varianceGroupBox.setEnabled(False)
        else:
            self.main_wnd.varianceGroupBox.setEnabled(True)

            var_axis_x = QtCharts.QValueAxis()
            var_axis_x.setTickCount(10)
            var_axis_x.setLabelFormat("%.2f")
            var_axis_x.setTitleText("Flux")

            var_axis_y = QtCharts.QValueAxis()
            var_axis_y.setTickCount(10)
            var_axis_y.setLabelFormat("%.2f")
            var_axis_y.setTitleText("Flux")

            var_series = values2series(wav, var, "Flux", )
            var_series.attachAxis(var_axis_x)
            var_series.attachAxis(var_axis_y)
            var_chart.addSeries(var_series)
            var_chart.addAxis(var_axis_x, QtCore.Qt.AlignBottom)
            var_chart.addAxis(var_axis_y, QtCore.Qt.AlignLeft)

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
            new_item = QtWidgets.QListWidgetItem(f"{sp.obj_id}")
            new_item.setCheckState(QtCore.Qt.CheckState.Checked)
            new_item.setToolTip(file)
            new_item.uuid = uuid.uuid4()

            self.open_spectra[new_item.uuid] = sp
            self.main_wnd.specListWidget.addItem(new_item)

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


def values2series(x_values, y_values, name):
    series = QtCharts.QLineSeries()
    series.setName(name)

    for x, y in zip(x_values, y_values):
        series.append(x, y)

    return series

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
