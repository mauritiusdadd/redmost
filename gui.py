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


class AdvancedQChartView(QtCharts.QChartView):
    """Subclass of QtCharts.QChartView with advanced features."""

    onMouseMoveSeries = QtCore.Signal(object)
    onMousePressSeries = QtCore.Signal(object)
    onMouseReleaseSeries = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def zoomIn(self):
        self.chart().zoomIn()

    def zoomOut(self):
        self.chart().zoomOut()

    def zoom(self, value):
        self.chart.zoom(value)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.onMouseMoveSeries.emit(self.toSeriesPos(event))

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.onMouseMoveSeries.emit(self.toSeriesPos(event))

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.onMouseMoveSeries.emit(self.toSeriesPos(event))

    def toSeriesPos(self, event):
        widgetPos = event.localPos()
        scenePos = self.mapToScene(widgetPos.x(), widgetPos.y())
        chartItemPos = self.chart().mapFromScene(scenePos)
        valueGivenSeries = self.chart().mapToValue(chartItemPos)
        return valueGivenSeries


class GuiApp():
    """General class for the main GUI."""

    def __init__(self, qt_backend: str):
        self.qapp = QtWidgets.QApplication.instance()
        if self.qapp is None:
            # if it does not exist then a QApplication is created
            self.qapp = QtWidgets.QApplication(sys.argv)

        self.open_spectra = {}

        self.main_wnd = loadUiWidget("main_window.ui", qt_backend=qt_backend)

        # Status Bar
        self.mousePosLabel = QtWidgets.QLabel("")
        self.main_wnd.statusBar().addPermanentWidget(self.mousePosLabel)

        self.main_wnd.fluxContainerWidget.setContentsMargins(0, 0, 0, 0)

        self.fluxQChartView = AdvancedQChartView(
            self.main_wnd.fluxGroupBox
        )
        self.fluxQChartView.setObjectName("fluxQChartView")
        self.fluxQChartView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.fluxQChartView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.fluxQChartView.setDragMode(
            QtWidgets.QGraphicsView.ScrollHandDrag
        )
        self.fluxQChartView.setContentsMargins(0, 0, 0, 0)
        self.fluxQChartView.chart().setContentsMargins(0, 0, 0, 0)
        self.fluxQChartView.chart().layout().setContentsMargins(0, 0, 0, 0)

        # self.fluxQChartView.setRubberBand(
        #     QtCharts.QChartView.HorizontalRubberBand
        # )
        self.main_wnd.fluxWidgetLayout.addWidget(self.fluxQChartView)

        self.varQChartView = AdvancedQChartView(
            self.main_wnd.varianceGroupBox
        )
        self.varQChartView.setObjectName("varQChartView")
        self.varQChartView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.varQChartView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.varQChartView.setContentsMargins(0, 0, 0, 0)
        self.varQChartView.chart().setContentsMargins(0, 0, 0, 0)
        self.varQChartView.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.main_wnd.varWidgetLayout.addWidget(self.varQChartView)

        # Connect signals

        self.main_wnd.importSpecPushButton.clicked.connect(
            self.doImportSpectra
        )
        self.main_wnd.specListWidget.currentItemChanged.connect(
            self.currentSpecItemChanged
        )
        self.main_wnd.actionZoomIn.triggered.connect(self.doZoomIn)
        self.main_wnd.actionZoomOut.triggered.connect(self.doZoomOut)

        self.fluxQChartView.onMouseMoveSeries.connect(self.updateMouseLabel)
        self.varQChartView.onMouseMoveSeries.connect(self.updateMouseLabel)

    def updateMouseLabel(self, mouse_pos):
        self.mousePosLabel.setText(f"\u03BB = {mouse_pos.x():.2f}")

    def doZoomIn(self, *args, **kwargs):
        """
        Zoom In flux and var plots.

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
        self.fluxQChartView.zoomIn()
        self.varQChartView.zoomIn()

    def doZoomOut(self, *args, **kwargs):
        """
        Zoom Out flux and var plots.

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
        self.fluxQChartView.zoomOut()
        self.varQChartView.zoomOut()

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
            var_unit = sp.uncertainty.unit
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = sp.uncertainty.array ** 2
            var_unit = sp.uncertainty.unit ** 2

        flux_chart = self.fluxQChartView.chart()

        flux_series = values2series(wav, flux, "Flux")

        flux_chart.addSeries(flux_series)
        #flux_chart.createDefaultAxes()

        flux_axis_x = QtCharts.QValueAxis()
        flux_axis_x.setTickInterval(500)
        flux_axis_x.setLabelFormat("%.2f")
        flux_axis_x.setTitleText(str(wav_unit))

        flux_axis_y = QtCharts.QValueAxis()
        flux_axis_y.setTickCount(10)
        flux_axis_y.setLabelFormat("%.2f")
        flux_axis_y.setTitleText(str(flux_unit))

        flux_chart.addAxis(flux_axis_x, QtCore.Qt.AlignBottom)
        flux_chart.addAxis(flux_axis_y, QtCore.Qt.AlignLeft)

        flux_series.attachAxis(flux_axis_x)
        flux_series.attachAxis(flux_axis_y)

        if var is None:
            self.main_wnd.varianceGroupBox.setEnabled(False)
        else:
            self.main_wnd.varianceGroupBox.setEnabled(True)
            var_chart = self.varQChartView.chart()
            var_series = values2series(wav, var, "Variance")
            var_chart.addSeries(var_series)

            var_axis_x = QtCharts.QValueAxis()
            var_axis_x.setTickInterval(500)
            var_axis_x.setLabelFormat("%.2f")
            var_axis_x.setTitleText(str(wav_unit))

            var_axis_y = QtCharts.QValueAxis()
            var_axis_y.setTickCount(10)
            var_axis_y.setLabelFormat("%.2f")
            var_axis_y.setTitleText(str(var_unit))

            var_chart.addAxis(var_axis_x, QtCore.Qt.AlignBottom)
            var_chart.addAxis(var_axis_y, QtCore.Qt.AlignLeft)

            var_series.attachAxis(var_axis_x)
            var_series.attachAxis(var_axis_y)

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
