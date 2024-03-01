#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: daddona
"""
import os
import sys
import uuid
from typing import Optional, Union, List, Dict
import logging

import numpy as np
from astropy.nddata import VarianceUncertainty, StdDevUncertainty

from specutils import Spectrum1D  # type: ignore

from pyzui import utils
from pyzui import backends

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
    onMouseDoubleClickSeries = QtCore.Signal(object)
    onMouseWheelEvent = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vertical_lock = False
        self.horizontall_lock = False

    def zoomIn(self, value=2, x_center=None, y_center=None):
        """
        Zoom in.

        Parameters
        ----------
        value : float
            Zoom value.
        x_center : float
            x of the zoom center
        y_center : float
            y of the zoom center

        Returns
        -------
        None.

        """
        self.zoom(value, x_center, y_center)

    def zoomOut(self, value=0.5, x_center=None, y_center=None):
        """
        Zoom out.

        Parameters
        ----------
        value : float
            Zoom value.
        x_center : float
            x of the zoom center
        y_center : float
            y of the zoom center

        Returns
        -------
        None.

        """
        self.zoom(value, x_center, y_center)

    def zoom(self, value, x_center=None, y_center=None):
        """
        Zoom.

        Parameters
        ----------
        value : float
            Zoom value.
        x_center : float
            x of the zoom center
        y_center : float
            y of the zoom center

        Returns
        -------
        None.

        """
        rect = self.chart().plotArea()

        if x_center is None:
            x_center = rect.width()/2

        if y_center is None:
            y_center = rect.height()/2

        if not self.horizontall_lock:
            width_original = rect.width()
            rect.setWidth(width_original / value)
            center_scale_x = x_center / width_original
            left_offset = x_center - (rect.width() * center_scale_x)
            rect.moveLeft(rect.x() + left_offset)

        if not self.vertical_lock:
            height_original = rect.height()
            rect.setHeight(height_original / value)
            center_scale_y = y_center / height_original
            top_offset = y_center - (rect.height() * center_scale_y)
            rect.moveTop(rect.y() + top_offset)

        self.chart().zoomIn(rect)

    def mouseDoubleClickEvent(self, event):
        """
        Handle mouse double click events.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().mouseDoubleClickEvent(event)
        self.onMouseDoubleClickSeries.emit((self.toSeriesPos(event), event))

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().mouseMoveEvent(event)
        self.onMouseMoveSeries.emit((self.toSeriesPos(event), event))

    def mousePressEvent(self, event):
        """
        Handle mouse button press events.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().mousePressEvent(event)
        self.onMousePressSeries.emit((self.toSeriesPos(event), event))

    def mouseReleaseEvent(self, event):
        """
        Handle mouse button release events.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().mouseReleaseEvent(event)
        self.onMouseReleaseSeries.emit((self.toSeriesPos(event), event))

    def wheelEvent(self, event):
        """
        Handle mouse wheel events.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().wheelEvent(event)
        delta_pix = event.pixelDelta().y() / 5

        modifiers = QtWidgets.QApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.ShiftModifier:
            # Vertical scroll
            if not self.vertical_lock:
                self.chart().scroll(0, delta_pix)
        elif modifiers == QtCore.Qt.ControlModifier:
            # Zoom
            if delta_pix > 0:
                self.zoomIn()
            else:
                self.zoomOut()
        else:
            # Horizontal scroll
            if not self.horizontall_lock:
                self.chart().scroll(delta_pix, 0)

        self.onMouseWheelEvent.emit(event)

    def toSeriesPos(self, event):
        """
        Convert mouse position from event location to data values.

        Parameters
        ----------
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        valueGivenSeries : TYPE
            DESCRIPTION.

        """
        widgetPos = event.position()
        scenePos = self.mapToScene(widgetPos.x(), widgetPos.y())
        chartItemPos = self.chart().mapFromScene(scenePos)
        valueGivenSeries = self.chart().mapToValue(chartItemPos)
        return valueGivenSeries


class GuiApp:
    """General class for the main GUI."""

    def __init__(self, qt_backend: str):
        self.qapp = QtWidgets.QApplication.instance()
        if self.qapp is None:
            # if it does not exist then a QApplication is created
            self.qapp = QtWidgets.QApplication(sys.argv)

        self.open_spectra: Dict[str, Spectrum1D] = {}
        self.current_smoothing: float = -1.0

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
        self.fluxQChartView.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.fluxQChartView.setContentsMargins(0, 0, 0, 0)
        self.fluxQChartView.chart().setContentsMargins(0, 0, 0, 0)
        self.fluxQChartView.chart().layout().setContentsMargins(0, 0, 0, 0)

        self.main_wnd.fluxWidgetLayout.addWidget(self.fluxQChartView)

        self.varQChartView = AdvancedQChartView(
            self.main_wnd.varianceGroupBox
        )
        self.varQChartView.vertical_lock = True
        self.varQChartView.setObjectName("varQChartView")
        self.varQChartView.setRenderHint(QtGui.QPainter.Antialiasing)
        self.varQChartView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self.varQChartView.setVerticalScrollBarPolicy(
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
        self.main_wnd.actionZoomFit.triggered.connect(self.doZoomReset)

        self.fluxQChartView.onMouseMoveSeries.connect(
            self._updateMouseLabelFromEvent
        )
        self.varQChartView.onMouseMoveSeries.connect(
            self._updateMouseLabelFromEvent
        )

        self.fluxQChartView.onMouseWheelEvent.connect(
            self.varQChartView.wheelEvent
        )
        self.fluxQChartView.onMousePressSeries.connect(self.mousePressedFlux)

        self.main_wnd.smoothingCheckBox.stateChanged.connect(
            self.toggleSmothing
        )
        self.main_wnd.smoothingDoubleSpinBox.valueChanged.connect(
            self.setSmoothingFactor
        )

    def _updateMouseLabelFromEvent(self, args):
        self._updateMouseLabel(args[0])

    def _updateMouseLabel(self, mouse_pos):
        self.mousePosLabel.setText(f"\u03BB = {mouse_pos.x():.2f}")

    def mouseScollFlux(self, args):
        data_pos = args[0]
        event = args[1]

    def mousePressedFlux(self, args):
        data_pos = args[0]
        event = args[1]

    def doZoomReset(self, *arg, **kwargs):
        self.fluxQChartView.chart().zoomReset()
        self.varQChartView.chart().zoomReset()

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

    def toggleSmothing(self, show_smoothing: int):
        if show_smoothing:
            curr_smt_val = self.main_wnd.smoothingDoubleSpinBox.value()
            self.current_smoothing = float(curr_smt_val)
        else:
            self.current_smoothing = -1.0
        self.redrawCurrentSpec()

    def setSmoothingFactor(self, smoothing_value: float):
        self.current_smoothing = float(smoothing_value)
        self.redrawCurrentSpec()

    def redrawCurrentSpec(self, *args, **kwargs):
        # TODO: save and restore current zoom
        self.currentSpecItemChanged(
            self.main_wnd.specListWidget.currentItem()
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
            var_unit = sp.uncertainty.unit
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = sp.uncertainty.array ** 2
            var_unit = sp.uncertainty.unit ** 2

        flux_chart = self.fluxQChartView.chart()
        flux_chart.removeAllSeries()
        for ax in flux_chart.axes():
            flux_chart.removeAxis(ax)

        flux_series = values2series(wav, flux, "Flux")

        flux_chart.addSeries(flux_series)

        flux_axis_x = QtCharts.QValueAxis()
        flux_axis_x.setTickInterval(500)
        flux_axis_x.setLabelFormat("%.2f")
        flux_axis_x.setTitleText(str(wav_unit))

        flux_axis_y = QtCharts.QValueAxis()
        flux_axis_y.setLabelFormat("%.2f")
        flux_axis_y.setTitleText(str(flux_unit))

        flux_chart.addAxis(flux_axis_x, QtCore.Qt.AlignBottom)
        flux_chart.addAxis(flux_axis_y, QtCore.Qt.AlignLeft)

        flux_series.attachAxis(flux_axis_x)
        flux_series.attachAxis(flux_axis_y)

        if self.current_smoothing >= 0:
            smoothing_sigma =  len(flux) / (1 + 2*self.current_smoothing)
            smoothed_flux = utils.smooth_fft(flux, sigma=smoothing_sigma)
            smoothed_flux_series = values2series(
                wav, smoothed_flux, "Smoothed flux"
            )
            flux_chart.addSeries(smoothed_flux_series)
            smoothed_flux_series.attachAxis(flux_axis_x)
            smoothed_flux_series.attachAxis(flux_axis_y)

        var_chart = self.varQChartView.chart()
        var_chart.removeAllSeries()
        for ax in var_chart.axes():
            var_chart.removeAxis(ax)

        if var is None:
            self.main_wnd.varianceGroupBox.setEnabled(False)
        else:
            self.main_wnd.varianceGroupBox.setEnabled(True)
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
            sp = utils.loadSpectrum(file)
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


def values2series(
    x_values: Union[List[float], np.ndarray],
    y_values: Union[List[float], np.ndarray],
    name: str
) -> QtCharts.QLineSeries:
    """
    Convert point values to a QLineSeries object.

    Parameters
    ----------
    x_values : TYPE
        DESCRIPTION.
    y_values : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    Returns
    -------
    series : TYPE
        DESCRIPTION.

    """
    series: QtCharts.QLineSeries = QtCharts.QLineSeries()
    series.setName(name)

    for x, y in zip(x_values, y_values):
        series.append(x, y)

    return series


def loadUiWidget(
    uifilename: str,
    parent: Optional[QtWidgets.QWidget] = None,
    qt_backend: Optional[str] = 'PySide6'
) -> QtWidgets.QWidget:
    """
    Load a UI file.

    Parameters
    ----------
    uifilename : str
        Path of the UI file to load.
    parent : Qwidget, optional
        Parent widget. The default is None.
    qt_backend : str, optional.
        The UI backed to use. The default value is 'PySide6'

    Returns
    -------
    ui : Qwidget
        The widget loaded from the UI file.

    """

    logging.debug(f"Loading UI using Qt backend '{qt_backend}'")
    ui_file_path: str = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'ui',
        uifilename
    )

    if qt_backend == 'PySide6':
        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile(ui_file_path)
        uifile.open(QtCore.QFile.ReadOnly)
        ui: QtWidgets.QWidget = loader.load(uifile, parent)
        uifile.close()
    elif qt_backend == 'PyQt6':
        ui = uic.loadUi(ui_file_path)

    return ui


def main() -> None:
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
