#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: daddona
"""
from __future__ import annotations

import os
import sys
import uuid
from enum import Enum
from typing import Optional, Union, Tuple, List, Dict, Any, cast
import logging

import numpy as np
from astropy.nddata import VarianceUncertainty, StdDevUncertainty
from astropy import units

from specutils import Spectrum1D  # type: ignore

from pyzui import utils
from pyzui import backends
from pyzui import lines

try:
    from PyQt6 import QtCore, QtGui, QtWidgets, QtCharts, uic
    from PyQt6.QtCore import pyqtSignal as Signal
except (ImportError, ModuleNotFoundError):
    from PySide6 import QtCore, QtGui, QtUiTools, QtWidgets, QtCharts
    from PySide6.QtCore import Signal
    QT_BACKEND = 'PySide6'
else:
    QT_BACKEND = 'PyQt6'


def getQApp() -> QtWidgets.QApplication:
    qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
    if qapp is None:
        # if it does not exist then a QApplication is created
        qapp = QtWidgets.QApplication(sys.argv)
    return qapp


class AdvancedQChartView(QtCharts.QChartView):
    """Subclass of QtCharts.QChartView with advanced features."""
    onMouseMoveSeries = Signal(object)
    onMousePressSeries = Signal(object)
    onMouseReleaseSeries = Signal(object)
    onMouseDoubleClickSeries = Signal(object)
    onMouseWheelEvent = Signal(object)

    def __init__(self, *args: List[Any]) -> None:
        super().__init__(*args)

        self.vertical_lock: bool = False
        self.horizontall_lock: bool = False

        self._last_mouse_pos: QtCore.QPointF | None = None

        self.setRubberBand(QtCharts.QChartView.RubberBand.RectangleRubberBand)
        self.setDragMode(QtCharts.QChartView.DragMode.NoDrag)
        self.setMouseTracking(True)

        # Pass events also to siblings
        self.siblings: List[AdvancedQChartView] = []
        self._sibling_locked: bool = False

    def addSibling(self, sibling: AdvancedQChartView) -> None:
        """
        Add sibling to this widget.

        Parameters
        ----------
        sibling : AdvancedQChartView
            A new sibling AdvancedQChartView

        Returns
        -------
        None

        """
        if sibling not in self.siblings:
            self.siblings.append(sibling)
            sibling.addSibling(self)

    def removeSibling(self, sibling: AdvancedQChartView) -> None:
        """
        Remove a sibling to this widget.

        Parameters
        ----------
        sibling : AdvancedQChartView
            The sibling to remove.

        Returns
        -------
        None

        """
        if sibling in self.siblings:
            self.siblings.pop(self.siblings.index(sibling))
            sibling.removeSibling(self)

    def zoomIn(
        self, value: float = 2.0,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None
    ) -> None:
        """
        Zoom in.

        Parameters
        ----------
        value : float
            Zoom value. The default value is 2.0.
        x_center : float
            x of the zoom center
        y_center : float
            y of the zoom center

        Returns
        -------
        None.

        """
        self.zoom(value, x_center, y_center)

    def zoomOut(
        self,
        value: float = 0.5,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None
    ) -> None:
        """
        Zoom out.

        Parameters
        ----------
        value : float
            Zoom value. The default value is 0.5.
        x_center : float, optional
            x of the zoom center
        y_center : float, optional
            y of the zoom center

        Returns
        -------
        None.

        """
        self.zoom(value, x_center, y_center)

    def _getDataBounds(self) -> Union[
        Tuple[float, float, float, float],
        Tuple[None, None, None, None],
    ]:
        x_min = None
        x_max = None
        y_min = None
        y_max = None
        for series in self.chart().series():
            data: np.ndarray = np.array(
                [(p.x(), p.y()) for p in series.points()]
            )
            p1x, p1y = np.min(data, axis=0)[0:2]
            p2x, p2y = np.max(data, axis=0)[0:2]

            if x_min is None:
                x_min = p1x
                y_min = p1y
                x_max = p2x
                y_max = p2y
            else:
                x_min = min(x_min, p1x)
                y_min = min(y_min, p1y)
                x_max = max(x_max, p2x)
                y_max = max(y_max, p2y)
        return x_min, y_min, x_max, y_max

    def zoomReset(self) -> None:
        """
        Reset the zoom to fit the chart content.

        Returns
        -------
        None

        """
        if self._sibling_locked:
            return

        self._sibling_locked = True
        for sibling in self.siblings:
            sibling.zoomReset()
        self._sibling_locked = False

        chart = self.chart()
        x1, y1, x2, y2 = self._getDataBounds()
        if x1 is not None:
            x_axis = chart.axes()[0]
            y_axis = chart.axes()[1]
            x_axis.setMin(x1)
            y_axis.setMin(y1)
            x_axis.setMax(x2)
            y_axis.setMax(y2)

    def zoom(
        self,
        value: float,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None
    ) -> None:
        """
        Zoom.

        Parameters
        ----------
        value : float
            Zoom value.
        x_center : float, optional
            x of the zoom center
        y_center : float, optional
            y of the zoom center

        Returns
        -------
        None.

        """
        if self._sibling_locked:
            return

        rect = self.chart().plotArea()

        self._sibling_locked = True
        for sibling in self.siblings:
            sibling.zoom(value, x_center, y_center)
        self._sibling_locked = False

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

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
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
        if self._sibling_locked:
            return

        self._sibling_locked = True
        for sibling in self.siblings:
            pass
        self._sibling_locked = False

        super().mouseDoubleClickEvent(event)
        self.onMouseDoubleClickSeries.emit((self.toSeriesPos(event), event))

    def mouseMoveEvent(self, event: QtGui.QMouseEvent, **kwargs) -> None:
        """
        Handle mouse move events.

        Parameters
        ----------
        **kwargs
        event : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self._sibling_locked:
            return

        if event.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            if self._last_mouse_pos is None:
                return

            delta = event.position() - self._last_mouse_pos

            self.scroll(-delta.x(), delta.y())

            self._last_mouse_pos = event.position()
            event.accept()

        super().mouseMoveEvent(event)
        self.onMouseMoveSeries.emit((self.toSeriesPos(event), event))

    def scroll(self, dx: float, dy: float) -> None:
        if self._sibling_locked:
            return

        self._sibling_locked = True
        for sibling in self.siblings:
            sibling.scroll(dx, dy)
        self._sibling_locked = False

        chart = self.chart()

        if self.vertical_lock and self.horizontall_lock:
            return
        elif not (self.vertical_lock or self.horizontall_lock):
            chart.scroll(dx, dy)
        elif self.vertical_lock:
            chart.scroll(dx, 0)
        else:
            chart.scroll(0, dy)

    def mousePressEvent(self, event: QtGui.QMouseEvent, **kwargs) -> None:
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
        if self._sibling_locked:
            return

        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            getQApp().setOverrideCursor(
                QtCore.Qt.CursorShape.SizeAllCursor
            )
            self._last_mouse_pos = event.position()
            event.accept()

        super().mousePressEvent(event)
        self.onMousePressSeries.emit((self.toSeriesPos(event), event))

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent, **kwargs) -> None:
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
        if self._sibling_locked:
            return

        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            getQApp().restoreOverrideCursor()
            event.accept()

        super().mouseReleaseEvent(event)
        self.onMouseReleaseSeries.emit((self.toSeriesPos(event), event))

    def wheelEvent(self, event: QtGui.QWheelEvent, **kwargs) -> None:
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

        if modifiers == QtCore.Qt.KeyboardModifier.ShiftModifier:
            # Vertical scroll
            if not self.vertical_lock:
                self.scroll(0, delta_pix)
        elif modifiers == QtCore.Qt.KeyboardModifier.ControlModifier:
            # Horizontal scroll
            if not self.horizontall_lock:
                self.scroll(delta_pix, 0)
        else:
            # Zoom
            if delta_pix > 0:
                self.zoomIn()
            else:
                self.zoomOut()

        self.onMouseWheelEvent.emit(event)

    def toSeriesPos(self, event: QtGui.QMouseEvent):
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
        widget_pos = event.position()
        scene_pos = self.mapToScene(int(widget_pos.x()), int(widget_pos.y()))
        chart_item_pos = self.chart().mapFromScene(scene_pos)
        value_in_series = self.chart().mapToValue(chart_item_pos)
        return value_in_series


class GuiApp:
    """General class for the main GUI."""

    GlobalState: Enum = Enum(
        'GlobalState',
        ['READY', 'SELECT_LINE_MANUAL']
    )

    def __init__(self, qt_backend: str) -> None:
        self.qapp: QtWidgets.QApplication = getQApp()

        self.open_spectra: Dict[uuid.UUID, Spectrum1D] = {}
        self.current_smoothing: float = -1.0
        self._current_spec_uuid: uuid.UUID | None = None
        self.global_state: Enum = self.GlobalState.READY

        self.main_wnd: QtWidgets.QMainWindow = loadUiWidget(
            "main_window.ui", qt_backend=qt_backend
        )

        # Status Bar
        self.mousePosLabel: QtWidgets.QLabel = QtWidgets.QLabel("")

        statusbar = self.main_wnd.statusBar()
        if statusbar:
            statusbar.addPermanentWidget(self.mousePosLabel)

        self.main_wnd.fluxContainerWidget.setContentsMargins(0, 0, 0, 0)

        self.fluxQChartView: AdvancedQChartView = AdvancedQChartView(
            self.main_wnd.fluxGroupBox
        )
        self.fluxQChartView.setObjectName("fluxQChartView")
        self.fluxQChartView.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing
        )
        self.fluxQChartView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.fluxQChartView.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.fluxQChartView.setContentsMargins(0, 0, 0, 0)
        self.fluxQChartView.chart().setContentsMargins(0, 0, 0, 0)
        self.fluxQChartView.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.fluxQChartView.setRubberBand(
            QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

        self.main_wnd.fluxWidgetLayout.addWidget(self.fluxQChartView)

        self.varQChartView: AdvancedQChartView = AdvancedQChartView(
            self.main_wnd.varianceGroupBox
        )
        self.varQChartView.vertical_lock = True
        self.varQChartView.setObjectName("varQChartView")
        self.varQChartView.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing
        )
        self.varQChartView.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.varQChartView.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.varQChartView.setContentsMargins(0, 0, 0, 0)
        self.varQChartView.chart().setContentsMargins(0, 0, 0, 0)
        self.varQChartView.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.varQChartView.setRubberBand(
            QtCharts.QChartView.RubberBand.NoRubberBand
        )

        self.main_wnd.varWidgetLayout.addWidget(self.varQChartView)

        self.fluxQChartView.addSibling(self.varQChartView)

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

        self.fluxQChartView.onMousePressSeries.connect(self.mousePressedFlux)

        self.main_wnd.smoothingCheckBox.stateChanged.connect(
            self.toggleSmothing
        )
        self.main_wnd.smoothingDoubleSpinBox.valueChanged.connect(
            self.setSmoothingFactor
        )

        self.main_wnd.linesAutoPushButton.clicked.connect(
            self.doIdentifyLines
        )

        self.main_wnd.addLinePushButton.clicked.connect(
            self.doAddNewLine
        )

    def _updateMouseLabelFromEvent(self, *args) -> None:
        self._updateMouseLabel(args[0][0])

    def _updateMouseLabel(self, mouse_pos: QtCore.QPointF) -> None:
        self.mousePosLabel.setText(f"\u03BB = {mouse_pos.x():.2f}")

    def mousePressedFlux(self, args) -> None:
        data_pos: QtCore.QPointF = args[0]
        event: QtGui.QMouseEvent = args[1]
        if self.global_state == self.GlobalState.SELECT_LINE_MANUAL:
            self.addLine(data_pos.x())
            self.global_state = self.GlobalState.READY
            self.qapp.restoreOverrideCursor()
            self.unlock()

    def lock(self, *args, **kwargs) -> None:
        self.main_wnd.redGroupBox.setEnabled(False)
        self.main_wnd.infoGroupBox.setEnabled(False)
        self.main_wnd.plotGroupBox.setEnabled(False)

    def unlock(self, *args, **kwargs) -> None:
        self.main_wnd.specGroupBox.setEnabled(True)
        self.main_wnd.redGroupBox.setEnabled(True)
        self.main_wnd.infoGroupBox.setEnabled(True)
        self.main_wnd.plotGroupBox.setEnabled(True)

    def doAddNewLine(self, *args, **kwargs) -> None:
        if self._current_spec_uuid is None:
            return
        self.lock()
        self.main_wnd.specGroupBox.setEnabled(False)
        self.global_state = self.GlobalState.SELECT_LINE_MANUAL
        self.qapp.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

    def doZoomReset(self, *arg, **kwargs) -> None:
        self.fluxQChartView.zoomReset()

    def doZoomIn(self, *args, **kwargs) -> None:
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

    def doZoomOut(self, *args, **kwargs) -> None:
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

    def toggleSmothing(self, show_smoothing: int) -> None:
        if show_smoothing:
            curr_smt_val = self.main_wnd.smoothingDoubleSpinBox.value()
            self.current_smoothing = float(curr_smt_val)
        else:
            self.current_smoothing = -1.0
        self.redrawCurrentSpec()

    def setSmoothingFactor(self, smoothing_value: float) -> None:
        self.current_smoothing = float(smoothing_value)
        self.redrawCurrentSpec()

    def redrawCurrentSpec(self, *args, **kwargs) -> None:
        self.currentSpecItemChanged(
            self.main_wnd.specListWidget.currentItem()
        )

    def doIdentifyLines(self, *args, **kwargs) -> None:
        if self._current_spec_uuid is None:
            return

        sp: Spectrum1D = self.open_spectra[self._current_spec_uuid]

        wav: np.ndarray = sp.spectral_axis.value
        wav_unit = sp.spectral_axis.unit
        flux: np.ndarray = sp.flux.value
        var: np.ndarray | None = None

        if isinstance(sp.uncertainty, VarianceUncertainty):
            var = sp.uncertainty.array
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = sp.uncertainty.array ** 2

        my_lines: List[
            Tuple[int, float, float, float]
        ] = lines.get_spectrum_lines(
            wavelengths=wav,
            flux=flux,
            var=var
        )

        self.main_wnd.linesTableWidget.setRowCount(0)
        self.main_wnd.linesTableWidget.setRowCount(len(my_lines))
        for j, (k, w, l, h) in enumerate(my_lines):
            new_item = QtWidgets.QTableWidgetItem(f"{w:.2f} A")
            new_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable |
                QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            self.main_wnd.linesTableWidget.setItem(j, 0, new_item)

    def addLine(self, wavelength: float) -> None:
        new_item = QtWidgets.QTableWidgetItem(f"{wavelength:.2f} A")
        new_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsSelectable |
            QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        new_item_row: int = self.main_wnd.linesTableWidget.rowCount()
        self.main_wnd.linesTableWidget.setRowCount(new_item_row+1)
        self.main_wnd.linesTableWidget.setItem(new_item_row, 0, new_item)

    def currentSpecItemChanged(self, new_item, *args, **kwargs) -> None:
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
        self.unlock()

        spec_uuid: uuid.UUID = new_item.uuid
        sp: Spectrum1D = self.open_spectra[spec_uuid]

        wav: np.ndarray = sp.spectral_axis.value
        wav_unit: units.Unit = sp.spectral_axis.unit

        flux: np.ndarray = sp.flux.value
        flux_unit: units.Unit = sp.flux.unit

        var: np.ndarray | None = None
        var_unit: units.Unit | None = None

        if isinstance(sp.uncertainty, VarianceUncertainty):
            var = sp.uncertainty.array
            var_unit = sp.uncertainty.unit
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = sp.uncertainty.array ** 2
            var_unit = sp.uncertainty.unit ** 2

        flux_chart = self.fluxQChartView.chart()
        flux_chart.removeAllSeries()

        var_chart = self.varQChartView.chart()
        var_chart.removeAllSeries()

        if spec_uuid != self._current_spec_uuid:
            # If we actually change the spectrum, then reset the view
            self._current_spec_uuid = spec_uuid
            self.main_wnd.linesTableWidget.setRowCount(0)
            for ax in flux_chart.axes():
                flux_chart.removeAxis(ax)
            for ax in var_chart.axes():
                var_chart.removeAxis(ax)

        flux_series = values2series(wav, flux, "Flux")
        flux_chart.addSeries(flux_series)

        if not flux_chart.axes():
            flux_axis_x = QtCharts.QValueAxis()
            flux_axis_y = QtCharts.QValueAxis()
            var_axis_x = QtCharts.QValueAxis()
            var_axis_y = QtCharts.QValueAxis()
            flux_chart.addAxis(
                flux_axis_x, QtCore.Qt.AlignmentFlag.AlignBottom
            )
            flux_chart.addAxis(
                flux_axis_y, QtCore.Qt.AlignmentFlag.AlignLeft
            )
            var_chart.addAxis(
                var_axis_x, QtCore.Qt.AlignmentFlag.AlignBottom
            )
            var_chart.addAxis(
                var_axis_y, QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            flux_axis_x = flux_chart.axes()[0]
            flux_axis_y = flux_chart.axes()[1]
            var_axis_x = var_chart.axes()[0]
            var_axis_y = var_chart.axes()[1]

        flux_axis_x.setTickInterval(500)
        flux_axis_x.setLabelFormat("%.2f")
        flux_axis_x.setTitleText(str(wav_unit))

        flux_axis_y.setLabelFormat("%.2f")
        flux_axis_y.setTitleText(str(flux_unit))

        flux_series.attachAxis(flux_axis_x)
        flux_series.attachAxis(flux_axis_y)

        if self.current_smoothing >= 0:
            flux_series.setOpacity(0.3)
            smoothing_sigma = len(flux) / (1 + 2*self.current_smoothing)
            smoothed_flux = utils.smooth_fft(flux, sigma=smoothing_sigma)
            smoothed_flux_series = values2series(
                wav, smoothed_flux, "Smoothed flux"
            )

            pen = smoothed_flux_series.pen()
            pen.setColor(QtGui.QColor("orange"))
            smoothed_flux_series.setPen(pen)

            flux_chart.addSeries(smoothed_flux_series)
            smoothed_flux_series.attachAxis(flux_axis_x)
            smoothed_flux_series.attachAxis(flux_axis_y)

        flux_chart.setContentsMargins(0, 0, 0, 0)
        flux_chart.setBackgroundRoundness(0)
        flux_chart.legend().hide()

        if var is None:
            self.main_wnd.varianceGroupBox.setEnabled(False)
        else:
            self.main_wnd.varianceGroupBox.setEnabled(True)
            var_series = values2series(wav, var, "Variance")
            var_chart.addSeries(var_series)

            var_axis_x.setTickInterval(500)
            var_axis_x.setLabelFormat("%.2f")
            var_axis_x.setTitleText(str(wav_unit))

            var_axis_y.setTickCount(10)
            var_axis_y.setLabelFormat("%.2f")
            var_axis_y.setTitleText(str(var_unit))

            var_series.attachAxis(var_axis_x)
            var_series.attachAxis(var_axis_y)

        var_chart.setContentsMargins(0, 0, 0, 0)
        var_chart.setBackgroundRoundness(0)
        var_chart.legend().hide()

    def doImportSpectra(self, *args, **kwargs) -> None:
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
            sp: Spectrum1D = utils.loadSpectrum(file)
            new_item = QtWidgets.QListWidgetItem(f"{sp.obj_id}")
            new_item.setCheckState(QtCore.Qt.CheckState.Checked)
            new_item.setToolTip(file)
            new_item.uuid = uuid.uuid4()

            self.open_spectra[new_item.uuid] = sp
            self.main_wnd.specListWidget.addItem(new_item)

    def run(self) -> None:
        """
        Run the main Qt Application.

        Returns
        -------
        None.

        """
        self.main_wnd.show()
        sys.exit(self.qapp.exec())


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
) -> QtWidgets.QMainWindow:
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
    ui : QtWidgets.QMainWindow
        The widget loaded from the UI file.

    """

    logging.debug(f"Loading UI using Qt backend '{qt_backend}'")
    ui_file_path: str = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'ui',
        uifilename
    )

    if qt_backend == 'PyQt6':
        ui: QtWidgets.QMainWindow = uic.loadUi(ui_file_path)
    elif qt_backend == 'PySide6':
        loader = QtUiTools.QUiLoader()
        uifile = QtCore.QFile(ui_file_path)
        uifile.open(QtCore.QIODeviceBase.OpenModeFlag.ReadOnly)
        ui = cast(
            QtWidgets.QMainWindow,
            loader.load(uifile, parent)
        )
        uifile.close()
    else:
        raise NotImplementedError("No GUI backend found!")

    return ui


def main() -> None:
    """
    Run the main GUI application using PySide.

    Returns
    -------
    None.

    """

    myapp: GuiApp = GuiApp(QT_BACKEND)
    myapp.run()


if __name__ == '__main__':
    main()
