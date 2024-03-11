#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: daddona
"""
from __future__ import annotations

import os
import pickle
import time
import sys
import json
import uuid
from io import StringIO
from enum import Enum
from typing import Optional, Union, Tuple, List, Dict, Any, cast, Callable
import logging

import numpy as np
from astropy.nddata import VarianceUncertainty
from astropy.nddata import StdDevUncertainty
from astropy.nddata import InverseVariance
from astropy import units
from astropy.table import Table

from specutils import Spectrum1D

from pyzui import loaders
from pyzui import utils
from pyzui import lines
from pyzui import backends

try:
    from PyQt6 import QtCore, QtGui, QtWidgets, QtCharts, uic
    from PyQt6.QtCore import pyqtSignal as Signal
except (ImportError, ModuleNotFoundError):
    try:
        from PySide6 import QtCore, QtGui, QtUiTools, QtWidgets, QtCharts
        from PySide6.QtCore import Signal
    except (ImportError, ModuleNotFoundError):
        from tkinter import messagebox

        messagebox.showerror(
            title="ERROR",
            message="Please install either PyQt6 or PySide6"
        )
    QT_BACKEND = 'PySide6'
else:
    QT_BACKEND = 'PyQt6'


def getQApp() -> QtWidgets.QApplication:
    qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
    if qapp is None:
        # if it does not exist then a QApplication is created
        qapp = QtWidgets.QApplication(sys.argv)
    return qapp


class SpectrumQChartView(QtCharts.QChartView):
    """Subclass of QtCharts.QChartView with advanced features."""
    onMouseMoveSeries = Signal(object)
    onMousePressSeries = Signal(object)
    onMouseReleaseSeries = Signal(object)
    onMouseDoubleClickSeries = Signal(object)
    onMouseWheelEvent = Signal(object)

    def __init__(self, *args: List[Any]) -> None:
        super().__init__(*args)

        self._last_mouse_pos: QtCore.QPointF | None = None
        self._static_lines_list: List[Tuple[float, str, str]] = []
        self._lines_buffer_list: List[Tuple[float, str, str]] = []
        self._lines_type: Union[str, None] = None
        self.line_colors: Dict[str, str] = {
            "absorption": "#b02000",
            "both": "#d47800",
            "emission": "#61b000",
            "unknown": "#222222"
        }
        self.line_plot_options: Dict[str, Any] = {
            "alpha": 0.5,
            "dash_pattern": (8, 8),
            "width": 1.5
        }

        self.horizontall_lock: bool = False
        self.master: bool = False
        self.show_lines: bool = False
        self.sync_height: bool = False
        self.sync_width: bool = False
        self.redshift: float = 0
        self.vertical_lock: bool = False

        self.setDragMode(QtCharts.QChartView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setRubberBand(QtCharts.QChartView.RubberBand.RectangleRubberBand)

        self.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )

        self.updateLines()

        # Pass events also to siblings
        self._sibling_locked: bool = False
        self.siblings: List[SpectrumQChartView] = []

        self.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding
            )
        )
        self.chart().setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding
            )
        )

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

    def addSibling(self, sibling: SpectrumQChartView) -> None:
        """
        Add sibling to this widget.

        Parameters
        ----------
        sibling : SpectrumQChartView
            A new sibling SpectrumQChartView

        Returns
        -------
        None

        """
        if sibling not in self.siblings:
            self.siblings.append(sibling)
            sibling.addSibling(self)

    def drawForeground(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF
    ) -> None:
        super().drawForeground(painter, rect)

        if not self.show_lines:
            return

        painter.save()
        plot_area: QtCore.QRectF = self.chart().plotArea()

        for line_lambda, line_name, line_type in self._lines_buffer_list:
            pen: QtGui.QPen = QtGui.QPen()

            pen_color: QtGui.QColor = QtGui.QColor(self.line_colors["unknown"])
            if ('AE' in line_type) or ('EA' in line_type):
                pen_color = QtGui.QColor(self.line_colors["both"])
            elif 'A' in line_type:
                pen_color = QtGui.QColor(self.line_colors["absorption"])
            elif 'E' in line_type:
                pen_color = QtGui.QColor(self.line_colors["emission"])

            pen_color.setAlphaF(self.line_plot_options["alpha"])
            pen.setColor(pen_color)
            pen.setWidthF(self.line_plot_options["width"])
            pen.setDashPattern(self.line_plot_options["dash_pattern"])
            painter.setPen(pen)

            line_x = self.chart().mapToPosition(
                QtCore.QPointF(line_lambda, 0)
            ).x()

            if (line_x >= plot_area.left()) and (line_x <= plot_area.right()):
                p_line_top = QtCore.QPointF(line_x, plot_area.top())
                p_line_bottom = QtCore.QPointF(line_x, plot_area.bottom())
                p_text_top = QtCore.QPointF(line_x, plot_area.top() - 5)
                painter.drawLine(p_line_bottom, p_line_top)
                painter.drawText(p_text_top, line_name)

        painter.restore()

    def hideLines(self) -> None:
        self.setLinesVisible(False)

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

        self._sibling_locked = True
        for sibling in self.siblings:
            sibling.mouseReleaseEvent(event)
        self._sibling_locked = False

        if event.button() == QtCore.Qt.MouseButton.MiddleButton:
            getQApp().restoreOverrideCursor()
            event.accept()
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
        else:
            # Passtrough for rubber band handling
            super().mouseReleaseEvent(event)

        self.onMouseReleaseSeries.emit((self.toSeriesPos(event), event))
        if self.rubberBand() != QtCharts.QChartView.RubberBand.NoRubberBand:
            self.syncSiblingAxes()

    def removeSibling(self, sibling: SpectrumQChartView) -> None:
        """
        Remove a sibling to this widget.

        Parameters
        ----------
        sibling : SpectrumQChartView
            The sibling to remove.

        Returns
        -------
        None

        """
        if sibling in self.siblings:
            self.siblings.pop(self.siblings.index(sibling))
            sibling.removeSibling(self)

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

    def setAxesRange(
        self,
        x_min: Optional[float],
        y_min: Optional[float],
        x_max: Optional[float],
        y_max: Optional[float]
    ) -> None:
        if self._sibling_locked:
            return

        axes = self.chart().axes()
        if len(axes) < 2:
            return
        x_axis, y_axis = axes[0:2]

        if not self.horizontall_lock:
            if x_min is not None:
                x_axis.setMin(x_min)
            if x_max is not None:
                x_axis.setMax(x_max)
        if not self.vertical_lock:
            if y_min is not None:
                y_axis.setMin(y_min)
            if y_max is not None:
                y_axis.setMax(y_max)

    def setLinesType(self, type: str) -> None:
        self._lines_type = type
        self.updateLines()

    def setLinesVisible(self, show: bool) -> None:
        self.show_lines = show
        self.chart().update()
        self.update()

    def setRedshift(self, redshift: float) -> None:
        self.redshift = redshift
        self.updateLines()

    def showLines(self) -> None:
        self.setLinesVisible(True)

    def syncSiblingAxes(self) -> None:
        if self._sibling_locked:
            return

        axes = self.chart().axes()
        if len(axes) < 2:
            return
        x_axis, y_axis = axes[0:2]

        self._sibling_locked = True
        # Get current range for x and y axes
        x_min = x_axis.min()
        y_min = y_axis.min()
        x_max = x_axis.max()
        y_max = y_axis.max()

        plot_area: QtCore.QRectF = self.chart().plotArea()

        for sibling in self.siblings:

            sibling.setAxesRange(x_min, y_min, x_max, y_max)

            if sibling.sync_width or sibling.sync_height:
                sibling_chart: QtCharts.QChart = sibling.chart()
                sibling_chart.setPlotArea(QtCore.QRectF())
                sibling_plot_area: QtCore.QRectF = sibling_chart.plotArea()
                if sibling.sync_width:
                    sibling_plot_area.setX(plot_area.x())
                    sibling_plot_area.setWidth(plot_area.width())

                if sibling.sync_height:
                    sibling_plot_area.setY(plot_area.y())
                    sibling_plot_area.setHeight(plot_area.height())

                sibling_chart.setPlotArea(sibling_plot_area)
                sibling_chart.update()
                sibling.update()

        self._sibling_locked = False

    def resizeEvent(self, event, QResizeEvent=None) -> None:
        super().resizeEvent(event)
        if self.master:
            self.syncSiblingAxes()

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

    def updateLines(self) -> None:
        known_lines: List[Tuple[float, str, str]] = lines.get_lines(
            line_type=self._lines_type,
            z=self.redshift
        )

        self._lines_buffer_list = known_lines + self._static_lines_list
        self.chart().update()
        self.update()

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

        self.setAxesRange(*self._getDataBounds())


class QRedrockHandler(QtCore.QObject):

    progress: Signal = Signal(int)
    maximum: Signal = Signal(int)
    reset: Signal = Signal()
    message: Signal = Signal(str)
    finished: Signal = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.qapp = getQApp()
        self.targets = []
        self.results = None
        self._t_dict: Union[None, Dict[str, uuid.UUID]] = None

        self._target_dump_file = 'rr.targets'
        self._result_dump_file = 'rr.result'
        self.process: QtCore.QProcess = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(
            self._process_output
        )
        self.process.finished.connect(
            self.retrieve_results
        )

        self._total_progress = 0
        self._total_maximum = 0

    def _process_output(self) -> None:
        data_bytes: QtCore.QByteArray = self.process.readAllStandardOutput()
        text = bytes(data_bytes).decode()
        for line in text.splitlines():
            simplified = ' '.join(line.lower().strip().split())
            if ("finished in" in simplified):
                self._total_progress += 1
                self.progress.emit(self._total_progress)
            elif "computing redshifts took:" in simplified:
                self.progress.emit(self._total_maximum)

        self.message.emit(str(text))

    def retrieve_results(self) -> None:
        try:
            with open(self._result_dump_file, 'rb') as f:
                self.results = pickle.load(f)
        except Exception:
            self.results = None
        self.finished.emit()

    def run_redrock(self, spectra: Dict[uuid.UUID, Spectrum1D]) -> None:
        self.reset.emit()
        self.maximum.emit(len(spectra))
        self.message.emit(self.qapp.tr("Building targets"))
        self.targets, self._t_dict = backends.rrock.build_redrock_targets(
            spectra,
            progress_callback=self.progress.emit
        )

        self.reset.emit()
        self.maximum.emit(0)
        backends.rrock.dump_targets(self.targets, self._target_dump_file)

        self._total_maximum = 22
        self._total_progress = 0
        self.maximum.emit(self._total_maximum)
        self.message.emit(self.qapp.tr("Running redrock backend"))
        self.process.start(
            sys.executable,
            [
                backends.rrock.__file__,
                self._target_dump_file,
                self._result_dump_file
            ]
        )


class GlobalState(Enum):
    READY = 0
    WAITING = 1
    SELECT_LINE_MANUAL = 2
    SAVE_OBJECT_STATE = 3
    LOAD_OBJECT_STATE = 4
    REUQUEST_CANCEL = 5


class GuiApp:
    """General class for the main GUI."""

    def __init__(self, qt_backend: str) -> None:
        self.qapp: QtWidgets.QApplication = getQApp()

        self.open_spectra: Dict[uuid.UUID, Spectrum1D] = {}
        self.open_spectra_files: Dict[uuid.UUID, str] = {}
        self.open_spectra_items: Dict[
            uuid.UUID, QtWidgets.QListWidgetItem
        ] = {}
        self.current_uuid: Optional[uuid.UUID] = None
        self.object_state_dict: Dict[uuid.UUID, Any] = {}
        self.current_project_file_path: Optional[str] = None

        self.global_state: Enum = GlobalState.READY

        self.qf_color: Dict[int, str] = {
            0: "#FFFFFF",
            1: "#55940a00",
            2: "#55945400",
            3: "#55176601",
            4: "#55012d66"
        }

        self.main_wnd: QtWidgets.QMainWindow = loadUiWidget(
            "main_window.ui", qt_backend=qt_backend
        )
        self.main_wnd.closeEvent = self.closeEvent

        self.msgBox: QtWidgets.QMessageBox = QtWidgets.QMessageBox(
            parent=self.main_wnd
        )

        # Status Bar
        self.mousePosLabel: QtWidgets.QLabel = QtWidgets.QLabel("")

        # Global cancel button
        self.cancel_button: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self.cancel_button.setText("Cancel")
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(
            self.requestCancelCurrentOperation
        )

        # Globsal progress bar
        self.pbar: QtWidgets.QProgressBar = QtWidgets.QProgressBar()
        self.pbar.hide()

        self.statusbar: QtWidgets.QStatusBar = self.main_wnd.statusBar()
        if self.statusbar is None:
            self.statusbar = QtWidgets.QStatusBar()
            self.main_wnd.setStatusBar(self.statusbar)

        self.statusbar.addPermanentWidget(self.pbar)
        self.statusbar.addPermanentWidget(self.cancel_button)
        self.statusbar.addPermanentWidget(self.mousePosLabel)

        # Fill single line combo box
        for lam, line_name, _ in lines.RESTFRAME_LINES:
            text = f"{line_name} - {lam:.2f} A"
            self.main_wnd.single_line_combo_box.addItem(text, lam)
        self.main_wnd.single_line_combo_box.currentIndexChanged.connect(
            self.setCurrentObjectRedshiftFromSingleLine
        )

        # Check for RedRock.
        if not backends.HAS_REDROCK:
            self.main_wnd.redrock_run_button.setEnabled(False)
            self.main_wnd.redrock_text_edit.setText(
                "*** WARNING ***\n"
                "\n"
                "Cannot load redrock python module.\n"
                "Please, check if redrock is correctly installed!\n"
            )

            self.redrock_handler = None
        else:
            self.main_wnd.redrock_run_button.clicked.connect(
                self.doStartRedrock
            )

            self.redrock_handler = QRedrockHandler()
            self.redrock_handler.reset.connect(
                self.main_wnd.redrock_progress_bar.reset
            )
            self.redrock_handler.maximum.connect(
                self.main_wnd.redrock_progress_bar.setMaximum
            )
            self.redrock_handler.progress.connect(
                self.main_wnd.redrock_progress_bar.setValue
            )
            self.redrock_handler.message.connect(
                self.main_wnd.redrock_text_edit.append
            )
            self.redrock_handler.finished.connect(
                self.collectRedrockResults
            )

        self.main_wnd.redrock_current_radio.setEnabled(False)

        # QChartView widget for flux

        self.flux_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.flux_group_box
        )
        self.flux_chart_view.master = True
        self.flux_chart_view.setObjectName("flux_chart_view")
        self.flux_chart_view.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing
        )
        self.flux_chart_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.flux_chart_view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.flux_chart_view.setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view.setRubberBand(
            QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

        self.main_wnd.flux_widget_layout.addWidget(self.flux_chart_view)

        # QChartView widget for variance

        self.var_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.other_charts_tab_widget
        )
        self.var_chart_view.vertical_lock = True
        self.var_chart_view.setObjectName("var_chart_view")
        self.var_chart_view.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing
        )
        self.var_chart_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.var_chart_view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.var_chart_view.setContentsMargins(0, 0, 0, 0)
        self.var_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.var_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.var_chart_view.setRubberBand(
            QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.main_wnd.var_widget_layout.addWidget(self.var_chart_view)

        # QChartView widget for resolution curve

        self.wdisp_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.other_charts_tab_widget
        )
        self.wdisp_chart_view.vertical_lock = True
        self.wdisp_chart_view.setObjectName("var_chart_view")
        self.wdisp_chart_view.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing
        )
        self.wdisp_chart_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.wdisp_chart_view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.wdisp_chart_view.setContentsMargins(0, 0, 0, 0)
        self.wdisp_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.wdisp_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.wdisp_chart_view.setRubberBand(
            QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.main_wnd.wdisp_widget_layout.addWidget(self.wdisp_chart_view)

        # QChartView widget for sky

        self.sky_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.other_charts_tab_widget
        )
        self.sky_chart_view.vertical_lock = True
        self.sky_chart_view.setObjectName("var_chart_view")
        self.sky_chart_view.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing
        )
        self.sky_chart_view.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.sky_chart_view.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.sky_chart_view.setContentsMargins(0, 0, 0, 0)
        self.sky_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.sky_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.sky_chart_view.setRubberBand(
            QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.main_wnd.sky_widget_layout.addWidget(self.sky_chart_view)

        # Add siblings to main flux QChartView

        self.flux_chart_view.addSibling(self.var_chart_view)
        self.flux_chart_view.addSibling(self.wdisp_chart_view)
        self.flux_chart_view.addSibling(self.sky_chart_view)

        # Connect signals

        self.main_wnd.spec_list_widget.currentItemChanged.connect(
            self.currentSpecItemChanged
        )

        self.main_wnd.spec_list_widget.itemDoubleClicked.connect(
            self.toggleSimilarSpecItems
        )

        self.main_wnd.toggle_done_button.clicked.connect(
            self.doToggleDone
        )

        self.main_wnd.toggle_similar_button.clicked.connect(
            self.doToggleSimilar
        )

        self.main_wnd.toggle_all_button.clicked.connect(
            self.doToggleAll
        )

        self.main_wnd.remove_selected_button.clicked.connect(
            self.doRemoveSelectedSpecItems
        )

        self.main_wnd.remove_spec_button.clicked.connect(
            self.doRemoveCurrentSpecItems
        )

        self.main_wnd.export_zcat_button.clicked.connect(
            self.doExportZcat
        )

        self.main_wnd.import_zcat_button.clicked.connect(
            self.doImportZcat
        )

        self.main_wnd.action_import_spectra.triggered.connect(
            self.doImportSpectra
        )
        self.main_wnd.action_zoom_in.triggered.connect(
            self.doZoomIn
        )
        self.main_wnd.action_zoom_out.triggered.connect(
            self.doZoomOut
        )
        self.main_wnd.action_zoom_fit.triggered.connect(
            self.doZoomReset
        )
        self.main_wnd.action_save_project_as.triggered.connect(
            self.doSaveProjectAs
        )
        self.main_wnd.action_save_project.triggered.connect(
            self.doSaveProject
        )
        self.main_wnd.action_open_project.triggered.connect(
            self.doOpenProject
        )
        self.main_wnd.actionNewProject.triggered.connect(
            self.doNewProject
        )

        self.flux_chart_view.onMouseMoveSeries.connect(
            self._updateMouseLabelFromEvent
        )
        self.flux_chart_view.onMousePressSeries.connect(self.mousePressedFlux)

        self.var_chart_view.onMouseMoveSeries.connect(
            self._updateMouseLabelFromEvent
        )

        self.main_wnd.smoothing_check_box.stateChanged.connect(
            self.toggleSmothing
        )
        self.main_wnd.smoothing_dspinbox.valueChanged.connect(
            self.setSmoothingFactor
        )

        self.main_wnd.lines_auto_button.clicked.connect(
            self.doIdentifyLines
        )

        self.main_wnd.lines_match_list_widget.currentRowChanged.connect(
            self.setCurrentObjectRedshiftFromLines
        )

        self.main_wnd.add_line_button.clicked.connect(
            self.doAddNewLine
        )

        self.main_wnd.delete_line_button.clicked.connect(
            self.doDeleteCurrentLine
        )

        self.main_wnd.delete_lines_button.clicked.connect(
            self.doDeleteAllLines
        )

        self.main_wnd.match_lines_button.clicked.connect(
            self.doRedshiftFromLines
        )

        self.main_wnd.show_lines_check_box.toggled.connect(
            self.flux_chart_view.setLinesVisible
        )

        self.main_wnd.z_dspinbox.valueChanged.connect(
            self.setCurrentObjectRedshift
        )

        self.main_wnd.qflag_combo_box.currentIndexChanged.connect(
            self.currentQualityFlagChanged
        )

        self.main_wnd.show_lines_combo_box.currentIndexChanged.connect(
            self.setShowLinesType
        )

        self.newProject()

    def _backup_current_object_state(self) -> None:
        """
        Backup the program state for the current object

        Returns
        -------
        None

        """
        if self.current_uuid is None:
            return

        self.global_state = GlobalState.SAVE_OBJECT_STATE

        lines_list = []
        for row_index in range(self.main_wnd.lines_table_widget.rowCount()):
            w_item = self.main_wnd.lines_table_widget.item(row_index, 0)
            line_info = {
                'row': row_index,
                'data': float(w_item.data(QtCore.Qt.ItemDataRole.UserRole)),
                'text': w_item.text(),
                'checked': w_item.checkState()
            }
            lines_list.append(line_info)

        redshifts_form_lines = []
        for row_index in range(self.main_wnd.lines_match_list_widget.count()):
            z_item = self.main_wnd.lines_match_list_widget.item(row_index)
            z_info = {
                'row': row_index,
                'text': z_item.text(),
                'data': z_item.data(QtCore.Qt.ItemDataRole.UserRole)
            }
            redshifts_form_lines.append(z_info)

        obj_state = {
            'redshift': self.main_wnd.z_dspinbox.value(),
            'quality_flag': self.main_wnd.qflag_combo_box.currentIndex(),
            'lines': {
                'list': lines_list,
                'redshifts': redshifts_form_lines
            }
        }

        self.object_state_dict[self.current_uuid] = obj_state
        self.global_state = GlobalState.READY

    def _update_spec_item_qf(self, item_uuid: uuid.UUID, qf: float) -> None:
        item: QtWidgets.QListWidgetItem = self.open_spectra_items[item_uuid]
        item.setBackground(QtGui.QColor(self.qf_color[qf]))

    def _lock(self, *args, **kwargs) -> None:
        self.main_wnd.spec_group_box.setEnabled(False)
        self.main_wnd.red_group_box.setEnabled(False)
        self.main_wnd.info_group_box.setEnabled(False)
        self.main_wnd.plot_group_box.setEnabled(False)
        self.flux_chart_view.setRubberBand(
            QtCharts.QChartView.RubberBand.NoRubberBand
        )

    def _restore_object_state(self, obj_uuid: uuid.UUID) -> None:
        """
        Restore the programs state for a given object.

        Parameters
        ----------
        obj_uuid : uuid.UUID
            The UUID of the object

        Returns
        -------
        None

        """
        self.global_state = GlobalState.LOAD_OBJECT_STATE

        # Clear current content of the widgets
        self.main_wnd.lines_table_widget.setRowCount(0)
        self.main_wnd.lines_match_list_widget.clear()

        # If object state has not been previously saved, then return
        if obj_uuid not in self.object_state_dict:
            self.global_state = GlobalState.READY
            return

        # otherwise restore any old previously saved data
        old_state = self.object_state_dict[obj_uuid]

        lines = old_state['lines']['list']
        self.main_wnd.lines_table_widget.setRowCount(len(lines))
        for line_info in lines:
            w_item = QtWidgets.QTableWidgetItem(line_info['text'])
            w_item.setData(QtCore.Qt.ItemDataRole.UserRole, line_info['data'])
            w_item.setCheckState(line_info['checked'])
            w_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable |
                QtCore.Qt.ItemFlag.ItemIsEnabled |
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )

            r_item = QtWidgets.QTableWidgetItem("")
            r_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable |
                QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            r_item.setData(QtCore.Qt.ItemDataRole.UserRole, 0.0)

            m_item = QtWidgets.QTableWidgetItem("")
            m_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable |
                QtCore.Qt.ItemFlag.ItemIsEnabled
            )

            self.main_wnd.lines_table_widget.setItem(
                line_info['row'], 0, w_item
            )

            self.main_wnd.lines_table_widget.setItem(
                line_info['row'], 1, r_item
            )

            self.main_wnd.lines_table_widget.setItem(
                line_info['row'], 2, m_item
            )

        redshifts_form_lines = old_state['lines']['redshifts']
        for z_info in redshifts_form_lines:
            z_item = QtWidgets.QListWidgetItem(z_info['text'])
            z_item.setData(QtCore.Qt.ItemDataRole.UserRole, z_info['data'])
            self.main_wnd.lines_match_list_widget.insertItem(
                z_info['row'], z_item
            )

        try:
           current_redshift = old_state['redshift']
        except KeyError:
            current_redshift = None

        if current_redshift:
            self.main_wnd.z_dspinbox.setValue(current_redshift)
        else:
            self.main_wnd.z_dspinbox.setValue(0)

        self.global_state = GlobalState.READY

        try:
            quality_flag = old_state['quality_flag']
        except KeyError:
            quality_flag = 0

        self.main_wnd.qflag_combo_box.setCurrentIndex(quality_flag)

        self._update_spec_item_qf(obj_uuid, quality_flag)

    def _unlock(self, *args, **kwargs) -> None:
        self.main_wnd.spec_group_box.setEnabled(True)
        self.main_wnd.red_group_box.setEnabled(True)
        self.main_wnd.info_group_box.setEnabled(True)
        self.main_wnd.plot_group_box.setEnabled(True)
        self.flux_chart_view.setRubberBand(
            QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

    def _updateMouseLabelFromEvent(self, *args) -> None:
        self._updateMouseLabel(args[0][0])

    def _updateMouseLabel(self, mouse_pos: QtCore.QPointF) -> None:
        self.mousePosLabel.setText(f"\u03BB = {mouse_pos.x():.2f}")

    def addLine(self, wavelength: float) -> None:
        """
        Add a new line by selecting its position with the mouse.

        Parameters
        ----------
        wavelength

        Returns
        -------
        None

        """
        lam_item = QtWidgets.QTableWidgetItem(f"{wavelength:.2f} A")
        lam_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsSelectable |
            QtCore.Qt.ItemFlag.ItemIsEnabled |
            QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        lam_item.setCheckState(QtCore.Qt.CheckState.Checked)
        lam_item.setData(QtCore.Qt.ItemDataRole.UserRole, wavelength)

        rest_lam = wavelength / (1 + self.main_wnd.z_dspinbox.value())
        rest_item = QtWidgets.QTableWidgetItem(f"{rest_lam:.2f} A")
        rest_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsSelectable |
            QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        rest_item.setData(QtCore.Qt.ItemDataRole.UserRole, 0.0)

        best_matches = [
            x[1]
            for x in  lines.get_lines(wrange=[rest_lam - 5, rest_lam + 5])
        ]

        match_item = QtWidgets.QTableWidgetItem('; '.join(best_matches))
        match_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsSelectable |
            QtCore.Qt.ItemFlag.ItemIsEnabled
        )

        new_item_row: int = self.main_wnd.lines_table_widget.rowCount()

        self.main_wnd.lines_table_widget.setRowCount(new_item_row + 1)
        self.main_wnd.lines_table_widget.setItem(new_item_row, 0, lam_item)
        self.main_wnd.lines_table_widget.setItem(new_item_row, 1, rest_item)
        self.main_wnd.lines_table_widget.setItem(new_item_row, 2, match_item)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.msgBox.setText(
            self.qapp.tr(
                "Do you want to save the current project before closing?"
            )
        )
        self.msgBox.setWindowTitle(self.qapp.tr("Closing..."))
        self.msgBox.setInformativeText("")
        self.msgBox.setDetailedText("")
        self.msgBox.setIcon(QtWidgets.QMessageBox.Icon.Question)
        self.msgBox.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No |
            QtWidgets.QMessageBox.StandardButton.Cancel
        )
        res = self.msgBox.exec()

        if res == QtWidgets.QMessageBox.StandardButton.Yes:
            # If Yes button is pressed, then accept the event only if the
            # project is correctly saved
            event.setAccepted(self.doSaveProject())
        elif res == QtWidgets.QMessageBox.StandardButton.No:
            event.accept()
        else:
            # Ignore the close event if Cancel button is pressed
            event.ignore()

    def collectRedrockResults(self) -> None:
        self.statusbar.showMessage(self.qapp.tr("Redrock results ready!"))
        res = self.redrock_handler.results

        if res is None:
            return

        zbest = res[1][res[1]['znum'] == 0]

        self._lock()
        self.global_state = GlobalState.LOAD_OBJECT_STATE
        self._backup_current_object_state()
        for row in zbest:
            obj_uuid = self.redrock_handler._t_dict[row['targetid']]

            try:
                info_dict = self.object_state_dict[obj_uuid]
            except KeyError:
                info_dict = {
                    'redshift': None,
                    'quality_flag': 0,
                    'lines': {
                        'list': [],
                        'redshifts': []
                    }
                }
                self.object_state_dict[obj_uuid] = info_dict

            info_dict['redshift'] = row['z']
            if row['zwarn'] != 0:
                self._update_spec_item_qf(obj_uuid, 1)
        self._restore_object_state(self.current_uuid)
        self.global_state = GlobalState.READY
        self._unlock()

    def currentQualityFlagChanged(self, df_index):
        if self.current_uuid is None:
            return
        self._update_spec_item_qf(self.current_uuid, df_index)
        self._backup_current_object_state()

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
        if new_item is None:
            self.main_wnd.redrock_current_radio.setEnabled(False)
            return
        else:
            self.main_wnd.redrock_current_radio.setEnabled(True)

        if self.global_state != GlobalState.READY:
            return

        self._unlock()

        spec_uuid: uuid.UUID = new_item.data(
            QtCore.Qt.ItemDataRole.UserRole
        )

        flux_chart = self.flux_chart_view.chart()
        flux_chart.removeAllSeries()

        var_chart = self.var_chart_view.chart()
        var_chart.removeAllSeries()

        wd_chart = self.wdisp_chart_view.chart()
        wd_chart.removeAllSeries()

        sky_chart = self.sky_chart_view.chart()
        sky_chart.removeAllSeries()

        self._backup_current_object_state()
        if spec_uuid != self.current_uuid:
            # If we actually change the spectrum, then reset the view
            self.current_uuid = spec_uuid
            self._restore_object_state(spec_uuid)

            for ax in flux_chart.axes():
                flux_chart.removeAxis(ax)
            for ax in var_chart.axes():
                var_chart.removeAxis(ax)
            for ax in wd_chart.axes():
                wd_chart.removeAxis(ax)
            for ax in sky_chart.axes():
                sky_chart.removeAxis(ax)

        sp: Spectrum1D = self.open_spectra[spec_uuid]

        wav: np.ndarray = sp.spectral_axis.value
        wav_unit: units.Unit = sp.spectral_axis.unit

        flux: np.ndarray = sp.flux.value
        flux_unit: units.Unit = sp.flux.unit

        var: np.ndarray | None = None
        var_unit: units.Unit | None = None

        wdisp: np.ndarray | None = None
        wdisp_unit: units.Unit | None = None

        sky: np.ndarray | None = None
        sky_unit: units.Unit | None = None

        if isinstance(sp.uncertainty, VarianceUncertainty):
            var = sp.uncertainty.array
            var_unit = sp.uncertainty.unit
        elif isinstance(sp.uncertainty, InverseVariance):
            var = 1 / sp.uncertainty.array
            var_unit = 1 / sp.uncertainty.unit
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = sp.uncertainty.array ** 2
            var_unit = sp.uncertainty.unit ** 2

        try:
            wd = sp.wd
        except AttributeError:
            pass
        else:
            if sp.wd is not None:
                wdisp = sp.wd.value
                wdisp_unit = sp.wd.unit

        try:
            sky = sp.sky
        except AttributeError:
            pass
        else:
            if sp.sky is not None:
                sky = sp.sky.value
                sky_unit = sp.sky.unit

        flux_series = values2series(wav, flux, self.qapp.tr("Flux"))
        flux_chart.addSeries(flux_series)

        if not flux_chart.axes():
            flux_axis_x = QtCharts.QValueAxis()
            flux_axis_y = QtCharts.QValueAxis()
            var_axis_x = QtCharts.QValueAxis()
            var_axis_y = QtCharts.QValueAxis()
            wd_axis_x = QtCharts.QValueAxis()
            wd_axis_y = QtCharts.QValueAxis()
            sky_axis_x = QtCharts.QValueAxis()
            sky_axis_y = QtCharts.QValueAxis()

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
            wd_chart.addAxis(
                wd_axis_x, QtCore.Qt.AlignmentFlag.AlignBottom
            )
            wd_chart.addAxis(
                wd_axis_y, QtCore.Qt.AlignmentFlag.AlignLeft
            )
            sky_chart.addAxis(
                sky_axis_x, QtCore.Qt.AlignmentFlag.AlignBottom
            )
            sky_chart.addAxis(
                sky_axis_y, QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            flux_axis_x = flux_chart.axes()[0]
            flux_axis_y = flux_chart.axes()[1]
            var_axis_x = var_chart.axes()[0]
            var_axis_y = var_chart.axes()[1]
            wd_axis_x = wd_chart.axes()[0]
            wd_axis_y = wd_chart.axes()[1]
            sky_axis_x = sky_chart.axes()[0]
            sky_axis_y = sky_chart.axes()[1]

        flux_axis_x.setTickInterval(500)
        flux_axis_x.setLabelFormat("%.2f")
        flux_axis_x.setTitleText(str(wav_unit))

        flux_axis_y.setLabelFormat("%.2f")
        flux_axis_y.setTitleText(str(flux_unit))

        flux_series.attachAxis(flux_axis_x)
        flux_series.attachAxis(flux_axis_y)

        smoothing_check_state = self.main_wnd.smoothing_check_box.checkState()
        if smoothing_check_state == QtCore.Qt.CheckState.Checked:
            smoothing_factor = self.main_wnd.smoothing_dspinbox.value()

            flux_series.setOpacity(0.2)
            smoothing_sigma = len(flux) / (1 + 2 * smoothing_factor)
            smoothed_flux = utils.smooth_fft(flux, sigma=smoothing_sigma)
            smoothed_flux_series = values2series(
                wav, smoothed_flux, self.qapp.tr("Smoothed flux")
            )

            pen: QtGui.QPen = smoothed_flux_series.pen()
            pen.setColor(QtGui.QColor("orange"))
            pen.setWidth(2)
            smoothed_flux_series.setPen(pen)

            flux_chart.addSeries(smoothed_flux_series)
            smoothed_flux_series.attachAxis(flux_axis_x)
            smoothed_flux_series.attachAxis(flux_axis_y)

        if var is None:
            self.main_wnd.container_tab_var.setEnabled(False)
        else:
            self.main_wnd.container_tab_var.setEnabled(True)
            var_series = values2series(wav, var, self.qapp.tr("Variance"))
            var_chart.addSeries(var_series)

            var_axis_x.setTickInterval(500)
            var_axis_x.setLabelFormat("%.2f")
            var_axis_x.setTitleText(str(wav_unit))

            var_axis_y.setTickCount(10)
            var_axis_y.setLabelFormat("%.2f")
            var_axis_y.setTitleText(str(var_unit))

            var_series.attachAxis(var_axis_x)
            var_series.attachAxis(var_axis_y)

        if wdisp is None:
            self.main_wnd.container_tab_wd.setEnabled(False)
        else:
            self.main_wnd.container_tab_wd.setEnabled(True)
            wd_series = values2series(wav, wdisp, self.qapp.tr("Reso. curve"))
            wd_chart.addSeries(wd_series)

            wd_axis_x.setTickInterval(500)
            wd_axis_x.setLabelFormat("%.2f")
            wd_axis_x.setTitleText(str(wav_unit))

            wd_axis_y.setTickCount(10)
            wd_axis_y.setLabelFormat("%.2f")
            wd_axis_y.setTitleText(str(wdisp_unit))

            wd_series.attachAxis(wd_axis_x)
            wd_series.attachAxis(wd_axis_y)

        if sky is None:
            self.main_wnd.container_tab_sky.setEnabled(False)
        else:
            self.main_wnd.container_tab_sky.setEnabled(True)
            sky_series = values2series(wav, sky, self.qapp.tr("Sky"))
            sky_chart.addSeries(sky_series)

            sky_axis_x.setTickInterval(500)
            sky_axis_x.setLabelFormat("%.2f")
            sky_axis_x.setTitleText(str(wav_unit))

            sky_axis_y.setTickCount(10)
            sky_axis_y.setLabelFormat("%.2f")
            sky_axis_y.setTitleText(str(sky_unit))

            sky_series.attachAxis(sky_axis_x)
            sky_series.attachAxis(sky_axis_y)

        flux_chart.setContentsMargins(0, 0, 0, 0)
        flux_chart.setBackgroundRoundness(0)
        flux_chart.legend().hide()

        var_chart.setContentsMargins(0, 0, 0, 0)
        var_chart.setBackgroundRoundness(0)
        var_chart.legend().hide()

        wd_chart.setContentsMargins(0, 0, 0, 0)
        wd_chart.setBackgroundRoundness(0)
        wd_chart.legend().hide()

        sky_chart.setContentsMargins(0, 0, 0, 0)
        sky_chart.setBackgroundRoundness(0)
        sky_chart.legend().hide()

    def doAddNewLine(self, *args, **kwargs) -> None:
        """
        Tell the main app to identify a new line

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        None

        """
        # Do nothing if no spectrum is currently selected
        if self.current_uuid is None:
            return
        self._lock()
        self.main_wnd.spec_group_box.setEnabled(False)
        self.global_state = GlobalState.SELECT_LINE_MANUAL
        self.qapp.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

    def doDeleteAllLines(self, *args, **kwargs) -> None:
        """
        Delete all identified lines.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        None
        """
        self.main_wnd.lines_table_widget.setRowCount(0)

    def doDeleteCurrentLine(self, *args, **kwargs) -> None:
        """
        Delete the current selected line from the table.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        None

        """
        # Do nothing if no spectrum is currently selected
        if self.current_uuid is None:
            return

        self.main_wnd.lines_table_widget.removeRow(
            self.main_wnd.lines_table_widget.currentRow()
        )

    def doExportZcat(self, *args, **kwargs) -> None:
        dest_file_path, file_type = QtWidgets.QFileDialog.getSaveFileName(
            self.main_wnd,
            self.qapp.tr("Export redshift catalogue to file"),
            '.',
            (
                f"{self.qapp.tr('FITS table')} (*.fits *.fit);;"
                f"{self.qapp.tr('CSV table')} (*.csv);;"
                f"{self.qapp.tr('VO table')} (*.votable);;"
            )
        )

        try:
            self.exportZcat(dest_file_path, file_type)
        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                 self.qapp.tr(
                     "An error has occurred while saving the cataglogue."
                 )
            )
            self.msgBox.setInformativeText('')
            self.msgBox.setDetailedText(str(exc))
            self.msgBox.setIcon(
                QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()

        self.statusbar.showMessage(self.qapp.tr("Redshift catalogue saved"))
        self.current_project_file_path = dest_file_path

    def doIdentifyLines(self, *args, **kwargs) -> None:
        """
        Automagically identify lines in the current spectrum.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        None

        """
        if self.current_uuid is None:
            return

        sp: Spectrum1D = self.open_spectra[self.current_uuid]

        wav: np.ndarray = sp.spectral_axis.value
        flux: np.ndarray = sp.flux.value
        var: np.ndarray | None = None

        if isinstance(sp.uncertainty, VarianceUncertainty):
            var = sp.uncertainty.array
        if isinstance(sp.uncertainty, InverseVariance):
            var = 1 / sp.uncertainty.array
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = sp.uncertainty.array ** 2

        my_lines: List[
            Tuple[int, float, float, float]
        ] = lines.get_spectrum_lines(
            wavelengths=wav,
            flux=flux,
            var=var
        )

        self.main_wnd.lines_table_widget.setRowCount(0)
        self.main_wnd.lines_table_widget.setRowCount(len(my_lines))
        for j, (k, w, l, h) in enumerate(my_lines):
            new_item = QtWidgets.QTableWidgetItem(f"{w:.2f} A")
            new_item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsSelectable |
                QtCore.Qt.ItemFlag.ItemIsEnabled |
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            new_item.setData(QtCore.Qt.ItemDataRole.UserRole, w)
            new_item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.main_wnd.lines_table_widget.setItem(j, 0, new_item)

    def doImportZcat(self, *args, **kwargs) -> None:
        proj_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_wnd,
            self.qapp.tr("Import a zcat"),
            '.',
            (
                f"{self.qapp.tr('FITS table')} (*.fits *.fit);;"
                f"{self.qapp.tr('CSV table')} (*.csv);;"
                f"{self.qapp.tr('VO table')} (*.votable);;"
                f"{self.qapp.tr('All files')} (*.*);;"
            )
        )

        if not proj_file_path:
            return

        try:
            self.importZcat(proj_file_path)
        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                self.qapp.tr(
                    "An error has occurred while importing the catalogue."
                )
            )
            self.msgBox.setInformativeText('')
            self.msgBox.setDetailedText(str(exc))
            self.msgBox.setIcon(
                QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()

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

        excetpion_tracker: Dict[uuid.UUID, Tuple[str, str]] = {}

        self._lock()
        self.global_state = GlobalState.WAITING

        n_files = len(file_list)

        self.pbar.setMaximum(n_files)
        self.pbar.show()
        self.cancel_button.show()
        for j, file in enumerate(file_list):
            if self.global_state == GlobalState.REUQUEST_CANCEL:
                break

            self.pbar.setValue(j + 1)
            self.statusbar.showMessage(
                self.qapp.tr("Loading file") + f"{j + 1:d}/{n_files}..."
            )
            self.qapp.processEvents()

            item_uuid: uuid.UUID = uuid.uuid4()

            # Check for a possible collision, even if it should neve happem
            while item_uuid in self.open_spectra_files.keys():
                item_uuid = uuid.uuid4()

            try:
                sp: Spectrum1D = loaders.read(file)
            except Exception as exc:
                excetpion_tracker[item_uuid] = (file, str(exc))
                continue

            new_item = QtWidgets.QListWidgetItem(os.path.basename(file))
            new_item.setCheckState(QtCore.Qt.CheckState.Checked)
            new_item.setToolTip(file)
            new_item.setData(QtCore.Qt.ItemDataRole.UserRole, item_uuid)

            info_dict: Dict[str, Any] = {
                'redshift': None,
                'quality_flag': 0,
                'lines': {
                    'list': [],
                    'redshifts': []
                }
            }

            self.open_spectra[item_uuid] = sp
            self.open_spectra_files[item_uuid] = os.path.abspath(file)
            self.open_spectra_items[item_uuid] = new_item
            self.object_state_dict[item_uuid] = info_dict
            self.main_wnd.spec_list_widget.addItem(new_item)
            self.qapp.processEvents()

        self.cancel_button.hide()
        self.pbar.hide()
        self._unlock()
        self.global_state = GlobalState.READY

        if excetpion_tracker:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                self.qapp.tr(
                    "An error has occurred while loading the project: one or "
                    "more files could not be loaded."
                )
            )
            self.msgBox.setDetailedText(
                "\n\n".join(
                    [
                        f"uuid: {exc_uuid}\n"
                        f"file: {exc_data[0]}\n"
                        f"reason: {exc_data[1]}"
                        for exc_uuid, exc_data in excetpion_tracker.items()
                    ]
                )
            )
            self.msgBox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            self.msgBox.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()

    def doNewProject(self, *args, **kwargs) -> None:
        self.newProject()

    def doOpenProject(self, *args, **kwargs) -> bool:
        proj_file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main_wnd,
            self.qapp.tr("Load a project from file"),
            '.',
            (
                f"{self.qapp.tr('Project file')} (*.json);;"
                f"{self.qapp.tr('All files')} (*.*);;"
            )
        )

        if not proj_file_path:
            return False

        self.newProject()

        try:
            self.openProject(file_name=proj_file_path)
        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                 self.qapp.tr(
                     "An error has occurred while opening the project."
                 )
            )
            self.msgBox.setInformativeText('')
            self.msgBox.setDetailedText(str(exc))
            self.msgBox.setIcon(
                QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
            return False

        self.current_project_file_path = proj_file_path
        return True

    def doRedshiftFromLines(self, *args, **kwargs) -> None:
        """
        Compute the redshift by matching identified lines.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        None
        """
        line_table: QtWidgets.QTableWidget = self.main_wnd.lines_table_widget
        z_list: QtWidgets.QListWidget = self.main_wnd.lines_match_list_widget
        tol: float = self.main_wnd.lines_tol_dspinbox.value()
        z_min: float = self.main_wnd.z_min_dspinbox.value()
        z_max: float = self.main_wnd.z_max_dspinbox.value()

        # Build a list of selected lines to be used.
        # If no lines are selected, then use all lines.
        lines_lam = []
        for row_index in range(line_table.rowCount()):
            item = line_table.item(row_index, 0)
            if item is None:
                continue
            elif item.checkState() != QtCore.Qt.CheckState.Checked:
                # Ignore lines that are not selected
                continue

            lines_lam.append(float(item.data(QtCore.Qt.ItemDataRole.UserRole)))

        res = lines.get_redshift_from_lines(
            lines_lam, z_min=z_min, z_max=z_max, tol=tol
        )

        if res is None:
            return

        z_list.clear()
        for z, prob in zip(res[0], res[1]):
            new_z_item = QtWidgets.QListWidgetItem(f"z={z:.4f} (p={prob:.4f})")
            new_z_item.setData(QtCore.Qt.ItemDataRole.UserRole, (z, prob))
            z_list.addItem(new_z_item)

    def doRemoveCurrentSpecItems(self) -> None:
        if self.main_wnd.spec_list_widget.count() == 0:
            return

        current_row = self.main_wnd.spec_list_widget.currentRow()
        item = self.main_wnd.spec_list_widget.item(current_row)
        item_uuid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.main_wnd.spec_list_widget.takeItem(current_row)
        self.open_spectra_items.pop(item_uuid)
        self.open_spectra_files.pop(item_uuid)

        try:
            self.object_state_dict.pop(item_uuid)
        except KeyError:
            pass

        del item

        if self.main_wnd.spec_list_widget.count() == 0:
            self.newProject()

    def doRemoveSelectedSpecItems(self) -> None:
        if self.main_wnd.spec_list_widget.count() == 0:
            return

        for row in range(self.main_wnd.spec_list_widget.count(), 0, -1):
            item = self.main_wnd.spec_list_widget.item(row - 1)

            if item is None:
                continue
            elif item.checkState() != QtCore.Qt.CheckState.Checked:
                continue

            item_uuid = item.data(QtCore.Qt.ItemDataRole.UserRole)
            self.main_wnd.spec_list_widget.takeItem(row - 1)
            self.open_spectra_items.pop(item_uuid)
            self.open_spectra_files.pop(item_uuid)

            try:
                self.object_state_dict.pop(item_uuid)
            except KeyError:
                pass

            del item

        if self.main_wnd.spec_list_widget.count() == 0:
            self.newProject()

    def doSaveProject(self, *args, **kwargs) -> bool:
        """
        Save the current project.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        return self.doSaveProjectAs(dest=self.current_project_file_path)

    def doSaveProjectAs(
            self,
            dest: Optional[str] = None,
            *args, **kwargs
    ) -> bool:
        """
        Save the current project to a file
        Parameters
        ----------
        dest
        args
        kwargs

        Returns
        -------
            True if the project is saved correctly, False otherwise.

        """
        if (dest is None) or (not os.path.exists(dest)):
            dest_file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.main_wnd,
                self.qapp.tr("Save project to file"),
                '.',
                (
                    f"{self.qapp.tr('Project file')} (*.json);;"
                    f"{self.qapp.tr('All files')} (*.*);;"
                )
            )

            if not dest_file_path:
                return False
        else:
            dest_file_path = dest

        try:
            self.saveProject(dest_file_path)
        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                 self.qapp.tr(
                     "An error has occurred while saving the project."
                 )
            )
            self.msgBox.setInformativeText('')
            self.msgBox.setDetailedText(str(exc))
            self.msgBox.setIcon(
                QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
            return False

        self.statusbar.showMessage(self.qapp.tr("Project saved"))
        self.current_project_file_path = dest_file_path
        return True

    def doStartRedrock(self, *args, **kwargs) -> None:
        self.main_wnd.redrock_text_edit.clear()
        self.statusbar.showMessage("Running redrock backend...")

        for_rr: Dict[uuid.UUID, Spectrum1D] = {}
        if self.main_wnd.redrock_all_radio.isChecked():
            for_rr = self.open_spectra
        elif self.main_wnd.redrock_selected_radio.isChecked():
            item: QtWidgets.QListWidgetItem
            for row in range(self.main_wnd.spec_list_widget.count()):
                item = self.main_wnd.spec_list_widget.item(row)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    item_uuid = item.data(QtCore.Qt.ItemDataRole.UserRole)
                    for_rr[item_uuid] = self.open_spectra[item_uuid]
        else:
            for_rr[self.current_uuid] = self.open_spectra[self.current_uuid]

        self.redrock_handler.run_redrock(for_rr)

    def doToggleDone(self) -> None:
        check_state = None
        for row in range(self.main_wnd.spec_list_widget.count()):
            item = self.main_wnd.spec_list_widget.item(row)
            item_uuid = item.data(QtCore.Qt.ItemDataRole.UserRole)

            if item_uuid not in self.object_state_dict:
                continue

            if self.object_state_dict[item_uuid]['quality_flag'] == 0:
                continue

            if check_state is None:
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    check_state = QtCore.Qt.CheckState.Unchecked
                else:
                    check_state = QtCore.Qt.CheckState.Checked

            item.setCheckState(check_state)

    def doToggleSimilar(self) -> None:
        self.toggleSimilarSpecItems(
            self.main_wnd.spec_list_widget.currentItem()
        )

    def doToggleAll(self) -> None:
        item = self.main_wnd.spec_list_widget.currentItem()
        if item is None:
            return

        if item.checkState() == QtCore.Qt.CheckState.Checked:
            ref_check_state = QtCore.Qt.CheckState.Unchecked
        else:
            ref_check_state = QtCore.Qt.CheckState.Checked

        for row in range(self.main_wnd.spec_list_widget.count()):
            other_item = self.main_wnd.spec_list_widget.item(row)
            other_item.setCheckState(ref_check_state)

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
        self.flux_chart_view.zoomIn()

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
        self.flux_chart_view.zoomOut()

    def doZoomReset(self, *arg, **kwargs) -> None:
        """
        Reset the zoom for the flux chart and all its siblings.
        Parameters
        ----------
        arg
        kwargs

        Returns
        -------
        None

        """
        self.flux_chart_view.zoomReset()

    def exportZcat(self, dest_file, file_type) -> None:
        zcat_tbl = Table(
            names=['INDEX', 'SPEC_FILE', 'Z', 'QF', 'UUID'],
            dtype=[int, str, float, int, str]
        )

        for j, (o_uuid, o_path) in enumerate(self.open_spectra_files.items()):
            spec_file = os.path.basename(o_path)

            try:
                info_dict = self.object_state_dict[o_uuid]
            except KeyError:
                redshift = -99.0
                quality_flag = 0
            else:
                if info_dict['redshift'] is None:
                    redshift = -99.0
                else:
                    redshift = info_dict['redshift']
                quality_flag = info_dict['quality_flag']

            new_row = (j, spec_file, redshift, quality_flag, o_uuid.hex)
            zcat_tbl.add_row(new_row)

        # Get valid file extensions
        valied_exts = (file_type.split('(')[1]).split(')')[0]
        valied_exts = valied_exts.replace('*', '').split()

        # get destination file extension
        dest_ext = os.path.splitext(dest_file)[1]

        # correct for mismatching extension
        if dest_ext not in valied_exts:
            dest_file = dest_file + valied_exts[0]

        zcat_tbl.write(dest_file, overwrite=True)

    def importZcat(self, catalogue_file) -> None:
        zcat_tbl = Table.read(catalogue_file)

        needed_cols = ['SPEC_FILE', 'Z']

        for col in needed_cols:
            if col not in zcat_tbl.colnames:
                raise ValueError(
                    self.qapp.tr(
                        "Catalogue format not supported"
                    )
                )

        self._lock()
        self.global_state = GlobalState.LOAD_OBJECT_STATE
        self.pbar.show()
        self.cancel_button.show()
        self.pbar.setMaximum(len(zcat_tbl))

        matched_uuids = []

        for j, row in enumerate(zcat_tbl):
            self.pbar.setValue(j + 1)

            if self.global_state == GlobalState.REUQUEST_CANCEL:
                break

            spec_file = str(row['SPEC_FILE']).strip()

            for obj_uuid, obj_file in self.open_spectra_files.items():
                if obj_uuid in matched_uuids:
                    continue
                elif os.path.basename(obj_file) == spec_file:
                    matched_uuids.append(obj_uuid)
                    try:
                        info_dict = self.object_state_dict[obj_uuid]
                    except KeyError:
                        info_dict = {
                            'redshift': None,
                            'quality_flag': 0,
                            'lines': {
                                'list': [],
                                'redshifts': []
                            }
                        }
                        self.object_state_dict[obj_uuid] = info_dict

                    if row['Z'] == -99:
                        info_dict["redshift"] = None
                    else:
                        info_dict["redshift"] = row['Z']

                    try:
                        info_dict["quality_flag"] = row['QF']
                    except KeyError:
                        pass
                    else:
                        self._update_spec_item_qf(obj_uuid, row['QF'])
                    break

        self.global_state = GlobalState.READY
        self.pbar.hide()
        self.cancel_button.hide()
        self._unlock()

    def mousePressedFlux(self, args) -> None:
        """
        Handle mouse button pressed events.

        Parameters
        ----------
        args

        Returns
        -------
        None

        """
        data_pos: QtCore.QPointF = args[0]
        event: QtGui.QMouseEvent = args[1]
        if self.global_state == GlobalState.SELECT_LINE_MANUAL:
            self.addLine(data_pos.x())
            self.global_state = GlobalState.READY
            self.qapp.restoreOverrideCursor()
            self._unlock()

    def newProject(self) -> None:
        self.open_spectra_files = {}
        self.open_spectra = {}
        self.object_state_dict = {}
        self.current_uuid = None

        self.main_wnd.spec_list_widget.clear()
        self.main_wnd.lines_match_list_widget.clear()
        self.main_wnd.lines_table_widget.setRowCount(0)
        self.main_wnd.smoothing_dspinbox.setValue(3.0)
        self.main_wnd.smoothing_check_box.setCheckState(
            QtCore.Qt.CheckState.Unchecked
        )
        self.main_wnd.obj_prop_table_widget.setRowCount(0)

        flux_chart = self.flux_chart_view.chart()
        flux_chart.removeAllSeries()
        for ax in flux_chart.axes():
            flux_chart.removeAxis(ax)

        var_chart = self.var_chart_view.chart()
        var_chart.removeAllSeries()
        for ax in var_chart.axes():
            var_chart.removeAxis(ax)

        self._lock()
        self.global_state = GlobalState.READY
        self.statusbar.showMessage(self.qapp.tr("New project created"))

    def openProject(self, file_name: str) -> None:
        """
        Load the project from a file

        Parameters
        ----------
        file_name : str
            The project file path.

        Returns
        -------
        None

        """
        with open(file_name, 'r') as f:
            serialized_dict: Dict[str, Any] = json.load(f)

        self._lock()
        self.global_state = GlobalState.WAITING
        self.pbar.setMaximum(0)
        self.pbar.show()

        n_files = len(serialized_dict['open_files'])
        excetpion_tracker: Dict[uuid.UUID, Tuple[str, str]] = {}

        self.pbar.setMaximum(n_files)

        try:
            current_uuid = uuid.UUID(serialized_dict['current_uuid'])
        except Exception:
            current_uuid = None

        current_spec_list_item: Optional[QtWidgets.QListWidgetItem] = None
        for j, file_info in enumerate(serialized_dict['open_files']):
            self.pbar.setValue(j + 1)
            self.statusbar.showMessage(
                self.qapp.tr("Loading project...")
            )
            self.qapp.processEvents()

            item_uuid = uuid.UUID(file_info['uuid'])
            item_row = int(file_info['index'])

            try:
                sp: Spectrum1D = loaders.read(file_info['path'])
            except Exception as exc:
                excetpion_tracker[item_uuid] = (file_info['path'], str(exc))
                continue

            new_item = QtWidgets.QListWidgetItem(file_info['text'])
            new_item.setCheckState(QtCore.Qt.CheckState(file_info['checked']))
            new_item.setToolTip(file_info['path'])
            new_item.setData(QtCore.Qt.ItemDataRole.UserRole, item_uuid)

            if item_uuid == current_uuid:
                current_spec_list_item = new_item

            self.main_wnd.spec_list_widget.addItem(new_item)
            self.open_spectra[item_uuid] = sp
            self.open_spectra_files[item_uuid] = file_info['path']
            self.open_spectra_items[item_uuid] = new_item
            self.main_wnd.spec_list_widget.insertItem(item_row, new_item)
            self.qapp.processEvents()

        for h_uuid, obj_info in serialized_dict['objects_properties'].items():

            obj_uuid = uuid.UUID(h_uuid)

            lines_list = []
            for line_info in obj_info['lines']['list']:
                lines_list.append({
                    'row': int(line_info['row']),
                    'data': float(line_info['data']),
                    'text': str(line_info['text']),
                    'checked': QtCore.Qt.CheckState(line_info['checked'])
                })

            z_list = []
            for z_info in obj_info['lines']['redshifts']:
                z_list.append({
                    'row': int(z_info['row']),
                    'text': str(z_info['text']),
                    'data': z_info['data']
                })

            try:
                redshift = obj_info['redshift']
            except KeyError:
                redshift = None

            try:
                quality_flag = obj_info['quality_flag']
            except KeyError:
                quality_flag = 0

            self._update_spec_item_qf(obj_uuid, quality_flag)
            self.object_state_dict[obj_uuid] = {
                'redshift': redshift,
                'quality_flag': quality_flag,
                'lines': {
                    'list': lines_list,
                    'redshifts': z_list,
                }
            }

        if current_spec_list_item is not None:
            self.main_wnd.spec_list_widget.setCurrentItem(
                current_spec_list_item
            )

        self.pbar.hide()
        self._unlock()
        self.global_state = GlobalState.READY

        self.redrawCurrentSpec()

        self.statusbar.showMessage(self.qapp.tr("Project loaded"))

        if excetpion_tracker:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                self.qapp.tr(
                    "An error has occurred while loading the project: one or "
                    "more files could not be loaded."
                )
            )
            self.msgBox.setDetailedText(
                "\n\n".join(
                    [
                        f"uuid: {exc_uuid}\n"
                        f"file: {exc_data[0]}\n"
                        f"reason: {exc_data[1]}"
                        for exc_uuid, exc_data in excetpion_tracker.items()
                    ]
                )
            )
            self.msgBox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            self.msgBox.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()

    def redrawCurrentSpec(self, *args, **kwargs) -> None:
        """
        Redraw the charts.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        None

        """
        self.currentSpecItemChanged(
            self.main_wnd.spec_list_widget.currentItem()
        )

    def requestCancelCurrentOperation(self) -> None:
        """
        Set the global state of the program to REUQUEST_CANCEL.

        Returns
        -------
        None

        """
        self.global_state = GlobalState.REUQUEST_CANCEL

    def saveProject(self, file_name: str) -> None:
        """
        Save the project to a file

        Parameters
        ----------
        file_name : str
            The destination file path.

        Returns
        -------
        None

        """
        # Store any pending information for the current object
        self._backup_current_object_state()

        # Serialize program state for json dumping
        serialized_open_file_list = []
        for k in range(self.main_wnd.spec_list_widget.count()):
            item = self.main_wnd.spec_list_widget.item(k)
            item_uuid: uuid.UUID = item.data(
                QtCore.Qt.ItemDataRole.UserRole
            )

            file_info = {
                'index': k,
                'uuid': item_uuid.hex,
                'text': item.text(),
                'path': self.open_spectra_files[item_uuid],
                'checked': int(item.checkState().value)
            }

            serialized_open_file_list.append(file_info)

        # Serialize objects info for json dumping
        serialized_object_info_dict: Dict[str, Any] = {}
        for obj_uuid, obj_info in self.object_state_dict.items():
            serialized_lines_list: List[Dict[str, Any]] = []
            serialized_z_list: List[Dict[str, Any]] = []

            for line_info in obj_info['lines']['list']:
                serialized_lines_list.append({
                    'row': int(line_info['row']),
                    'data': float(line_info['data']),
                    'text': str(line_info['text']),
                    'checked': int(line_info['checked'].value)
                })

            for z_info in obj_info['lines']['redshifts']:
                serialized_z_list.append({
                    'row': int(z_info['row']),
                    'text': str(z_info['text']),
                    'data': z_info['data']
                })

            serialized_info_dict: Dict[str, Dict] = {
                'redshift': obj_info['redshift'],
                'quality_flag': obj_info['quality_flag'],
                'lines': {
                    'list': serialized_lines_list,
                    'redshifts': serialized_z_list,
                }
            }

            serialized_object_info_dict[obj_uuid.hex] = serialized_info_dict

        if self.current_uuid is None:
            serialized_current_uuid = None
        else:
            serialized_current_uuid = self.current_uuid.hex

        project_dict = {
            'current_uuid': serialized_current_uuid,
            'open_files': serialized_open_file_list,
            'objects_properties': serialized_object_info_dict
        }

        with open(file_name, 'w') as f:
            json.dump(project_dict, f, indent=2)

    def setCurrentObjectRedshift(self, redshift: Optional[float]) -> None:
        if redshift is None:
            redshift = 0

        self.flux_chart_view.setRedshift(redshift=redshift)

        item_col_0: QtWidgets.QTableWidgetItem
        item_col_1: QtWidgets.QTableWidgetItem
        item_col_2: QtWidgets.QTableWidgetItem
        for j in range(self.main_wnd.lines_table_widget.rowCount()):
            item_col_0 = self.main_wnd.lines_table_widget.item(j, 0)
            item_col_1 = self.main_wnd.lines_table_widget.item(j, 1)
            item_col_2 = self.main_wnd.lines_table_widget.item(j, 2)

            line_lam: float = item_col_0.data(QtCore.Qt.ItemDataRole.UserRole)
            rest_lam = line_lam / (1 + redshift)
            item_col_1.setText(f"{rest_lam:.2f} A")

            best_matches = [
                x[1]
                for x in lines.get_lines(wrange=[rest_lam - 5, rest_lam + 5])
            ]

            item_col_2.setText('; '.join(best_matches))

    def setCurrentObjectRedshiftFromLines(self, row: int) -> None:
        if row < 0:
            return

        self.main_wnd.z_dspinbox.setValue(
            self.main_wnd.lines_match_list_widget.item(row).data(
                QtCore.Qt.ItemDataRole.UserRole
            )[0]
        )

    def setCurrentObjectRedshiftFromSingleLine(self, row: int) -> None:
        if row < 0:
            return

        lines_tbl_row = self.main_wnd.lines_table_widget.currentRow()
        if  lines_tbl_row < 0:
            return

        w_item = self.main_wnd.lines_table_widget.item(lines_tbl_row, 0)
        obs_lam = w_item.data(QtCore.Qt.ItemDataRole.UserRole)

        rest_lam = self.main_wnd.single_line_combo_box.itemData(
            row, QtCore.Qt.ItemDataRole.UserRole
        )

        redshift = (obs_lam/rest_lam) - 1

        self.main_wnd.z_dspinbox.setValue(redshift)

    def setShowLinesType(self, type_index: int) -> None:
        if type_index == 0:
            self.flux_chart_view.setLinesType('')
        elif type_index == 1:
            self.flux_chart_view.setLinesType('E')
        elif type_index == 2:
            self.flux_chart_view.setLinesType('A')

    def setSmoothingFactor(self, smoothing_value: float) -> None:
        self.redrawCurrentSpec()

    def toggleSimilarSpecItems(
        self,
        item: Optional[QtWidgets.QListWidgetItem]
    ) -> None:
        if item is None:
            return

        ref_uuid = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if ref_uuid in self.object_state_dict:
            ref_qf = self.object_state_dict[ref_uuid]['quality_flag']
        else:
            ref_qf = 0

        if item.checkState() == QtCore.Qt.CheckState.Checked:
            ref_check_state = QtCore.Qt.CheckState.Unchecked
        else:
            ref_check_state = QtCore.Qt.CheckState.Checked

        for row in range(self.main_wnd.spec_list_widget.count()):
            other_item = self.main_wnd.spec_list_widget.item(row)
            other_uuid = other_item.data(QtCore.Qt.ItemDataRole.UserRole)
            if other_uuid in self.object_state_dict:
                other_qf = self.object_state_dict[other_uuid]['quality_flag']
            else:
                other_qf = 0

            if other_qf == ref_qf:
                other_item.setCheckState(ref_check_state)

    def toggleSmothing(self, show_smoothing: int) -> None:
        self.redrawCurrentSpec()

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
