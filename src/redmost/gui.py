#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: Maurizio D'Addona
"""
from __future__ import annotations

import os
import pickle
import sys
import json
import uuid
import webbrowser
from enum import Enum
from typing import TYPE_CHECKING
from typing import Optional, Union, Tuple, List, Dict, Callable, Any, cast

import numpy as np
from astropy.nddata import VarianceUncertainty  # type: ignore
from astropy.nddata import StdDevUncertainty  # type: ignore
from astropy.nddata import InverseVariance  # type: ignore
from astropy import units  # type: ignore
from astropy.table import Table  # type: ignore

from specutils import Spectrum1D  # type: ignore

import redmost
from redmost import loaders
from redmost import utils
from redmost import lines
from redmost import backends

from redmost.qt_compat import qt_api, get_qapp

if TYPE_CHECKING:
    # NOTE: These imports will be actually executed only by the type-checking
    #       programs, like mypy.
    from redmost.qt_compat import QtCore, QtGui, QtWidgets, QtCharts
    from redmost.qt_compat import Signal


class SpectrumQChartView(qt_api.QtCharts.QChartView):
    """Subclass of qt_api.QtCharts.QChartView with advanced features."""
    onMouseMoveSeries = qt_api.Signal(object)
    onMousePressSeries = qt_api.Signal(object)
    onMouseReleaseSeries = qt_api.Signal(object)
    onMouseDoubleClickSeries = qt_api.Signal(object)
    onMouseWheelEvent = qt_api.Signal(object)

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

        self._last_mouse_pos: QtCore.QPointF | None = None
        self._static_lines_list: List[Tuple[float, str, str]] = []
        self._lines_buffer_list: List[Tuple[float, str, str]] = []
        self._mouse_lambda: Union[None, QtCore.QPointF] = None
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

        self.horizontal_lock: bool = False
        self.master: bool = False
        self.show_lines: bool = False
        self.sync_height: bool = False
        self.sync_width: bool = False
        self.redshift: float = 0
        self.vertical_lock: bool = False

        self.setDragMode(qt_api.QtCharts.QChartView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

        self.setViewportUpdateMode(
            qt_api.QtWidgets.
            QGraphicsView.
            ViewportUpdateMode.
            FullViewportUpdate
        )

        self.updateLines()

        # Pass events also to siblings
        self._sibling_locked: bool = False
        self.siblings: List[SpectrumQChartView] = []

        self.setSizePolicy(
            qt_api.QtWidgets.QSizePolicy(
                qt_api.QtWidgets.QSizePolicy.Policy.Expanding,
                qt_api.QtWidgets.QSizePolicy.Policy.Expanding
            )
        )
        self.chart().setSizePolicy(
            qt_api.QtWidgets.QSizePolicy(
                qt_api.QtWidgets.QSizePolicy.Policy.Expanding,
                qt_api.QtWidgets.QSizePolicy.Policy.Expanding
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

        :param sibling: A new sibling SpectrumQChartView
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

        if len(self.chart().series()) == 0:
            return

        painter.save()
        plot_area: QtCore.QRectF = self.chart().plotArea()
        pen: QtGui.QPen = qt_api.QtGui.QPen()
        pen_color: QtGui.QColor

        if self._mouse_lambda is not None:
            pen_color = qt_api.QtGui.QColor("#aa333333")
            pen.setColor(pen_color)
            pen.setWidthF(1.0)
            pen.setDashPattern((6, 6, 6, 6))
            painter.setPen(pen)

            mouse_line_x = self.chart().mapToPosition(
                qt_api.QtCore.QPointF(self._mouse_lambda, 0.0)
            ).x()

            if (
                    (mouse_line_x >= plot_area.left()) and
                    (mouse_line_x <= plot_area.right())
            ):
                p_line_top = qt_api.QtCore.QPointF(
                    mouse_line_x, plot_area.top()
                )
                p_line_bottom = qt_api.QtCore.QPointF(
                    mouse_line_x, plot_area.bottom()
                )
                painter.drawLine(p_line_bottom, p_line_top)

        if not self.show_lines:
            painter.restore()
            return

        for line_lambda, line_name, line_type in self._lines_buffer_list:
            pen_color = qt_api.QtGui.QColor(self.line_colors["unknown"])

            if ('AE' in line_type) or ('EA' in line_type):
                pen_color = qt_api.QtGui.QColor(self.line_colors["both"])
            elif 'A' in line_type:
                pen_color = qt_api.QtGui.QColor(self.line_colors["absorption"])
            elif 'E' in line_type:
                pen_color = qt_api.QtGui.QColor(self.line_colors["emission"])

            pen_color.setAlphaF(self.line_plot_options["alpha"])
            pen.setColor(pen_color)
            pen.setWidthF(self.line_plot_options["width"])
            pen.setDashPattern(self.line_plot_options["dash_pattern"])
            painter.setPen(pen)

            line_x = self.chart().mapToPosition(
                qt_api.QtCore.QPointF(line_lambda, 0)
            ).x()

            if (line_x >= plot_area.left()) and (line_x <= plot_area.right()):
                p_line_top = qt_api.QtCore.QPointF(
                    line_x, plot_area.top()
                )
                p_line_bottom = qt_api.QtCore.QPointF(
                    line_x, plot_area.bottom()
                )
                p_text_top = qt_api.QtCore.QPointF(
                    line_x, plot_area.top() - 5
                )
                painter.drawLine(p_line_bottom, p_line_top)
                painter.drawText(p_text_top, line_name)

        painter.restore()

    def hideLines(self) -> None:
        self.setLinesVisible(False)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse double click events.

        :param event: The input event.
        """
        if self._sibling_locked:
            return

        self._sibling_locked = True
        for sibling in self.siblings:
            pass
        self._sibling_locked = False

        super().mouseDoubleClickEvent(event)
        self.onMouseDoubleClickSeries.emit((self.toSeriesPos(event), event))

    def mouseMoveEvent(
        self,
        event: QtGui.QMouseEvent,
        *args, **kwargs
    ) -> None:
        """
        Handle mouse move events.

        :param event: The input event.
        """
        if self._sibling_locked:
            return

        try:
            mouse_pos = event.position()
        except AttributeError:
            mouse_pos = event.pos()

        if event.buttons() & qt_api.QtCore.Qt.MouseButton.MiddleButton:
            if self._last_mouse_pos is None:
                return

            delta = mouse_pos - self._last_mouse_pos
            self._last_mouse_pos = mouse_pos

            self.scroll(-delta.x(), delta.y())

            event.accept()

        data_pos = self.toSeriesPos(event)

        self._mouse_lambda = data_pos.x()

        self._sibling_locked = True
        for sibling in self.siblings:
            sibling._mouse_lambda = self._mouse_lambda
            sibling.chart().update()
            sibling.update()
        self._sibling_locked = False

        self.chart().update()
        self.update()
        super().mouseMoveEvent(event)

        self.onMouseMoveSeries.emit((data_pos, event))

    def mousePressEvent(
        self,
        event: QtGui.QMouseEvent,
        *args, **kwargs
    ) -> None:
        """
        Handle mouse button press events.

        :param event: The input event.
        """
        if self._sibling_locked:
            return

        if event.button() == qt_api.QtCore.Qt.MouseButton.MiddleButton:
            get_qapp().setOverrideCursor(
                qt_api.QtCore.Qt.CursorShape.SizeAllCursor
            )

            try:
                self._last_mouse_pos = event.position()
            except AttributeError:
                self._last_mouse_pos = event.pos()

            event.accept()

        super().mousePressEvent(event)
        self.onMousePressSeries.emit((self.toSeriesPos(event), event))

    def mouseReleaseEvent(
        self,
        event: QtGui.QMouseEvent,
        *args, **kwargs
    ) -> None:
        """
        Handle mouse button release events.

        :param event: The input event.
        """
        if self._sibling_locked:
            return

        self._sibling_locked = True
        for sibling in self.siblings:
            sibling.mouseReleaseEvent(event)
        self._sibling_locked = False

        if event.button() == qt_api.QtCore.Qt.MouseButton.MiddleButton:
            get_qapp().restoreOverrideCursor()
            event.accept()
        elif event.button() == qt_api.QtCore.Qt.MouseButton.RightButton:
            qt_api.QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
        else:
            # Passtrough for rubber band handling
            super().mouseReleaseEvent(event)

        self.onMouseReleaseSeries.emit((self.toSeriesPos(event), event))
        band = self.rubberBand()
        if band != qt_api.QtCharts.QChartView.RubberBand.NoRubberBand:
            self.syncSiblingAxes()

    def removeSibling(self, sibling: SpectrumQChartView) -> None:
        """
        Remove a sibling to this widget.

        :param sibling: The sibling to remove.
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

        if self.vertical_lock and self.horizontal_lock:
            return
        elif not (self.vertical_lock or self.horizontal_lock):
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

        if not self.horizontal_lock:
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

        sibling_chart: QtCharts.QChart
        sibling_plot_area: QtCore.QRectF
        for sibling in self.siblings:

            sibling.setAxesRange(x_min, y_min, x_max, y_max)

            if sibling.sync_width or sibling.sync_height:
                sibling_chart = sibling.chart()
                sibling_chart.setPlotArea(qt_api.QtCore.QRectF())
                sibling_plot_area = sibling_chart.plotArea()
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

        :param event: The input event.
        """
        try:
            widget_pos = event.position()
        except AttributeError:
            widget_pos = event.pos()

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

        :param event: The input event.
        """

        super().wheelEvent(event)
        if not event.angleDelta().isNull():
            delta_y = event.angleDelta().y() / 5
            delta_x = event.angleDelta().x() / 5
        elif not event.pixelDelta().isNull():
            delta_y = event.pixelDelta().y()
            delta_x = event.pixelDelta().x()
        else:
            delta_y = 0
            delta_x = 0

        modifiers = qt_api.QtWidgets.QApplication.keyboardModifiers()

        if modifiers == qt_api.QtCore.Qt.KeyboardModifier.ShiftModifier:
            # Vertical scroll
            self.scroll(delta_y, delta_x)
        elif modifiers == qt_api.QtCore.Qt.KeyboardModifier.ControlModifier:
            # Horizontal scroll
            if delta_y > 0:
                self.zoomIn()
            else:
                self.zoomOut()
        else:
            self.scroll(delta_x, delta_y)

        self.onMouseWheelEvent.emit(event)

    def zoom(
        self,
        value: float,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None
    ) -> None:
        """
        Zoom.

        :param value: Zoom value.
        :param x_center: x of the zoom center
        :param y_center: y of the zoom center
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

        if not self.horizontal_lock:
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

        :param value: Zoom value. The default value is 2.0.
        :param x_center: x of the zoom center
        :param y_center: y of the zoom center
        """
        self.zoom(value, x_center, y_center)

    def zoomOut(
        self,
        value: float = 0.5,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None
    ) -> None:
        """
        Zoom in.

        :param value: Zoom value. The default value is 0.5.
        :param x_center: x of the zoom center
        :param y_center: y of the zoom center
        """
        self.zoom(value, x_center, y_center)

    def zoomReset(self) -> None:
        """Reset the zoom to fit the chart content."""
        if self._sibling_locked:
            return

        self._sibling_locked = True
        for sibling in self.siblings:
            sibling.zoomReset()
        self._sibling_locked = False

        self.setAxesRange(*self._getDataBounds())


class QRedrockHandler(qt_api.QtCore.QObject):
    """Class to handle redrock subprocess."""

    progress: Signal = qt_api.Signal(int)
    maximum: Signal = qt_api.Signal(int)
    reset: Signal = qt_api.Signal()
    message: Signal = qt_api.Signal(str)
    finished: Signal = qt_api.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.qapp = get_qapp()
        self.targets: List[Any] = []
        self.results = None
        self._t_dict: Union[None, Dict[str, uuid.UUID]] = None

        self._target_dump_file = 'rr.targets'
        self._result_dump_file = 'rr.result'
        self.process: QtCore.QProcess = qt_api.QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(  # type: ignore
            self._process_output
        )
        self.process.finished.connect(  # type: ignore
            self.retrieve_results
        )

        self._total_progress = 0
        self._total_maximum = 0

    def _process_output(self) -> None:
        data_bytes: QtCore.QByteArray
        data_bytes = self.process.readAllStandardOutput()
        text = bytes(data_bytes).decode()  # type: ignore
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


class AboutDialog(qt_api.QtWidgets.QDialog):

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        flags: QtCore.Qt.WindowType = qt_api.QtCore.Qt.WindowType.Dialog
    ):
        super().__init__(parent, flags)

        qapp = get_qapp()

        properties: Dict[str, str] = {
            qapp.tr("PROGRAM VERSION"): redmost.__version__,
            qapp.tr("QT BACKEND"): qt_api.qt_api_name,
            qapp.tr("HAS REDROCK"): (
                qapp.tr("YES") if backends.HAS_REDROCK else qapp.tr("NO")
            )
        }

        button_box = qt_api.QtWidgets.QDialogButtonBox(self)
        button_box.setStandardButtons(
            qt_api.QtWidgets.QDialogButtonBox.StandardButton.Ok
        )

        html_info_text = ""
        for line in redmost.ABOUT_TEXT.split('\n\n'):
            line = line.replace('\n', '<br>')
            html_info_text += f"<p>{line}</p>"

        text_edit: QtWidgets.QTextBrowser = qt_api.QtWidgets.QTextBrowser()
        text_edit.setReadOnly(True)
        text_edit.setOpenExternalLinks(True)
        text_edit.setHtml(html_info_text)
        text_edit.setWordWrapMode(qt_api.QtGui.QTextOption.WrapMode.NoWrap)

        status_tbl: QtWidgets.QTableWidget = qt_api.QtWidgets.QTableWidget()
        status_tbl.setColumnCount(2)
        status_tbl.setRowCount(len(properties))
        status_tbl.setSizePolicy(
            qt_api.QtWidgets.QSizePolicy(
                qt_api.QtWidgets.QSizePolicy.Policy.Expanding,
                qt_api.QtWidgets.QSizePolicy.Policy.Expanding
            )
        )
        status_tbl.setHorizontalHeaderLabels(
            [qapp.tr("PROPERTY"), qapp.tr("VALUE")]
        )

        item_key: QtWidgets.QTableWidgetItem
        item_val: QtWidgets.QTableWidgetItem
        for j, (key, val) in enumerate(properties.items()):
            item_key = qt_api.QtWidgets.QTableWidgetItem(key)
            item_key.setFlags(qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled)

            item_val = qt_api.QtWidgets.QTableWidgetItem(val)
            item_val.setFlags(qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled)

            status_tbl.setItem(j, 0, item_key)
            status_tbl.setItem(j, 1, item_val)

        header: Union[QtWidgets.QHeaderView, None]
        header = status_tbl.horizontalHeader()
        if header is None:
            header = qt_api.QtWidgets.QHeaderView(
                qt_api.QtCore.Qt.Orientation.Horizontal
            )
            status_tbl.setHorizontalHeader(header)

        header.setSectionResizeMode(
            qt_api.QtWidgets.QHeaderView.ResizeMode.Stretch
        )

        main_layout: QtWidgets.QVBoxLayout = qt_api.QtWidgets.QVBoxLayout()
        main_layout.addWidget(text_edit)
        main_layout.addWidget(status_tbl)
        main_layout.addStretch()
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.setWindowTitle("About redmost")

        button_box.accepted.connect(
            self.close
        )

        self.setMinimumWidth(640)


class ImportZcatDialog(qt_api.QtWidgets.QDialog):

    def __init__(self) -> None:
        super().__init__()
        qapp: QtWidgets.QApplication = get_qapp()

        self.zcat_tbl: Union[Table, None] = None

        button_box: QtWidgets.QDialogButtonBox
        button_box = qt_api.QtWidgets.QDialogButtonBox(self)
        button_box.setStandardButtons(
            qt_api.QtWidgets.QDialogButtonBox.StandardButton.Ok |
            qt_api.QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        main_layout: QtWidgets.QVBoxLayout
        main_layout = qt_api.QtWidgets.QVBoxLayout()

        mapping_widgets_layout = qt_api.QtWidgets.QGridLayout()

        id_select_label: QtWidgets.QLabel
        id_select_label = qt_api.QtWidgets.QLabel(
            qapp.tr("Match with catalogue:")
        )

        self.id_combo_box: QtWidgets.QComboBox
        self.id_combo_box = qt_api.QtWidgets.QComboBox()

        z_select_label: QtWidgets.QLabel
        z_select_label = qt_api.QtWidgets.QLabel(
            qapp.tr("Get redshift from column:")
        )

        self.z_combo_box: QtWidgets.QComboBox
        self.z_combo_box = qt_api.QtWidgets.QComboBox()

        qf_select_label: QtWidgets.QLabel
        qf_select_label = qt_api.QtWidgets.QLabel(
            qapp.tr("Quality flag (optional):")
        )

        self.qf_combo_box: QtWidgets.QComboBox
        self.qf_combo_box = qt_api.QtWidgets.QComboBox()

        mapping_widgets_layout.addWidget(id_select_label)
        mapping_widgets_layout.addWidget(self.id_combo_box)
        mapping_widgets_layout.addWidget(z_select_label)
        mapping_widgets_layout.addWidget(self.z_combo_box)
        mapping_widgets_layout.addWidget(qf_select_label)
        mapping_widgets_layout.addWidget(self.qf_combo_box)

        main_layout.addLayout(mapping_widgets_layout)
        main_layout.addStretch()
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)
        self.setWindowTitle(qapp.tr("Import Catalogue"))

        button_box.accepted.connect(
            self.accept
        )

        button_box.rejected.connect(
            self.reject
        )

    def setup(self, zcat_tbl: Table) -> None:

        self.zcat_tbl = zcat_tbl

        self.id_combo_box.clear()
        self.z_combo_box.clear()
        self.qf_combo_box.clear()

        self.qf_combo_box.addItem("None", None)
        for col_name in zcat_tbl.colnames:
            self.id_combo_box.addItem(col_name, col_name)
            self.z_combo_box.addItem(col_name, col_name)
            self.qf_combo_box.addItem(col_name, col_name)

    def get_mapping(self) -> Tuple[str, str, Union[str, None]]:
        id_col = str(
            self.id_combo_box.currentData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole
            )
        )
        z_col = str(
            self.z_combo_box.currentData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole
            )
        )

        qf_col = self.qf_combo_box.currentData(
            qt_api.QtCore.Qt.ItemDataRole.UserRole
        )
        if qf_col is not None:
            qf_col = str(qf_col)

        return id_col, z_col, qf_col


class SettingsDialog(qt_api.QtWidgets.QDialog):
    """Class definition for the QDialog created with the designer."""

    settings_tab_widget: QtWidgets.QTabWidget
    update_check_box: QtWidgets.QCheckBox
    qt_backend_check_box: QtWidgets.QCheckBox
    qt_backend_combo_box: QtWidgets.QComboBox
    style_combo_box: QtWidgets.QComboBox
    icon_theme_combo_box: QtWidgets.QComboBox


class MainWindow(qt_api.QtWidgets.QMainWindow):
    """Class definition for the QMainWindow created with the designer."""

    # actions
    action_about: QtGui.QAction
    action_controls_layout: QtGui.QAction
    action_exit: QtGui.QAction
    action_import_spectra: QtGui.QAction
    action_import_zcat: QtGui.QAction
    action_export_zcat: QtGui.QAction
    action_export_as_a_picture: QtGui.QAction
    action_settings: QtGui.QAction
    action_zoom_in: QtGui.QAction
    action_zoom_out: QtGui.QAction
    action_zoom_fit: QtGui.QAction
    action_save_project_as: QtGui.QAction
    action_save_project: QtGui.QAction
    action_online_user_manual: QtGui.QAction
    action_open_project: QtGui.QAction
    action_new_project: QtGui.QAction

    # Widgets
    add_line_button: QtWidgets.QPushButton
    container_tab_var: QtWidgets.QWidget
    container_tab_wd: QtWidgets.QWidget
    container_tab_sky: QtWidgets.QWidget
    delete_line_button: QtWidgets.QPushButton
    delete_lines_button: QtWidgets.QPushButton
    export_zcat_button: QtWidgets.QPushButton
    flux_group_box: QtWidgets.QGroupBox
    flux_widget_layout: QtWidgets.QHBoxLayout
    import_zcat_button: QtWidgets.QPushButton
    info_group_box: QtWidgets.QGroupBox
    lines_auto_button: QtWidgets.QPushButton
    lines_tol_dspinbox: QtWidgets.QDoubleSpinBox
    lines_match_list_widget: QtWidgets.QListWidget
    lines_table_widget: QtWidgets.QTableWidget
    log_y_check_box: QtWidgets.QCheckBox
    match_lines_button: QtWidgets.QPushButton
    next_spec_button: QtWidgets.QPushButton
    obj_prop_table_widget: QtWidgets.QTableWidget
    other_charts_tab_widget: QtWidgets.QTabWidget
    plot_group_box: QtWidgets.QGroupBox
    previous_spec_button: QtWidgets.QPushButton
    qflag_combo_box: QtWidgets.QComboBox
    red_group_box: QtWidgets.QGroupBox
    redrock_all_radio: QtWidgets.QRadioButton
    redrock_selected_radio: QtWidgets.QRadioButton
    redrock_run_button: QtWidgets.QPushButton
    redrock_text_edit: QtWidgets.QTextEdit
    redrock_progress_bar: QtWidgets.QProgressBar
    redrock_current_radio: QtWidgets.QRadioButton
    remove_selected_button: QtWidgets.QPushButton
    remove_spec_button: QtWidgets.QPushButton
    show_lines_check_box: QtWidgets.QCheckBox
    show_lines_combo_box: QtWidgets.QComboBox
    single_line_combo_box: QtWidgets.QComboBox
    sky_widget_layout: QtWidgets.QHBoxLayout
    smoothing_dspinbox: QtWidgets.QDoubleSpinBox
    smoothing_check_box: QtWidgets.QCheckBox
    spec_group_box: QtWidgets.QGroupBox
    spec_list_widget: QtWidgets.QListWidget
    splitter_main: QtWidgets.QSplitter
    splitter_plots: QtWidgets.QSplitter
    toggle_all_button: QtWidgets.QPushButton
    toggle_done_button: QtWidgets.QPushButton
    toggle_similar_button: QtWidgets.QPushButton
    var_widget_layout: QtWidgets.QHBoxLayout
    wdisp_widget_layout: QtWidgets.QHBoxLayout
    z_dspinbox: QtWidgets.QDoubleSpinBox
    z_min_dspinbox: QtWidgets.QDoubleSpinBox
    z_max_dspinbox: QtWidgets.QDoubleSpinBox


class GuiApp:
    """General class for the main GUI."""

    def __init__(self) -> None:
        self.qapp: QtWidgets.QApplication = get_qapp()
        self.qapp.setApplicationName("Redmost")
        self.qapp.setOrganizationName("DAddona")
        self.qapp.setOrganizationDomain(
            "github.com/mauritiusdadd/redmost"
        )

        self.settings: QtCore.QSettings = qt_api.QtCore.QSettings()

        self.open_spectra: Dict[uuid.UUID, Spectrum1D] = {}
        self.open_spectra_files: Dict[uuid.UUID, str] = {}
        self.open_spectra_items: Dict[
            uuid.UUID, QtWidgets.QListWidgetItem
        ] = {}
        self.current_uuid: Optional[uuid.UUID] = None
        self.object_state_dict: Dict[uuid.UUID, Any] = {}
        self.current_project_file_path: Optional[str] = None
        self.current_open_dir: Optional[str] = None

        self.global_state: Enum = GlobalState.READY

        self.qf_color: Dict[int, str] = {
            0: "#FFFFFF",
            1: "#55940a00",
            2: "#55945400",
            3: "#55176601",
            4: "#55012d66"
        }

        self.main_wnd: MainWindow = cast(
            MainWindow,
            qt_api.loadUiWidget("main_window.ui")
        )
        setattr(self.main_wnd, "closeEvent", self.closeEvent)

        self.settings_wnd = cast(
            SettingsDialog,
            qt_api.loadUiWidget("settings_dialog.ui")
        )
        self.settings_wnd.accepted.connect(
            self.acceptSettings
        )
        self.settings_wnd.rejected.connect(
            self.restoreSettings
        )

        self.main_wnd.setWindowTitle("redmost")
        self.main_wnd.setWindowIcon(
            qt_api.QtGui.QIcon(utils.get_data_file("redmost.png"))
        )
        self.qapp.setWindowIcon(
            qt_api.QtGui.QIcon(utils.get_data_file("redmost.png"))
        )

        self.msgBox: QtWidgets.QMessageBox
        self.msgBox = qt_api.QtWidgets.QMessageBox(
            parent=self.main_wnd
        )

        # About dialog

        self.about_wnd: AboutDialog = AboutDialog()

        # Status Bar
        self.mousePosLabel: QtWidgets.QLabel
        self.mousePosLabel = qt_api.QtWidgets.QLabel("")

        # Global cancel button
        self.cancel_button: QtWidgets.QPushButton
        self.cancel_button = qt_api.QtWidgets.QPushButton()
        self.cancel_button.setText("Cancel")
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(
            self.requestCancelCurrentOperation
        )

        # Global progress bar
        self.pbar: QtWidgets.QProgressBar
        self.pbar = qt_api.QtWidgets.QProgressBar()
        self.pbar.hide()

        self.statusbar: QtWidgets.QStatusBar
        self.statusbar = qt_api.QtWidgets.QStatusBar()
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

        # Dialog for importing zcat

        self.zcat_mapping_dialog = ImportZcatDialog()
        self.zcat_mapping_dialog.finished.connect(self.getZcatColumnMapping)

        # QChartView widget for flux

        self.flux_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.flux_group_box
        )
        self.flux_chart_view.master = True
        self.flux_chart_view.setObjectName("flux_chart_view")
        self.flux_chart_view.setRenderHint(
            qt_api.QtGui.QPainter.RenderHint.Antialiasing
        )
        self.flux_chart_view.setHorizontalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.flux_chart_view.setVerticalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.flux_chart_view.setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

        self.main_wnd.flux_widget_layout.addWidget(self.flux_chart_view)

        # QChartView widget for variance

        self.var_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.other_charts_tab_widget
        )
        self.var_chart_view.vertical_lock = True
        self.var_chart_view.setObjectName("var_chart_view")
        self.var_chart_view.setRenderHint(
            qt_api.QtGui.QPainter.RenderHint.Antialiasing
        )
        self.var_chart_view.setHorizontalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.var_chart_view.setVerticalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.var_chart_view.setContentsMargins(0, 0, 0, 0)
        self.var_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.var_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.var_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.main_wnd.var_widget_layout.addWidget(self.var_chart_view)

        # QChartView widget for resolution curve

        self.wdisp_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.other_charts_tab_widget
        )
        self.wdisp_chart_view.vertical_lock = True
        self.wdisp_chart_view.setObjectName("var_chart_view")
        self.wdisp_chart_view.setRenderHint(
            qt_api.QtGui.QPainter.RenderHint.Antialiasing
        )
        self.wdisp_chart_view.setHorizontalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.wdisp_chart_view.setVerticalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.wdisp_chart_view.setContentsMargins(0, 0, 0, 0)
        self.wdisp_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.wdisp_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.wdisp_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.main_wnd.wdisp_widget_layout.addWidget(self.wdisp_chart_view)

        # QChartView widget for sky

        self.sky_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.other_charts_tab_widget
        )
        self.sky_chart_view.vertical_lock = True
        self.sky_chart_view.setObjectName("var_chart_view")
        self.sky_chart_view.setRenderHint(
            qt_api.QtGui.QPainter.RenderHint.Antialiasing
        )
        self.sky_chart_view.setHorizontalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.sky_chart_view.setVerticalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.sky_chart_view.setContentsMargins(0, 0, 0, 0)
        self.sky_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.sky_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.sky_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.main_wnd.sky_widget_layout.addWidget(self.sky_chart_view)

        # Add siblings to main flux QChartView

        self.flux_chart_view.addSibling(self.var_chart_view)
        self.flux_chart_view.addSibling(self.wdisp_chart_view)
        self.flux_chart_view.addSibling(self.sky_chart_view)

        # Set QSplitters initial sizes

        self.main_wnd.splitter_main.setSizes([200, 1000, 200])
        self.main_wnd.splitter_plots.setSizes([500, 300])

        # Connect signals

        self.main_wnd.log_y_check_box.toggled.connect(
            self.redrawCurrentSpec
        )

        self.main_wnd.next_spec_button.clicked.connect(
            self._next_spec
        )

        self.main_wnd.previous_spec_button.clicked.connect(
            self._previous_spec
        )

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
        self.main_wnd.action_new_project.triggered.connect(
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

        self.main_wnd.action_online_user_manual.triggered.connect(
            self._open_online_doc
        )

        self.main_wnd.action_about.triggered.connect(
            self.about_wnd.exec
        )

        self.main_wnd.action_settings.triggered.connect(
            self.openSettingsDialog
        )

        self.main_wnd.action_export_as_a_picture.triggered.connect(
            self.doExportAsPicture
        )

        # Final steps

        # This dictionary contains the global settings that should be saved
        # upon closing and restored when the application is started again.
        #
        # Keys are strings containing a unique name for the properties.
        # Values are tuples of three elements, and each element is a callable
        # object: the first one is used to retrieve the property value, the
        # second one is used to set the property value and the third one is
        # used to cast the valued read from the settings to the appropriate
        # type. This last callable is optional and can be replaced with a
        # None if no casting is required.
        self._global_setting: Dict[
            str,
            Tuple[Callable, Callable, Optional[Callable]]
        ] = {
            "geometry": (
                self.main_wnd.saveGeometry,
                self.main_wnd.restoreGeometry,
                None
            ),
            "smoothing_state": (
                self.main_wnd.smoothing_check_box.checkState,
                self.main_wnd.smoothing_check_box.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "smoothing_value": (
                self.main_wnd.smoothing_dspinbox.value,
                self.main_wnd.smoothing_dspinbox.setValue,
                float,
            ),
            "show_lines": (
                self.main_wnd.show_lines_check_box.checkState,
                self.main_wnd.show_lines_check_box.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "lines_type": (
                self.main_wnd.show_lines_combo_box.currentIndex,
                self.main_wnd.show_lines_combo_box.setCurrentIndex,
                int
            ),
            "log_y": (
                self.main_wnd.log_y_check_box.checkState,
                self.main_wnd.log_y_check_box.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "check_for_updates": (
                self.settings_wnd.update_check_box.checkState,
                self.settings_wnd.update_check_box.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "application_style": (
                self.settings_wnd.style_combo_box.currentIndex,
                self.settings_wnd.style_combo_box.setCurrentIndex,
                int
            ),
            "icon_theme": (
                self.settings_wnd.icon_theme_combo_box.currentIndex,
                self.settings_wnd.icon_theme_combo_box.setCurrentIndex,
                int
            )
        }

        self.loadSettings()
        self.newProject()

        self.main_wnd.show()

        if self.settings_wnd.update_check_box.isChecked():
            is_outdated, new_ver = utils.check_updates()
            if is_outdated:
                self._show_update_message(new_ver)

    def _backup_current_object_state(self) -> None:
        """Backup the program state for the current object."""
        if self.current_uuid is None:
            return

        self.global_state = GlobalState.SAVE_OBJECT_STATE

        lines_list = []
        for row_index in range(self.main_wnd.lines_table_widget.rowCount()):
            w_item: Union[None, QtWidgets.QTableWidgetItem]
            w_item = self.main_wnd.lines_table_widget.item(row_index, 0)
            if w_item is None:
                continue

            line_info = {
                'row': row_index,
                'data': float(
                    w_item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)
                ),
                'text': w_item.text(),
                'checked': w_item.checkState()
            }
            lines_list.append(line_info)

        redshifts_form_lines = []
        for row_index in range(self.main_wnd.lines_match_list_widget.count()):
            z_item: Union[None, QtWidgets.QListWidgetItem]
            z_item = self.main_wnd.lines_match_list_widget.item(row_index)
            if z_item is None:
                continue

            z_info = {
                'row': row_index,
                'text': z_item.text(),
                'data': z_item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)
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

    def _lock(self, *args, **kwargs) -> None:
        self.main_wnd.spec_group_box.setEnabled(False)
        self.main_wnd.red_group_box.setEnabled(False)
        self.main_wnd.info_group_box.setEnabled(False)
        self.flux_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.NoRubberBand
        )

    def _safe_set_spec_index(self, row: int) -> None:
        if (
            (row < 0) or
            (row >= self.main_wnd.spec_list_widget.count())
        ):
            return
        self.main_wnd.spec_list_widget.setCurrentRow(row)

    def _next_spec(self) -> None:
        self._safe_set_spec_index(
            self.main_wnd.spec_list_widget.currentRow() + 1
        )

    def _previous_spec(self) -> None:
        self._safe_set_spec_index(
            self.main_wnd.spec_list_widget.currentRow() - 1
        )

    def _load_icons(self, theme: Optional[str] = "feather") -> None:

        # Icons for actions

        self.main_wnd.action_about.setIcon(
            qt_api.get_qicon("info", theme)
        )
        self.main_wnd.action_controls_layout.setIcon(
            qt_api.get_qicon("sliders", theme)
        )
        self.main_wnd.action_exit.setIcon(
            qt_api.get_qicon("log-out", theme)
        )
        self.main_wnd.action_export_as_a_picture.setIcon(
            qt_api.get_qicon("camera", theme)
        )
        self.main_wnd.action_export_zcat.setIcon(
            qt_api.get_qicon("share", theme)
        )
        self.main_wnd.action_import_spectra.setIcon(
            qt_api.get_qicon("file-plus", theme)
        )
        self.main_wnd.action_import_zcat.setIcon(
            qt_api.get_qicon("table", theme)
        )
        self.main_wnd.action_new_project.setIcon(
            qt_api.get_qicon("file", theme)
        )
        self.main_wnd.action_online_user_manual.setIcon(
            qt_api.get_qicon("book", theme)
        )
        self.main_wnd.action_open_project.setIcon(
            qt_api.get_qicon("folder", theme)
        )
        self.main_wnd.action_save_project.setIcon(
            qt_api.get_qicon("save", theme)
        )
        self.main_wnd.action_save_project_as.setIcon(
            qt_api.get_qicon("save_as", theme)
        )
        self.main_wnd.action_settings.setIcon(
            qt_api.get_qicon("settings", theme)
        )
        self.main_wnd.action_zoom_fit.setIcon(
            qt_api.get_qicon("maximize", theme)
        )
        self.main_wnd.action_zoom_in.setIcon(
            qt_api.get_qicon("zoom-in", theme)
        )
        self.main_wnd.action_zoom_out.setIcon(
            qt_api.get_qicon("zoom-out", theme)
        )

        # icons for buttons

        self.cancel_button.setIcon(
            qt_api.get_qicon("x-octagon", theme)
        )
        self.main_wnd.add_line_button.setIcon(
            qt_api.get_qicon("plus-square", theme)
        )
        self.main_wnd.delete_line_button.setIcon(
            qt_api.get_qicon("minus-square", theme)
        )
        self.main_wnd.delete_lines_button.setIcon(
            qt_api.get_qicon("delete", theme)
        )
        self.main_wnd.export_zcat_button.setIcon(
            qt_api.get_qicon("share", theme)
        )
        self.main_wnd.import_zcat_button.setIcon(
            qt_api.get_qicon("table", theme)
        )
        self.main_wnd.lines_auto_button.setIcon(
            qt_api.get_qicon("", theme)
        )
        self.main_wnd.match_lines_button.setIcon(
            qt_api.get_qicon("", theme)
        )

        next_icon = qt_api.get_qicon("chevrons-right", theme)
        if next_icon.isNull():
            self.main_wnd.next_spec_button.setText("Next")
        else:
            self.main_wnd.next_spec_button.setText("")
            self.main_wnd.next_spec_button.setIcon(next_icon)

        previous_icon = qt_api.get_qicon("chevrons-left", theme)
        if next_icon.isNull():
            self.main_wnd.previous_spec_button.setText("Previous")
        else:
            self.main_wnd.previous_spec_button.setText("")
            self.main_wnd.previous_spec_button.setIcon(previous_icon)

        self.main_wnd.redrock_run_button.setIcon(
            qt_api.get_qicon("play", theme)
        )
        self.main_wnd.remove_selected_button.setIcon(
            qt_api.get_qicon("file-minus", theme)
        )
        self.main_wnd.remove_spec_button.setIcon(
            qt_api.get_qicon("file-minus", theme)
        )
        self.main_wnd.toggle_all_button.setIcon(
            qt_api.get_qicon("check-square", theme)
        )
        self.main_wnd.toggle_done_button.setIcon(
            qt_api.get_qicon("check-square", theme)
        )
        self.main_wnd.toggle_similar_button.setIcon(
            qt_api.get_qicon("check-square", theme)
        )

    def _open_online_doc(self, *args: Any, **kwargs: Any) -> None:
        webbrowser.open(redmost.ONLINE_DOC_URL)

    def _restore_object_state(self, obj_uuid: uuid.UUID) -> None:
        """
        Restore the programs state for a given object.

        :param obj_uuid: The UUID of the object
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
            w_item = qt_api.QtWidgets.QTableWidgetItem(line_info['text'])
            w_item.setData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole,
                line_info['data']
            )
            w_item.setCheckState(line_info['checked'])
            w_item.setFlags(
                qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
                qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled |
                qt_api.QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )

            r_item = qt_api.QtWidgets.QTableWidgetItem("")
            r_item.setFlags(
                qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
                qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            r_item.setData(qt_api.QtCore.Qt.ItemDataRole.UserRole, 0.0)

            m_item = qt_api.QtWidgets.QTableWidgetItem("")
            m_item.setFlags(
                qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
                qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled
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
            z_item = qt_api.QtWidgets.QListWidgetItem(z_info['text'])
            z_item.setData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole,
                z_info['data']
            )
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

    def _show_update_message(self, new_version: Optional[str]) -> None:
        self.msgBox.setText(
            self.qapp.tr(
                "A new version of Redmost is available"
            )
            + f": {new_version}." if new_version else "."
        )
        self.msgBox.setWindowTitle(self.qapp.tr("New version available"))
        self.msgBox.setInformativeText(
            self.qapp.tr("Please update the program using pip or visit") +
            f" <a href='{redmost.PYPI_REPO_PAGE}'>{redmost.PYPI_REPO_PAGE}</a>"
        )
        self.msgBox.setDetailedText("")
        self.msgBox.setTextFormat(qt_api.QtCore.Qt.TextFormat.RichText)
        self.msgBox.setIcon(qt_api.QtWidgets.QMessageBox.Icon.Information)
        self.msgBox.setStandardButtons(
            qt_api.QtWidgets.QMessageBox.StandardButton.Ok
        )
        self.msgBox.exec()

    def _unlock(self, *args, **kwargs) -> None:
        self.main_wnd.spec_group_box.setEnabled(True)
        self.main_wnd.red_group_box.setEnabled(True)
        self.main_wnd.info_group_box.setEnabled(True)
        self.main_wnd.plot_group_box.setEnabled(True)
        self.flux_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

    def _update_spec_item_qf(self, item_uuid: uuid.UUID, qf: int) -> None:
        item: QtWidgets.QListWidgetItem
        item = self.open_spectra_items[item_uuid]
        item.setBackground(qt_api.QtGui.QColor(self.qf_color[qf]))

    def _updateMouseLabelFromEvent(self, *args) -> None:
        self._updateMouseLabel(args[0][0])

    def _updateMouseLabel(self, mouse_pos: QtCore.QPointF) -> None:
        self.mousePosLabel.setText(f"\u03BB = {mouse_pos.x():.2f}")


    def acceptSettings(self, *args, **kwargs) -> None:
        self.saveSettings()
        self.loadSettings()

    def addLine(self, wavelength: float) -> None:
        """
        Add a new line by selecting its position with the mouse.

        :param wavelength: The wavelength of the line.
        """
        lam_item = qt_api.QtWidgets.QTableWidgetItem(f"{wavelength:.2f} A")
        lam_item.setFlags(
            qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
            qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled |
            qt_api.QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        lam_item.setCheckState(qt_api.QtCore.Qt.CheckState.Checked)
        lam_item.setData(qt_api.QtCore.Qt.ItemDataRole.UserRole, wavelength)

        rest_lam = wavelength / (1 + self.main_wnd.z_dspinbox.value())
        rest_item = qt_api.QtWidgets.QTableWidgetItem(f"{rest_lam:.2f} A")
        rest_item.setFlags(
            qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
            qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        rest_item.setData(qt_api.QtCore.Qt.ItemDataRole.UserRole, 0.0)

        best_matches = [
            x[1]
            for x in lines.get_lines(wrange=[rest_lam - 5, rest_lam + 5])
        ]

        match_item = qt_api.QtWidgets.QTableWidgetItem('; '.join(best_matches))
        match_item.setFlags(
            qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
            qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled
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
        self.msgBox.setIcon(qt_api.QtWidgets.QMessageBox.Icon.Question)
        self.msgBox.setStandardButtons(
            qt_api.QtWidgets.QMessageBox.StandardButton.Yes |
            qt_api.QtWidgets.QMessageBox.StandardButton.No |
            qt_api.QtWidgets.QMessageBox.StandardButton.Cancel
        )
        res = self.msgBox.exec()

        if res == qt_api.QtWidgets.QMessageBox.StandardButton.Yes:
            # If Yes button is pressed, then accept the event only if the
            # project is correctly saved
            event.setAccepted(self.doSaveProject())
        elif res == qt_api.QtWidgets.QMessageBox.StandardButton.No:
            event.accept()
        else:
            # Ignore the close event if Cancel button is pressed
            event.ignore()

        self.saveSettings()

    def collectRedrockResults(self) -> None:
        if self.redrock_handler is None:
            return

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

    def currentSpecItemChanged(
        self,
        new_item: Optional[QtWidgets.QListWidgetItem],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        Update widgets when the current object changes.

        :param new_item: The new list item.
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
            qt_api.QtCore.Qt.ItemDataRole.UserRole
        )

        self.showObjectInfo(spec_uuid)

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

        var: Union[np.ndarray, None] = None
        var_unit: Union[units.Unit, None] = None

        wdisp: Union[np.ndarray, None] = None
        wdisp_unit: Union[units.Unit, None] = None

        sky: Union[np.ndarray, None] = None
        sky_unit: Union[units.Unit, None] = None

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
            wd_data = sp.wd  # type: ignore
        except AttributeError:
            pass
        else:
            if wd_data is not None:
                wdisp = wd_data.value
                wdisp_unit = wd_data.unit

        try:
            sky_data = sp.sky  # type: ignore
        except AttributeError:
            pass
        else:
            if sky_data is not None:
                sky = sky_data.value
                sky_unit = sky_data.unit

        flux_series = values2series(wav, flux, self.qapp.tr("Flux"))
        flux_chart.addSeries(flux_series)

        if not flux_chart.axes():
            flux_axis_x = qt_api.QtCharts.QValueAxis()
            flux_axis_y = qt_api.QtCharts.QValueAxis()

            flux_chart.addAxis(
                flux_axis_x, qt_api.QtCore.Qt.AlignmentFlag.AlignBottom
            )
            flux_chart.addAxis(
                flux_axis_y, qt_api.QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            flux_axis_x = flux_chart.axes()[0]
            flux_axis_y = flux_chart.axes()[1]

        if not var_chart.axes():
            var_axis_x = qt_api.QtCharts.QValueAxis()

            if self.main_wnd.log_y_check_box.isChecked():
                var_axis_y = qt_api.QtCharts.QLogValueAxis()
            else:
                var_axis_y = qt_api.QtCharts.QValueAxis()

            var_chart.addAxis(
                var_axis_x, qt_api.QtCore.Qt.AlignmentFlag.AlignBottom
            )
            var_chart.addAxis(
                var_axis_y, qt_api.QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            var_axis_x = var_chart.axes()[0]
            var_axis_y = var_chart.axes()[1]

        if not wd_chart.axes():
            wd_axis_x = qt_api.QtCharts.QValueAxis()

            if self.main_wnd.log_y_check_box.isChecked():
                wd_axis_y = qt_api.QtCharts.QLogValueAxis()
            else:
                wd_axis_y = qt_api.QtCharts.QValueAxis()

            wd_chart.addAxis(
                wd_axis_x, qt_api.QtCore.Qt.AlignmentFlag.AlignBottom
            )
            wd_chart.addAxis(
                wd_axis_y, qt_api.QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            wd_axis_x = wd_chart.axes()[0]
            wd_axis_y = wd_chart.axes()[1]

        if not sky_chart.axes():
            sky_axis_x = qt_api.QtCharts.QValueAxis()

            if self.main_wnd.log_y_check_box.isChecked():
                sky_axis_y = qt_api.QtCharts.QLogValueAxis()
            else:
                sky_axis_y = qt_api.QtCharts.QValueAxis()

            sky_chart.addAxis(
                sky_axis_x, qt_api.QtCore.Qt.AlignmentFlag.AlignBottom
            )
            sky_chart.addAxis(
                sky_axis_y, qt_api.QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            sky_axis_x = sky_chart.axes()[0]
            sky_axis_y = sky_chart.axes()[1]

        flux_axis_x.setTickInterval(500)
        flux_axis_x.setLabelFormat("%.2f")
        flux_axis_x.setTitleText(str(wav_unit))

        flux_axis_y.setLabelFormat("%.2f")
        flux_axis_y.setTitleText(str(flux_unit))

        flux_series.attachAxis(flux_axis_x)
        flux_series.attachAxis(flux_axis_y)

        if self.main_wnd.smoothing_check_box.isChecked():
            smoothing_factor = self.main_wnd.smoothing_dspinbox.value()

            flux_series.setOpacity(0.2)
            smoothing_sigma = len(flux) / (1 + 2 * smoothing_factor)
            smoothed_flux = utils.smooth_fft(flux, sigma=smoothing_sigma)
            smoothed_flux_series = values2series(
                wav, smoothed_flux, self.qapp.tr("Smoothed flux")
            )

            pen: QtGui.QPen = smoothed_flux_series.pen()
            pen.setColor(qt_api.QtGui.QColor("orange"))
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

            if isinstance(var_axis_y, qt_api.QtCharts.QLogValueAxis):
                var_axis_y.setBase(10.0)
            else:
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

            if not isinstance(wd_axis_y, qt_api.QtCharts.QLogValueAxis):
                wd_axis_y.setBase(10.0)
            else:
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

            if not isinstance(sky_axis_y, qt_api.QtCharts.QLogValueAxis):
                sky_axis_y.setBase(10.0)
            else:
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

    def doAddNewLine(self, *args: Any, **kwargs: Any) -> None:
        """Tell the main app to identify a new line."""
        # Do nothing if no spectrum is currently selected
        if self.current_uuid is None:
            return
        self._lock()
        self.main_wnd.spec_group_box.setEnabled(False)
        self.global_state = GlobalState.SELECT_LINE_MANUAL
        self.qapp.setOverrideCursor(qt_api.QtCore.Qt.CursorShape.CrossCursor)

    def doDeleteAllLines(self, *args: Any, **kwargs: Any) -> None:
        """Delete all identified lines."""
        self.main_wnd.lines_table_widget.setRowCount(0)

    def doDeleteCurrentLine(self, *args: Any, **kwargs: Any) -> None:
        """Delete the current selected line from the table."""
        # Do nothing if no spectrum is currently selected
        if self.current_uuid is None:
            return

        self.main_wnd.lines_table_widget.removeRow(
            self.main_wnd.lines_table_widget.currentRow()
        )

    def doExportAsPicture(self) -> None:
        """Use saveFileDialog to save current spectrum as a file."""
        item: QtWidgets.QListWidgetItem
        item = self.main_wnd.spec_list_widget.currentItem()

        if item is None:
            return
        elif self.current_open_dir is not None:
            cur_dir = self.current_open_dir
        elif self.current_project_file_path is not None:
            cur_dir = os.path.dirname(self.current_project_file_path)
        else:
            cur_dir = '.'

        base_name = os.path.splitext(str(item.text()))[0]

        open_list = qt_api.QtWidgets.QFileDialog.getSaveFileName(
            self.main_wnd,
            self.qapp.tr("Export spectrum as image"),
            os.path.join(cur_dir, f'{base_name}.png'),
            (
                f"{self.qapp.tr('PNG image')} (*.png);;"
                f"{self.qapp.tr('JPG image')} (*.jpg);;"
                f"{self.qapp.tr('TIFF file')} (*.tiff)"
            ),
            f"{self.qapp.tr('PNG image')} (*.png)"
        )

        dest_file_path, _ = open_list

        self.exportAsPicture(dest_file_path)

    def exportAsPicture(self, dest_file_path: str) -> None:
        """
        Save current spectrum to a file.

        :param dest_file_path: Path of the destination picture file.
        """
        try:
            p = self.flux_chart_view.grab()
            p.save(dest_file_path)
        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                self.qapp.tr(
                    "An error has occurred while exporting the "
                    "current spectrum as a picture."
                )
            )
            self.msgBox.setInformativeText('')
            self.msgBox.setDetailedText(str(exc))
            self.msgBox.setIcon(
                qt_api.QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
        else:
            self.current_open_dir = os.path.dirname(dest_file_path)
            self.statusbar.showMessage(
                self.qapp.tr("Picture saved")
            )

    def doExportZcat(self, *args, **kwargs) -> None:
        if self.current_open_dir is not None:
            cur_dir = self.current_open_dir
        elif self.current_project_file_path is not None:
            cur_dir = os.path.dirname(self.current_project_file_path)
        else:
            cur_dir = '.'

        open_list = qt_api.QtWidgets.QFileDialog.getSaveFileName(
            self.main_wnd,
            self.qapp.tr("Export redshift catalogue to file"),
            os.path.join(cur_dir, 'zcat.fits'),
            (
                f"{self.qapp.tr('FITS table')} (*.fits *.fit);;"
                f"{self.qapp.tr('CSV table')} (*.csv);;"
                f"{self.qapp.tr('VO table')} (*.votable)"
            ),
            f"{self.qapp.tr('FITS table')} (*.fits *.fit)"
        )

        try:
            dest_file_path, file_type = open_list
            self.exportZcat(dest_file_path, file_type)
        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                 self.qapp.tr(
                     "An error has occurred while saving the catalogue."
                 )
            )
            self.msgBox.setInformativeText('')
            self.msgBox.setDetailedText(str(exc))
            self.msgBox.setIcon(
                qt_api.QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
        else:
            self.current_open_dir = os.path.dirname(dest_file_path)
            self.statusbar.showMessage(
                self.qapp.tr("Redshift catalogue saved")
            )

    def doIdentifyLines(self,  *args: Any, **kwargs: Any) -> None:
        """Automagically identify lines in the current spectrum."""
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
            new_item = qt_api.QtWidgets.QTableWidgetItem(f"{w:.2f} A")
            new_item.setFlags(
                qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
                qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled |
                qt_api.QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            new_item.setData(qt_api.QtCore.Qt.ItemDataRole.UserRole, w)
            new_item.setCheckState(qt_api.QtCore.Qt.CheckState.Checked)
            self.main_wnd.lines_table_widget.setItem(j, 0, new_item)

    def doImportZcat(self, *args, **kwargs) -> None:
        """
        Open a catalogue file

        :param args: A placeholder.
        :param kwargs: A placeholder.
        :return: THe path of the catalogue file.
        """
        if self.current_open_dir is not None:
            cur_dir = self.current_open_dir
        elif self.current_project_file_path is not None:
            cur_dir = os.path.dirname(self.current_project_file_path)
        else:
            cur_dir = '.'
        catalogue_file, _ = qt_api.QtWidgets.QFileDialog.getOpenFileName(
            self.main_wnd,
            self.qapp.tr("Import a zcat"),
            cur_dir,
            (
                f"{self.qapp.tr('FITS table')} (*.fits *.fit);;"
                f"{self.qapp.tr('CSV table')} (*.csv);;"
                f"{self.qapp.tr('VO table')} (*.votable);;"
                f"{self.qapp.tr('All files')} (*.*)"
            ),
            f"{self.qapp.tr('FITS table')} (*.fits *.fit)"
        )
        self._doImportZcat(catalogue_file)

    def _doImportZcat(self, catalogue_file: Optional[str]) -> None:
        if not catalogue_file:
            return

        try:
            zcat_tbl = Table.read(catalogue_file)
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
                qt_api.QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
            return

        self.current_open_dir = os.path.dirname(catalogue_file)
        self.zcat_mapping_dialog.setup(zcat_tbl)
        self.zcat_mapping_dialog.open()

    def doImportSpectra(self,  *args: Any, **kwargs: Any) -> None:
        """Use QFileDialog to get spectra files."""
        if self.current_open_dir is not None:
            cur_dir = self.current_open_dir
        elif self.current_project_file_path is not None:
            cur_dir = os.path.dirname(self.current_project_file_path)
        else:
            cur_dir = '.'

        file_list, file_types = qt_api.QtWidgets.QFileDialog.getOpenFileNames(
            self.main_wnd,
            self.qapp.tr("Import Spectra"),
            cur_dir,
            (
                f"{self.qapp.tr('FITS')} (*.fit *.fits);;"
                f"{self.qapp.tr('ASCII')} (*.txt *.dat *.cat);;"
                f"{self.qapp.tr('All Files')} (*.*)"
            ),
            f"{self.qapp.tr('FITS')} (*.fit *.fits)"
        )

        self.importSpectra(file_list, file_types)

    def importSpectra(
        self,
        file_list: List[str],
        file_types: Optional[List[str]] = None
    ) -> None:
        """
        Import spectra.

        :param file_list: List of file paths.
        :param file_types:  List of file types.
        """
        exception_tracker: Dict[uuid.UUID, Tuple[str, str]] = {}

        self._lock()
        self.global_state = GlobalState.WAITING

        n_files = len(file_list)

        self.pbar.setMaximum(n_files)
        self.pbar.show()
        self.cancel_button.show()

        al_least_one: bool = False
        for j, file in enumerate(file_list):
            if self.global_state == GlobalState.REUQUEST_CANCEL:
                break

            self.pbar.setValue(j + 1)
            self.statusbar.showMessage(
                self.qapp.tr("Loading file") + f"{j + 1:d}/{n_files}..."
            )
            self.qapp.processEvents()

            item_uuid: uuid.UUID = uuid.uuid4()

            # Check for a possible collision, even if it should never happen
            while item_uuid in self.open_spectra_files.keys():
                item_uuid = uuid.uuid4()

            try:
                sp: Spectrum1D = loaders.read(file)
            except Exception as exc:
                exception_tracker[item_uuid] = (file, str(exc))
                continue
            else:
                if not al_least_one:
                    self.current_open_dir = os.path.dirname(file)
                    al_least_one = True

            new_item = qt_api.QtWidgets.QListWidgetItem(os.path.basename(file))
            new_item.setCheckState(qt_api.QtCore.Qt.CheckState.Checked)
            new_item.setToolTip(file)
            new_item.setData(qt_api.QtCore.Qt.ItemDataRole.UserRole, item_uuid)

            info_dict: Dict[str, Any] = {
                'redshift': 0,
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
        if len(self.open_spectra) > 0:
            self._unlock()
        self.global_state = GlobalState.READY

        if exception_tracker:
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
                        for exc_uuid, exc_data in exception_tracker.items()
                    ]
                )
            )
            self.msgBox.setIcon(qt_api.QtWidgets.QMessageBox.Icon.Critical)
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()

    def doNewProject(self, *args, **kwargs) -> None:
        self.newProject()

    def doOpenProject(self, *args, **kwargs) -> bool:
        if self.current_project_file_path is not None:
            cur_dir = os.path.basename(self.current_project_file_path)
        elif self.current_open_dir is not None:
            cur_dir = self.current_open_dir
        else:
            cur_dir = '.'

        proj_file_path, _ = qt_api.QtWidgets.QFileDialog.getOpenFileName(
            self.main_wnd,
            self.qapp.tr("Load a project from file"),
            cur_dir,
            (
                f"{self.qapp.tr('Project file')} (*.json);;"
                f"{self.qapp.tr('All files')} (*.*)"
            ),
            f"{self.qapp.tr('Project file')} (*.json)"
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
                qt_api.QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
            return False

        self.current_project_file_path = proj_file_path
        return True

    def doRedshiftFromLines(self, *args: Any, **kwargs: Any) -> None:
        """Compute the redshift by matching identified lines."""
        line_table: QtWidgets.QTableWidget
        line_table = self.main_wnd.lines_table_widget

        z_list: QtWidgets.QListWidget
        z_list = self.main_wnd.lines_match_list_widget

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
            elif item.checkState() != qt_api.QtCore.Qt.CheckState.Checked:
                # Ignore lines that are not selected
                continue

            lines_lam.append(
                float(item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole))
            )

        res = lines.get_redshift_from_lines(
            lines_lam, z_min=z_min, z_max=z_max, tol=tol
        )

        if res is None:
            return

        z_list.clear()
        for z, prob in zip(res[0], res[1]):
            new_z_item = qt_api.QtWidgets.QListWidgetItem(
                f"z={z:.4f} (p={prob:.4f})"
            )
            new_z_item.setData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole,
                (z, prob)
            )
            z_list.addItem(new_z_item)

    def doRemoveCurrentSpecItems(self) -> None:
        if self.main_wnd.spec_list_widget.count() == 0:
            return

        current_row = self.main_wnd.spec_list_widget.currentRow()
        item = self.main_wnd.spec_list_widget.item(current_row)
        if item is None:
            return

        item_uuid = item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)
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
            elif item.checkState() != qt_api.QtCore.Qt.CheckState.Checked:
                continue

            item_uuid = item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)
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

    def doSaveProject(self, *args: Any, **kwargs: Any) -> bool:
        """Save the current project."""
        return self.doSaveProjectAs(dest=self.current_project_file_path)

    def doSaveProjectAs(
            self,
            dest: Optional[str] = None,
            *args, **kwargs
    ) -> bool:
        """
        Save the current project to a file.

        :return: True if the project is saved correctly, False otherwise.
        """
        if (dest is None) or (not os.path.exists(str(dest))):
            if self.current_project_file_path is not None:
                cur_dir = self.current_project_file_path
            elif self.current_open_dir is not None:
                cur_dir = self.current_open_dir
            else:
                cur_dir = '.'

            dest_file_path, _ = qt_api.QtWidgets.QFileDialog.getSaveFileName(
                self.main_wnd,
                self.qapp.tr("Save project to file"),
                cur_dir,
                (
                    f"{self.qapp.tr('Project file')} (*.json);;"
                    f"{self.qapp.tr('All files')} (*.*)"
                ),
                f"{self.qapp.tr('Project file')} (*.json)"
            )

            if not dest_file_path:
                return False
        else:
            dest_file_path = str(dest)

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
                qt_api.QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
            return False

        self.statusbar.showMessage(self.qapp.tr("Project saved"))
        self.current_project_file_path = dest_file_path
        return True

    def doStartRedrock(self, *args, **kwargs) -> None:
        if self.redrock_handler is None:
            # Should never happen since we lock the button to start this
            # action if the redrock backend is not found
            return

        self.main_wnd.redrock_text_edit.clear()
        self.statusbar.showMessage("Running redrock backend...")

        for_rr: Dict[uuid.UUID, Spectrum1D] = {}
        if self.main_wnd.redrock_all_radio.isChecked():
            for_rr = self.open_spectra
        elif self.main_wnd.redrock_selected_radio.isChecked():

            item: Union[None, QtWidgets.QListWidgetItem]
            for row in range(self.main_wnd.spec_list_widget.count()):
                item = self.main_wnd.spec_list_widget.item(row)

                if item is None:
                    # should never happen!
                    continue

                if item.checkState() == qt_api.QtCore.Qt.CheckState.Checked:
                    item_uuid = item.data(
                        qt_api.QtCore.Qt.ItemDataRole.UserRole
                    )
                    for_rr[item_uuid] = self.open_spectra[item_uuid]
        else:
            if self.current_uuid is None:
                return
            for_rr[self.current_uuid] = self.open_spectra[self.current_uuid]

        self.redrock_handler.run_redrock(for_rr)

    def doToggleDone(self) -> None:
        check_state = None
        item: Union[QtWidgets.QListWidgetItem, None]
        for row in range(self.main_wnd.spec_list_widget.count()):
            item = self.main_wnd.spec_list_widget.item(row)
            if item is None:
                continue

            item_uuid = item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)

            if item_uuid not in self.object_state_dict:
                continue

            if self.object_state_dict[item_uuid]['quality_flag'] == 0:
                continue

            if check_state is None:
                if item.checkState() == qt_api.QtCore.Qt.CheckState.Checked:
                    check_state = qt_api.QtCore.Qt.CheckState.Unchecked
                else:
                    check_state = qt_api.QtCore.Qt.CheckState.Checked

            item.setCheckState(check_state)

    def doToggleSimilar(self) -> None:
        self.toggleSimilarSpecItems(
            self.main_wnd.spec_list_widget.currentItem()
        )

    def doToggleAll(self) -> None:
        item = self.main_wnd.spec_list_widget.currentItem()
        if item is None:
            return

        if item.checkState() == qt_api.QtCore.Qt.CheckState.Checked:
            ref_check_state = qt_api.QtCore.Qt.CheckState.Unchecked
        else:
            ref_check_state = qt_api.QtCore.Qt.CheckState.Checked

        other_item: Union[None, QtWidgets.QListWidgetItem]
        for row in range(self.main_wnd.spec_list_widget.count()):
            other_item = self.main_wnd.spec_list_widget.item(row)
            if other_item is None:
                continue
            other_item.setCheckState(ref_check_state)

    def doZoomIn(self, *args, **kwargs) -> None:
        """Zoom In flux and var plots."""
        self.flux_chart_view.zoomIn()

    def doZoomOut(self, *args, **kwargs) -> None:
        """Zoom Out flux and var plots."""
        self.flux_chart_view.zoomOut()

    def doZoomReset(self, *arg, **kwargs) -> None:
        """Reset the zoom for the flux chart and all its siblings."""
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

    def getZcatColumnMapping(self, result: int) -> None:

        try:
            self.importZcat(self.zcat_mapping_dialog.zcat_tbl)
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
                qt_api.QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()

    def importZcat(self, zcat_tbl) -> None:
        id_col, z_col, qf_col = self.zcat_mapping_dialog.get_mapping()

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

            spec_file = str(row[id_col]).strip()

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

                    if row[z_col] == -99:
                        info_dict["redshift"] = None
                    else:
                        info_dict["redshift"] = row[z_col]

                    try:
                        info_dict["quality_flag"] = row[qf_col]
                    except Exception:
                        pass
                    else:
                        self._update_spec_item_qf(obj_uuid, row[qf_col])
                    break

        self.global_state = GlobalState.READY
        self.pbar.hide()
        self.cancel_button.hide()
        self._unlock()

    def loadIcons(self) -> None:
        theme_name = str(self.settings_wnd.icon_theme_combo_box.currentText())
        self._load_icons(theme_name)

    def loadSettings(self) -> None:
        for key, (_, set_func, cast_func) in self._global_setting.items():
            if self.settings.contains(key):
                value = self.settings.value(key)
                if value is None:
                    continue
                elif cast_func is not None:
                    value = cast_func(value)
                try:
                    set_func(value)
                except (TypeError, ValueError):
                    continue
        self.loadIcons()

    def mousePressedFlux(self, args) -> None:
        """Handle mouse button pressed events."""
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
        self.current_project_file_path = None
        self.current_open_dir = None

        self.main_wnd.spec_list_widget.clear()
        self.main_wnd.lines_match_list_widget.clear()
        self.main_wnd.lines_table_widget.setRowCount(0)
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
        self.statusbar.showMessage(
            self.qapp.tr("Import some spectra to start!")
        )

    def openProject(self, file_name: str) -> None:
        """Load the project from a file."""
        with open(file_name, 'r') as f:
            serialized_dict: Dict[str, Any] = json.load(f)

        self._lock()
        self.global_state = GlobalState.WAITING
        self.pbar.setMaximum(0)
        self.pbar.show()

        n_files = len(serialized_dict['open_files'])
        exception_tracker: Dict[uuid.UUID, Tuple[str, str]] = {}

        self.pbar.setMaximum(n_files)

        try:
            current_uuid = uuid.UUID(serialized_dict['current_uuid'])
        except Exception:
            current_uuid = None

        current_spec_list_item: Optional[QtWidgets.QListWidgetItem]
        current_spec_list_item = None
        for j, file_info in enumerate(serialized_dict['open_files']):
            self.pbar.setValue(j + 1)
            self.statusbar.showMessage(
                self.qapp.tr("Loading project...")
            )
            self.qapp.processEvents()

            item_uuid = uuid.UUID(file_info['uuid'])
            item_row = int(file_info['index'])

            file_path = os.path.realpath(
                os.path.join(
                    os.path.dirname(file_name),
                    file_info['path']
                )
            )

            if not os.path.exists(file_path):
                if os.path.exists(file_info['path']):
                    file_path = file_info['path']
                else:
                    exception_tracker[item_uuid] = (
                        file_path,
                        f"File {file_path} does not exist!"
                    )
                    continue

            try:
                sp: Spectrum1D = loaders.read(file_path)
            except Exception as exc:
                exception_tracker[item_uuid] = (file_path, str(exc))
                continue

            new_item = qt_api.QtWidgets.QListWidgetItem(file_info['text'])
            new_item.setCheckState(
                qt_api.QtCore.Qt.CheckState(file_info['checked'])
            )
            new_item.setToolTip(file_path)
            new_item.setData(qt_api.QtCore.Qt.ItemDataRole.UserRole, item_uuid)

            if item_uuid == current_uuid:
                current_spec_list_item = new_item

            self.main_wnd.spec_list_widget.addItem(new_item)
            self.open_spectra[item_uuid] = sp
            self.open_spectra_files[item_uuid] = file_path
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
                    'checked': QtCore.Qt.CheckState(
                        line_info['checked']
                    )
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

        if exception_tracker:
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
                        for exc_uuid, exc_data in exception_tracker.items()
                    ]
                )
            )
            self.msgBox.setIcon(qt_api.QtWidgets.QMessageBox.Icon.Critical)
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()

    def openSettingsDialog(self, *args, **kwargs) -> None:
        self.saveSettings()
        self.settings_wnd.open()

    def redrawCurrentSpec(self, *args, **kwargs) -> None:
        """Redraw the charts."""
        self.currentSpecItemChanged(
            self.main_wnd.spec_list_widget.currentItem()
        )

    def requestCancelCurrentOperation(self) -> None:
        """Set the global state of the program to REUQUEST_CANCEL."""
        self.global_state = GlobalState.REUQUEST_CANCEL

    def restoreSettings(self) -> None:
        self.loadSettings()

    def saveProject(self, file_name: str) -> None:
        """Save the project to a file."""
        # Store any pending information for the current object
        self._backup_current_object_state()

        # Serialize program state for json dumping
        serialized_open_file_list = []
        item: Union[QtWidgets.QListWidgetItem, None]
        proj_path = os.path.dirname(file_name)
        for k in range(self.main_wnd.spec_list_widget.count()):
            item = self.main_wnd.spec_list_widget.item(k)
            if item is None:
                continue
            item_uuid: uuid.UUID = item.data(
                qt_api.QtCore.Qt.ItemDataRole.UserRole
            )

            file_info = {
                'index': k,
                'uuid': item_uuid.hex,
                'text': item.text(),
                'path': os.path.relpath(
                    self.open_spectra_files[item_uuid],
                    proj_path
                ),
                'checked': int(item.checkState().value)  # type: ignore
            }

            serialized_open_file_list.append(file_info)

        # Serialize objects info for json dumping
        serialized_object_info_dict: Dict[str, Any] = {}
        for obj_uuid, obj_info in self.object_state_dict.items():
            serialized_lines_list: List[Dict[str, Any]] = []
            serialized_z_list: List[Dict[str, Any]] = []

            for line_info in obj_info['lines']['list']:
                try:
                    serialized_lines_list.append({
                        'row': int(line_info['row']),
                        'data': float(line_info['data']),
                        'text': str(line_info['text']),
                        'checked': int(line_info['checked'].value)
                    })
                except (TypeError, ValueError):
                    continue

            for z_info in obj_info['lines']['redshifts']:
                try:
                    serialized_z_list.append({
                        'row': int(z_info['row']),
                        'text': str(z_info['text']),
                        'data': z_info['data']
                    })
                except (TypeError, ValueError):
                    continue

            serialized_info_dict: Dict[str, Any] = {
                'redshift': float(obj_info['redshift']),
                'quality_flag': int(obj_info['quality_flag']),
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

    def saveSettings(self) -> None:
        for key, (get_func, _, _) in self._global_setting.items():
            self.settings.setValue(key, get_func())

    def setCurrentObjectRedshift(self, redshift: Optional[float]) -> None:
        if redshift is None:
            redshift = 0

        self.flux_chart_view.setRedshift(redshift=redshift)

        item_col_0: Union[QtWidgets.QTableWidgetItem, None]
        item_col_1: Union[QtWidgets.QTableWidgetItem, None]
        item_col_2: Union[QtWidgets.QTableWidgetItem, None]
        for j in range(self.main_wnd.lines_table_widget.rowCount()):
            item_col_0 = self.main_wnd.lines_table_widget.item(j, 0)
            item_col_1 = self.main_wnd.lines_table_widget.item(j, 1)
            item_col_2 = self.main_wnd.lines_table_widget.item(j, 2)

            if item_col_0 is None or item_col_1 is None or item_col_2 is None:
                continue

            line_lam: float = item_col_0.data(
                qt_api.QtCore.Qt.ItemDataRole.UserRole
            )
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

        item: Union[QtWidgets.QListWidgetItem, None]
        item = self.main_wnd.lines_match_list_widget.item(row)
        if item is None:
            return

        try:
            self.main_wnd.z_dspinbox.setValue(
                item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)[0]
            )
        except (IndexError, TypeError):
            return

    def setCurrentObjectRedshiftFromSingleLine(self, row: int) -> None:
        if row < 0:
            return

        lines_tbl_row = self.main_wnd.lines_table_widget.currentRow()
        if lines_tbl_row < 0:
            return

        w_item: Union[None, QtWidgets.QTableWidgetItem]
        w_item = self.main_wnd.lines_table_widget.item(lines_tbl_row, 0)
        if w_item is None:
            return

        obs_lam = w_item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)

        rest_lam = self.main_wnd.single_line_combo_box.itemData(
            row, qt_api.QtCore.Qt.ItemDataRole.UserRole
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

    def showObjectInfo(self, object_uuid: uuid.UUID) -> None:
        sp: Spectrum1D = self.open_spectra[object_uuid]

        self.main_wnd.obj_prop_table_widget.setRowCount(0)

        if sp.meta is None:
            return

        try:
            header = sp.meta['header']
        except KeyError:
            return

        self.main_wnd.obj_prop_table_widget.setRowCount(len(header))
        self.main_wnd.obj_prop_table_widget.setVerticalHeaderLabels(
            list(header.keys())
        )

        val_item: QtWidgets.QTableWidgetItem
        com_item: QtWidgets.QTableWidgetItem
        for j, (key, val, comment) in enumerate(header.cards):

            val_item = qt_api.QtWidgets.QTableWidgetItem(
                str(val)
            )
            val_item.setFlags(
                qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
                qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled
            )

            com_item = qt_api.QtWidgets.QTableWidgetItem(
                str(comment)
            )
            com_item.setFlags(
                qt_api.QtCore.Qt.ItemFlag.ItemIsSelectable |
                qt_api.QtCore.Qt.ItemFlag.ItemIsEnabled
            )

            self.main_wnd.obj_prop_table_widget.setItem(j, 0, val_item)
            self.main_wnd.obj_prop_table_widget.setItem(j, 1, com_item)

    def toggleSimilarSpecItems(
        self,
        item: Optional[QtWidgets.QListWidgetItem]
    ) -> None:
        if item is None:
            return

        ref_uuid = item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)
        if ref_uuid in self.object_state_dict:
            ref_qf = self.object_state_dict[ref_uuid]['quality_flag']
        else:
            ref_qf = 0

        if item.checkState() == qt_api.QtCore.Qt.CheckState.Checked:
            ref_check_state = qt_api.QtCore.Qt.CheckState.Unchecked
        else:
            ref_check_state = qt_api.QtCore.Qt.CheckState.Checked

        other_item: Union[None, QtWidgets.QListWidgetItem]
        for row in range(self.main_wnd.spec_list_widget.count()):
            other_item = self.main_wnd.spec_list_widget.item(row)

            if other_item is None:
                continue

            other_uuid = other_item.data(
                qt_api.QtCore.Qt.ItemDataRole.UserRole
            )
            if other_uuid in self.object_state_dict:
                other_qf = self.object_state_dict[other_uuid]['quality_flag']
            else:
                other_qf = 0

            if other_qf == ref_qf:
                other_item.setCheckState(ref_check_state)

    def toggleSmothing(self, show_smoothing: int) -> None:
        self.redrawCurrentSpec()

    def run(self) -> None:
        """Run the main Qt Application."""
        self.main_wnd.show()
        sys.exit(self.qapp.exec())


def values2series(
    x_values: Union[List[float], np.ndarray],
    y_values: Union[List[float], np.ndarray],
    name: str
) -> qt_api.QtCharts.QLineSeries:
    """
    Convert point values to a QLineSeries object.

    :param x_values: x values of the points.
    :param y_values: y values of the points.
    :param name: The name of the series.
    :return series: The series object.
    """
    series: QtCharts.QLineSeries = qt_api.QtCharts.QLineSeries()
    series.setName(name)
    series.setUseOpenGL(False)  # issues with transparency when set to True!
    for x, y in zip(x_values, y_values):
        series.append(x, y)

    return series


def main() -> None:
    """Run the main GUI application using PySide."""
    myapp: GuiApp = GuiApp()
    myapp.run()


if __name__ == '__main__':
    main()
