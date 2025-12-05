#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: Maurizio D'Addona
"""
from __future__ import annotations

import os
import copy
import pickle
import sys
import json
import uuid
import pathlib
import webbrowser
import traceback
import hashlib
from enum import Enum
from typing import TYPE_CHECKING
from typing import Optional, Union, Tuple, List, Dict, Callable, Any, cast

import numpy as np
import specutils
from astropy.nddata import VarianceUncertainty  # type: ignore
from astropy.nddata import StdDevUncertainty  # type: ignore
from astropy.nddata import InverseVariance  # type: ignore
from astropy import units  # type: ignore
from astropy.table import Table  # type: ignore
from scipy.ndimage import gaussian_filter

try:
    from specutils import Spectrum
except ImportError:
    # The Spectrum1D class is deprecated and may be removed in a future version
    from specutils import Spectrum1D as Spectrum

import redmost
from redmost import loaders
from redmost import utils
from redmost import lines
from redmost import backends

from redmost import splashscreen

from redmost.qt_compat import qt_api, get_qapp

try:
    from matplotlib import colormaps
except ImportError:
    HAS_MPL = False
else:
    HAS_MPL = True

if TYPE_CHECKING:
    # NOTE: These imports will be actually executed only by the type-checking
    #       programs, like mypy.
    from redmost.qt_compat import QtCore, QtGui, QtWidgets, QtCharts
    from redmost.qt_compat import Signal


def bool2tr(val: bool, qapp: Any = get_qapp()):
    return qapp.tr("YES") if val else qapp.tr("NO")


def _open_online_doc(*args: Any, **kwargs: Any) -> None:
    webbrowser.open(redmost.ONLINE_DOC_URL)


class SpectrumQChartView(qt_api.QtCharts.QChartView):
    """Subclass of qt_api.QtCharts.QChartView with advanced features."""
    onMouseMoveSeries = qt_api.Signal(object)
    onMousePressSeries = qt_api.Signal(object)
    onMouseReleaseSeries = qt_api.Signal(object)
    onMouseDoubleClickSeries = qt_api.Signal(object)
    onMouseWheelEvent = qt_api.Signal(object)
    maskSelected = qt_api.Signal(object)
    maskSelectedIndexChanged = qt_api.Signal(int)
    maskHighlighted = qt_api.Signal(object)
    maskHighlightedIndex = qt_api.Signal(int)

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
        self.masks: List[Dict[str, Any]] = []
        self.allow_empty: bool = False

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
            try:
                p1x, p1y = np.min(data, axis=0)[0:2]
                p2x, p2y = np.max(data, axis=0)[0:2]
            except ValueError:
                continue

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

            if y_min == y_max:
                y_max = y_min + 1

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

        if (not self.allow_empty) and (len(self.chart().series()) == 0):
            return

        palette: QtGui.QPalette = self.palette()
        painter.save()
        plot_area: QtCore.QRectF = self.chart().plotArea()
        pen: QtGui.QPen = qt_api.QtGui.QPen()
        brush: QtGui.QBrush = qt_api.QtGui.QBrush()

        pen_color: QtGui.QColor
        pen_color = qt_api.QtGui.QColor("#aa333333")
        if self._mouse_lambda is not None:
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

        for mask in self.masks:
            try:
                if not mask['enabled']:
                    continue
            except KeyError:
                pass

            try:
                mask_color = qt_api.QtGui.QColor(mask['color'])
            except Exception:
                mask_color = qt_api.QtGui.QColor("#aa333333")

            try:
                mask_color.setAlphaF(mask['alpha'])
            except KeyError:
                mask_color.setAlphaF(0.5)

            pen.setColor(mask_color)
            pen.setWidthF(1)
            brush.setColor(mask_color)
            try:
                brush.setStyle(qt_api.QtCore.Qt.BrushStyle[mask['pattern']])
            except KeyError:
                brush.setStyle(qt_api.QtCore.Qt.BrushStyle.BDiagPattern)

            painter.setPen(pen)
            painter.setBrush(brush)

            mask_x_start = self.chart().mapToPosition(
                qt_api.QtCore.QPointF(mask['start'], 0)
            ).x()

            mask_x_end = self.chart().mapToPosition(
                qt_api.QtCore.QPointF(mask['end'], 0)
            ).x()

            if (
                (mask_x_end < plot_area.left()) or
                (mask_x_start > plot_area.right())
            ):
                continue

            if mask_x_end >= plot_area.right():
                mask_x_end = plot_area.right()

            if mask_x_start <= plot_area.left():
                mask_x_start = plot_area.left()

            mask_p1: QtCore.QPointF = qt_api.QtCore.QPointF(
                mask_x_start, plot_area.top()
            )
            mask_p2: QtCore.QPointF = qt_api.QtCore.QPointF(
                mask_x_end, plot_area.bottom()
            )
            mask_rect: QtCore.QRectF = qt_api.QtCore.QRectF(mask_p1, mask_p2)

            path: QtGui.QPainterPath = qt_api.QtGui.QPainterPath()
            path.addRect(mask_rect)
            painter.fillPath(path, brush)

            try:
                if mask["selected"]:
                    pen.setDashPattern((1,))
                    pen.setWidthF(2)
                    pen.setColor(palette.highlight().color())
                    painter.setPen(pen)
                    painter.drawRect(mask_rect)
                elif mask["highlighted"]:
                    pen.setDashPattern((6, 6, 6, 6))
                    pen.setColor(palette.highlight().color())
                    pen.setWidthF(2)
                    painter.setPen(pen)
                    painter.drawRect(mask_rect)
            except KeyError:
                pass

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
                p_line_top: QtCore.QPointF = qt_api.QtCore.QPointF(
                    line_x, plot_area.top()
                )
                p_line_bottom: QtCore.QPointF = qt_api.QtCore.QPointF(
                    line_x, plot_area.bottom()
                )
                p_text_top: QtCore.QPointF = qt_api.QtCore.QPointF(
                    line_x, plot_area.top() - 5
                )
                painter.drawLine(p_line_bottom, p_line_top)
                painter.drawText(p_text_top, line_name)

        painter.restore()

    def isMouseOnMask(
        self, mask: Dict[str, Any], mouse_pos: QtCore.QPointF,
        plot_area: Optional[QtCore.QRectF] = None
    ) -> bool:

        if plot_area is None:
            plot_area: QtCore.QRectF = self.chart().plotArea()

        try:
            if not mask['enabled']:
                return False
        except KeyError:
            pass

        mask_x_start = self.chart().mapToPosition(
            qt_api.QtCore.QPointF(mask['start'], 0)
        ).x()

        mask_x_end = self.chart().mapToPosition(
            qt_api.QtCore.QPointF(mask['end'], 0)
        ).x()

        if (
                (mask_x_end < plot_area.left()) or
                (mask_x_start > plot_area.right())
        ):
            return False

        if mask_x_end >= plot_area.right():
            mask_x_end = plot_area.right()

        if mask_x_start <= plot_area.left():
            mask_x_start = plot_area.left()

        if mask_x_start <= mouse_pos.x() <= mask_x_end:
            return True

        return False

    def getMasksUnderCursor(self, mouse_pos: QtCore.QPointF) -> list[int]:
        """
        Get a list ov masks under the mouse pointer.

        :param mouse_pos: The mouse position
        :returns: A list of mask indices
        """

        mask_indices: list[int] = []
        plot_area: QtCore.QRectF = self.chart().plotArea()
        for mask_idx, mask in enumerate(self.masks):
            if self.isMouseOnMask(mask, mouse_pos, plot_area=plot_area):
                mask_indices.append(mask_idx)
        return  mask_indices

    def setMasks(self, masks: List[Dict[str, Any]]) -> None:
        self.masks = masks

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

        if event.buttons():
            if qt_api.QtCore.Qt.MouseButton.MiddleButton:
                if self._last_mouse_pos is None:
                    return

                delta = mouse_pos - self._last_mouse_pos
                self._last_mouse_pos = mouse_pos

                self.scroll(-delta.x(), delta.y())
        else:
            for mask_index, mask in enumerate(self.masks):
                if self.isMouseOnMask(mask, mouse_pos):
                    mask["highlighted"] = True
                    self.maskHighlighted.emit(mask)
                    self.maskHighlightedIndex.emit(mask_index)
                else:
                    mask["highlighted"] = False

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

        try:
            event_position = event.position()
        except AttributeError:
            event_position = event.pos()

        if event.button() == qt_api.QtCore.Qt.MouseButton.MiddleButton:
            get_qapp().setOverrideCursor(
                qt_api.QtCore.Qt.CursorShape.SizeAllCursor
            )
            self._last_mouse_pos = event_position
            event.accept()
        elif event.button() == qt_api.QtCore.Qt.MouseButton.LeftButton:
            for mask in self.masks:
                mask["selected"] = False

            for mask_index, mask in enumerate(self.masks):
                if self.isMouseOnMask(mask, event_position):
                    mask["selected"] = True
                    self.maskSelected.emit(mask)
                    self.maskSelectedIndexChanged.emit(mask_index)
                    break

            event.accept()

        super().mousePressEvent(event)
        self.onMousePressSeries.emit((self.toSeriesPos(event), event))
        self.chart().update()
        self.update()
        self._sibling_locked = True
        for sibling in self.siblings:
            sibling.chart().update()
            sibling.update()
        self._sibling_locked = False

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

    def setLinesType(self, line_type: str) -> None:
        self._lines_type = line_type
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

    def resizeEvent(self, event: QtGui.QResizeEvent = None) -> None:
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


class SpectrumQChartView2D(SpectrumQChartView):

    def __init__(self, *args: Any) -> None:
        self._orig_data: Union[np.ndarray, None] = None
        self._data: Union[np.ndarray, None] = None
        self._spectral_axis: Union[np.ndarray, None] = None
        self.spec_qimg: Union[QtGui.QImage, None] = None
        self.cmap: Union[Callable, None] = None
        self.vmin: Union[float, None] = None
        self.vmax: Union[float, None] = None
        self.start_1d: Union[float, None] = None
        self.end_1d: Union[float, None] = None
        super().__init__(*args)

    def setColorMap(self, cmap, update=True):
        self.cmap = cmap
        if update:
            self.update()

    def setVmin(self, vmin, update=True):
        self.vmin = vmin
        if update:
            self.update()

    def setVmax(self, vmax, update=True):
        self.vmax = vmax
        if update:
            self.update()

    def update(self):
        self.setSpectrumImage(self._orig_data)
        self.chart().update()
        super().update()

    def setSpectralAxis(self, wav_angstrom: Union[np.ndarray, List, Tuple]):
        self._spectral_axis = np.asarray(wav_angstrom)

    def setSpectrumImage(self, data: np.ndarray) -> None:
        self._orig_data = data
        if data is None:
            return

        if self.cmap is not None:
            x: np.ndarray = np.clip(data, a_min=self.vmin, a_max=self.vmax)
            x = x - np.nanmin(x)
            x = x / np.nanmax(x)
            x[~np.isfinite(x)] = 0
            self._data = (255 * self.cmap(x)[..., 0:3]).astype(np.uint8).copy()
            img_format = qt_api.QtGui.QImage.Format.Format_RGB888
            bytes_per_line = 3 * self._data.shape[1]
        else:
            x: np.ndarray = data - np.nanmin(data)
            x[~np.isfinite(x)] = 0
            img_format = qt_api.QtGui.QImage.Format.Format_Grayscale8
            self._data = (255 * (x / np.nanmax(x))).astype(np.uint8).copy()
            bytes_per_line = self._data.shape[1]

        self.spec_qimg = qt_api.QtGui.QImage(
            self._data.data,
            self._data.shape[1], self._data.shape[0],
            bytes_per_line, img_format
        )

    def drawForeground(
            self,
            painter: QtGui.QPainter,
            rect: QtCore.QRectF
    ) -> None:
        if self.spec_qimg is not None:

            plot_area: QtCore.QRectF = self.chart().plotArea()
            if self._spectral_axis is None:
                qimg_rect = plot_area
            else:
                x_left = self.chart().mapToPosition(
                    qt_api.QtCore.QPointF(np.nanmin(self._spectral_axis), 0.0)
                ).x()

                x_rigt = self.chart().mapToPosition(
                    qt_api.QtCore.QPointF(
                        np.nanmax(self._spectral_axis), self.spec_qimg.height()
                    )
                ).x()

                top_left: QtCore.QPointF = qt_api.QtCore.QPointF(
                    x_left, plot_area.top()
                )
                bottom_right: QtCore.QPointF = qt_api.QtCore.QPointF(
                    x_rigt, plot_area.bottom()
                )

                qimg_rect: QtCore.QRectF = qt_api.QtCore.QRectF(
                    top_left, bottom_right
                )

            painter.drawImage(qimg_rect, self.spec_qimg)
            painter.restore()

            # Draw 1D start and end vertical limits
            pen: QtGui.QPen = qt_api.QtGui.QPen()
            pen_color: QtGui.QColor = qt_api.QtGui.QColor("red")
            pen_color.setAlphaF(0.7)

            pen.setColor(pen_color)
            pen.setWidthF(2)
            painter.setPen(pen)

            plot_area_h = plot_area.bottom() - plot_area.top()

            if self.start_1d is not None:
                spec_start_y = self.start_1d / self._data.shape[0]
                start_line_left = qt_api.QtCore.QPointF(
                    plot_area.left(),
                    plot_area.top() + plot_area_h * spec_start_y
                )
                start_line_right = qt_api.QtCore.QPointF(
                    plot_area.right(),
                    plot_area.top() + plot_area_h * spec_start_y
                )
                painter.drawLine(start_line_left, start_line_right)

            if self.end_1d is not None:
                spec_end_y =  self.end_1d / self._data.shape[0]
                end_line_left = qt_api.QtCore.QPointF(
                    plot_area.left(),
                    plot_area.top() + plot_area_h * spec_end_y
                )
                end_line_right = qt_api.QtCore.QPointF(
                    plot_area.right(),
                    plot_area.top() + plot_area_h * spec_end_y
                )
                painter.drawLine(end_line_left, end_line_right)

        super().drawForeground(painter, rect)


class XCorrQChartView(qt_api.QtCharts.QChartView):
    """Subclass of qt_api.QtCharts.QChartView with advanced features."""
    onMouseMoveSeries = qt_api.Signal(object)
    onMousePressSeries = qt_api.Signal(object)
    onMouseReleaseSeries = qt_api.Signal(object)
    onMouseDoubleClickSeries = qt_api.Signal(object)
    onMouseWheelEvent = qt_api.Signal(object)

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

        self._last_mouse_pos: QtCore.QPointF | None = None
        self._best_z: Union[None, float] = None
        self._mouse_z: Union[None, QtCore.QPointF] = None

        self.master: bool = False
        self.horizontal_lock: bool = True
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

    def setRedshift(self, best_z: Optional[float] = None):
        self._best_z = best_z

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

        if self._best_z is not None:
            pen_color = qt_api.QtGui.QColor("#aa333333")
            pen.setColor(pen_color)
            pen.setWidthF(1.0)
            pen.setDashPattern((6, 6, 6, 6))
            painter.setPen(pen)

            best_z_line_x = self.chart().mapToPosition(
                qt_api.QtCore.QPointF(self._best_z, 0.0)
            ).x()

            p_line_top = qt_api.QtCore.QPointF(
                best_z_line_x, plot_area.top()
            )
            p_line_bottom = qt_api.QtCore.QPointF(
                best_z_line_x, plot_area.bottom()
            )
            painter.drawLine(p_line_bottom, p_line_top)
            painter.drawText(p_line_bottom, f"{self._best_z:.4}")

        if self._mouse_z is not None:
            pen_color = qt_api.QtGui.QColor("#aa333333")
            pen.setColor(pen_color)
            pen.setWidthF(1.0)
            pen.setDashPattern((6, 6, 6, 6))
            painter.setPen(pen)

            mouse_line_x = self.chart().mapToPosition(
                qt_api.QtCore.QPointF(self._mouse_z, 0.0)
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

        painter.restore()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        Handle mouse double click events.

        :param event: The input event.
        """
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

        self._mouse_z = data_pos.x()

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
        if event.button() == qt_api.QtCore.Qt.MouseButton.MiddleButton:
            get_qapp().restoreOverrideCursor()
            event.accept()
        elif event.button() == qt_api.QtCore.Qt.MouseButton.RightButton:
            qt_api.QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)
        else:
            # Passtrough for rubber band handling
            super().mouseReleaseEvent(event)

        self.onMouseReleaseSeries.emit((self.toSeriesPos(event), event))


    def scroll(self, dx: float, dy: float) -> None:
        chart = self.chart()

        if self.vertical_lock and self.horizontal_lock:
            return
        elif not (self.vertical_lock or self.horizontal_lock):
            chart.scroll(dx, dy)
        elif self.vertical_lock:
            chart.scroll(dx, 0)
        else:
            chart.scroll(0, dy)

    def resizeEvent(self, event: QtGui.QResizeEvent = None) -> None:
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

        rect = self.chart().plotArea()

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
        self.process.readyReadStandardError.connect(  # type: ignore
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
            if "finished in" in simplified:
                self._total_progress += 1
                self.progress.emit(self._total_progress)
            elif "computing redshifts took:" in simplified:
                self.progress.emit(self._total_maximum)

        self.message.emit(str(text))

    def retrieve_results(self, exit_status: int) -> None:
        if exit_status != 0:
            self.message.emit("*** REDROCK EXECUTION FAILED ***")
            self.results = None
        else:
            try:
                with open(self._result_dump_file, 'rb') as f:
                    self.results = pickle.load(f)
            except Exception:
                self.message.emit("*** REDROCK EXECUTION FAILED ***")
                self.results = None
        self.finished.emit()

    def run(self, spectra: Dict[uuid.UUID, Spectrum]) -> None:
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


class QXCorrHandler(qt_api.QtCore.QObject):
    """Class to handle xcorr subprocess."""

    progress: Signal = qt_api.Signal(int)
    maximum: Signal = qt_api.Signal(int)
    reset: Signal = qt_api.Signal()
    message: Signal = qt_api.Signal(str)
    finished: Signal = qt_api.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.qapp = get_qapp()
        self.templates: Union[None, Dict[str, Spectrum]] = None
        self.use_gpu: Union[str, bool] = False
        self.gpu_device: Union[None, str] = None
        self.z_min: float = 0
        self.z_max: float = 6
        self.z_step: float = 0.001
        self.w_min_ang: Union[float, None] = None
        self.w_max_ang: Union[float, None] = None
        self.batch_size: int = 32
        self.subtract_continuum: int = 1
        self.continuum_smoothing_level: float = 50
        self.chebyshev_order: int = 5
        self.best_z: Dict[uuid.UUID: Dict[str: float]] = {}
        self.corr_data: Dict[uuid.UUID: Dict[str: np.ndarray]] = {}

    def setGpuBackend(self, use_gpu: Union[str, bool]) -> None:
        self.use_gpu = use_gpu

    def setGpuDevice(self, device: Optional[str] = None) -> None:
        self.gpu_device = device

    def setContinuumSubtraction(self, index: int) -> None:
        self.subtract_continuum = index

    def setFFTSmoothing(self, val: float) -> None:
        self.continuum_smoothing_level = val

    def setChebyshevOrder(self, order: int) -> None:
        self.chebyshev_order = order

    def clear(self) -> None:
        self.best_z = {}
        self.corr_data = {}

    def run(
        self,
        spectra: Dict[uuid.UUID, Spectrum],
        w_masks: Optional[Dict[uuid.UUID, List[Dict[str, Any]]]] = None
    ) -> None:
        self.reset.emit()
        self.maximum.emit(0)
        self.maximum.emit(100)

        if self.use_gpu:
            msg_header = (
                self.qapp.tr("Running X-Corr backend on GPU")
                + f" [{self.gpu_device}]"
            )
        else:
            msg_header = self.qapp.tr("Running X-Corr backend on CPU")

        if not self.subtract_continuum:
            sub_mode = None
        elif self.subtract_continuum == 1:
            sub_mode = 'chebyshev'
        else:
            sub_mode = 'fft'

        msg_header += f" [c.sub.={sub_mode}]"

        self.message.emit(msg_header)

        def progress_callback(prog, total, text: str):
            if total == 0:
                self.maximum.emit(0)
                self.progress.emit(0)
            else:
                self.maximum.emit(total)
                self.progress.emit(prog)
            self.message.emit(f"{msg_header}: {text}")
            self.qapp.processEvents()

        # NOTE: xcorr_2d if faster xcorr_1d when run on multiple objects
        #       even when run on CPU, so we stick with it.
        results = backends.xcorr.xcorr_2d(
            spectra=spectra,
            templates=self.templates,
            z_min=self.z_min,
            z_max=self.z_max,
            z_step=self.z_step,
            w_min=self.w_min_ang,
            w_max=self.w_max_ang,
            w_masks=w_masks,
            use_gpu=self.use_gpu,
            device=self.gpu_device,
            continuum_subtracted=sub_mode,
            batch_size=self.batch_size,
            progress_callback=progress_callback,
            fft_sigma=self.continuum_smoothing_level,
            chebyshev_order=self.chebyshev_order,
        )

        self.best_z.update(results[0])
        self.corr_data.update(results[1])

        self.reset.emit()
        self.message.emit(self.qapp.tr("X-Corr done!"))
        self.finished.emit()


class GlobalState(Enum):
    READY = 0
    WAITING = 1
    SELECT_LINE_MANUAL = 2
    SAVE_OBJECT_STATE = 3
    LOAD_OBJECT_STATE = 4
    REUQUEST_CANCEL = 5


class ControlMappingDialog(qt_api.QtWidgets.QDialog):
    control_table_widget: QtWidgets.QTableWidget


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
            qapp.tr("HAS REDROCK"): bool2tr(backends.HAS_REDROCK),
            qapp.tr("HAS X-CORR"): bool2tr(backends.HAS_XCORR),
            qapp.tr("HAS GPU SUPPORT"): bool2tr(
                utils.HAS_TENSORFLOW or
                utils.HAS_TORCH
            ),
            qapp.tr("HAS PYTORCH"): bool2tr(utils.HAS_TORCH),
            qapp.tr("HAS TENSORFLOW"): bool2tr(utils.HAS_TENSORFLOW)
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
    gpu_backend_combo_box: QtWidgets.QComboBox
    gpu_device_combo_box: QtWidgets.QComboBox
    template_dirs_list_widget: QtWidgets.QListWidget
    template_files_list_widget: QtWidgets.QListWidget
    add_template_dir_push_button: QtWidgets.QPushButton
    remove_template_dir_push_button: QtWidgets.QPushButton


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
    action_mask_copy: QtGui.QAction
    action_mask_paste: QtGui.QAction
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
    cmap_combo_box: QtWidgets.QComboBox
    cmap_max_dspinbox: QtWidgets.QDoubleSpinBox
    cmap_min_dspinbox: QtWidgets.QDoubleSpinBox
    container_tab_var: QtWidgets.QWidget
    container_tab_wd: QtWidgets.QWidget
    container_tab_sky: QtWidgets.QWidget
    delete_line_button: QtWidgets.QPushButton
    delete_lines_button: QtWidgets.QPushButton
    export_zcat_button: QtWidgets.QPushButton
    flux_tabwidget: QtWidgets.QTableWidget
    flux_widget_layout_1d: QtWidgets.QHBoxLayout
    flux_widget_layout_2d: QtWidgets.QHBoxLayout
    import_zcat_button: QtWidgets.QPushButton
    info_tab_widget: QtWidgets.QTabWidget
    lines_auto_button: QtWidgets.QPushButton
    lines_tol_dspinbox: QtWidgets.QDoubleSpinBox
    lines_match_list_widget: QtWidgets.QListWidget
    lines_table_widget: QtWidgets.QTableWidget
    log_y_check_box: QtWidgets.QCheckBox
    mask_add_button: QtWidgets.QPushButton
    mask_trasparency_spinbox: QtWidgets.QSpinBox
    mask_apply_all_checkbox: QtWidgets.QCheckBox
    mask_apply_button: QtWidgets.QPushButton
    mask_cancel_button: QtWidgets.QPushButton
    mask_color_textbox: QtWidgets.QLineEdit
    mask_delete_button: QtWidgets.QPushButton
    mask_deleteall_button: QtWidgets.QPushButton
    mask_desc_textbox: QtWidgets.QLineEdit
    mask_enabled_checkbox: QtWidgets.QCheckBox
    mask_end_spinbox: QtWidgets.QSpinBox
    mask_fill_combobox: QtWidgets.QComboBox
    mask_name_textbox: QtWidgets.QLineEdit
    mask_start_spinbox: QtWidgets.QSpinBox
    mask_list_widget: QtWidgets.QListWidget
    mask_load_button: QtWidgets.QPushButton
    mask_properties_groupbox: QtWidgets.QGroupBox
    mask_save_button: QtWidgets.QPushButton
    masking_tab: QtWidgets.QWidget
    match_lines_button: QtWidgets.QPushButton
    next_spec_button: QtWidgets.QPushButton
    obj_prop_table_widget: QtWidgets.QTableWidget
    other_charts_tab_widget: QtWidgets.QTabWidget
    plot_group_box: QtWidgets.QGroupBox
    plot_templates_check_box: QtWidgets.QCheckBox
    previous_spec_button: QtWidgets.QPushButton
    qflag_spinbox: QtWidgets.QSpinBox
    redshift_tools_tab: QtWidgets.QGroupBox
    redrock_all_radio: QtWidgets.QRadioButton
    redrock_selected_radio: QtWidgets.QRadioButton
    redrock_run_button: QtWidgets.QPushButton
    redrock_text_edit: QtWidgets.QTextEdit
    redrock_progress_bar: QtWidgets.QProgressBar
    redrock_current_radio: QtWidgets.QRadioButton
    remove_selected_button: QtWidgets.QPushButton
    remove_spec_button: QtWidgets.QPushButton
    show_legend_check_box: QtWidgets.QCheckBox
    show_lines_check_box: QtWidgets.QCheckBox
    show_lines_combo_box: QtWidgets.QComboBox
    single_line_combo_box: QtWidgets.QComboBox
    sky_widget_layout: QtWidgets.QHBoxLayout
    smoothing_1d_dspinbox: QtWidgets.QDoubleSpinBox
    smoothing_1d_check_box: QtWidgets.QCheckBox
    smoothing_2d_dspinbox: QtWidgets.QDoubleSpinBox
    smoothing_2d_check_box: QtWidgets.QCheckBox
    spec_group_box: QtWidgets.QGroupBox
    spec_list_widget: QtWidgets.QListWidget
    splitter_main: QtWidgets.QSplitter
    splitter_plots: QtWidgets.QSplitter
    toggle_all_button: QtWidgets.QPushButton
    toggle_done_button: QtWidgets.QPushButton
    toggle_similar_button: QtWidgets.QPushButton
    tools_tabwidget: QtWidgets.QTableWidget
    var_widget_layout: QtWidgets.QHBoxLayout
    wdisp_widget_layout: QtWidgets.QHBoxLayout
    xcorr_all_radio: QtWidgets.QRadioButton
    xcorr_csub_combobox: QtWidgets.QComboBox
    xcorr_csub_chebyshev_order_spinbox: QtWidgets.QSpinBox
    xcorr_csub_fft_sigma_dspinbox: QtWidgets.QDoubleSpinBox
    xcorr_csub_preview_checkbox: QtWidgets.QCheckBox
    xcorr_current_radio: QtWidgets.QRadioButton
    xcorr_gpu_checkbox: QtWidgets.QCheckBox
    xcorr_preview_button: QtWidgets.QPushButton
    xcorr_run_button: QtWidgets.QPushButton
    xcorr_selected_radio: QtWidgets.QRadioButton
    xcorr_templates_deselect_all_button: QtWidgets.QPushButton
    xcorr_templates_select_all_button: QtWidgets.QPushButton
    xcorr_templates_tree_widget: QtWidgets.QTreeWidget
    xcorr_widget_layout: QtWidgets.QHBoxLayout
    xcorr_wav_max_spinbox: QtWidgets.QSpinBox
    xcorr_wav_min_spinbox: QtWidgets.QSpinBox
    xcorr_wrange_checkbox: QtWidgets.QCheckBox
    xcorr_use_z_button: QtWidgets.QPushButton
    xcorr_z_max_dspinbox: QtWidgets.QDoubleSpinBox
    xcorr_z_min_dspinbox: QtWidgets.QDoubleSpinBox
    xcorr_z_step_dspinbox: QtWidgets.QDoubleSpinBox
    z_dspinbox: QtWidgets.QDoubleSpinBox
    z_min_dspinbox: QtWidgets.QDoubleSpinBox
    z_max_dspinbox: QtWidgets.QDoubleSpinBox
    z_comment_textbox: QtWidgets.QLineEdit


class MaskListWidgetItem(qt_api.QtWidgets.QWidget):

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.label_name: QtWidgets.QLabel = qt_api.QtWidgets.QLabel()
        self.label_flags: QtWidgets.QLabel = qt_api.QtWidgets.QLabel()
        self.mask_preview: MaskPreviewWidget = MaskPreviewWidget()

        self.mask_preview.setMinimumWidth(30)
        self.mask_preview.setSizePolicy(
            qt_api.QtWidgets.QSizePolicy.Policy.Minimum,
            qt_api.QtWidgets.QSizePolicy.Policy.Minimum,
        )

        self.label_flags.setMinimumWidth(30)
        self.label_flags.setSizePolicy(
            qt_api.QtWidgets.QSizePolicy.Policy.Minimum,
            qt_api.QtWidgets.QSizePolicy.Policy.Minimum,
        )

        self.label_name.setSizePolicy(
            qt_api.QtWidgets.QSizePolicy.Policy.Expanding,
            qt_api.QtWidgets.QSizePolicy.Policy.Minimum,
        )

        main_layout = qt_api.QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.mask_preview)
        main_layout.addWidget(self.label_flags)
        main_layout.addWidget(self.label_name)

        self.setLayout(main_layout)

    def setName(self, text: str) -> None:
        self.label_name.setText(text)
        self.label_flags.adjustSize()
        self.label_name.adjustSize()

    def setFlagText(self, text: str) -> None:
        self.label_flags.setText(text)
        self.label_flags.adjustSize()
        self.label_name.adjustSize()

    def setPreviewMask(self, mask: Dict[str, Any]) -> None:
        self.mask_preview.setMask(mask)

    def clearPreviewMask(self) -> None:
        self.mask_preview.clearMask()


class MaskPreviewWidget(qt_api.QtWidgets.QWidget):

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.mask: Union[None, Dict[str, Any]] = None

    def setMask(self: QtWidgets.QWidget, mask: Dict[str, Any]) -> None:
        self.mask = mask
        self.repaint()

    def clearMask(self: QtWidgets.QWidget) -> None:
        self.mask = None
        self.repaint()

    def paintEvent(self: QtWidgets.QWidget, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)

        if self.mask is None:
            return

        painter: QtGui.QPainter = qt_api.QtGui.QPainter(self)
        painter.save()

        plot_area: QtCore.QRectF = qt_api.QtCore.QRectF(
            0, 0, self.width(), self.height()
        )
        pen: QtGui.QPen = qt_api.QtGui.QPen()
        brush: QtGui.QBrush = qt_api.QtGui.QBrush()

        try:
            mask_color = qt_api.QtGui.QColor(self.mask['color'])
        except Exception:
            mask_color = qt_api.QtGui.QColor("#aa333333")

        try:
            mask_color.setAlphaF(self.mask['alpha'])
        except KeyError:
            mask_color.setAlphaF(0.5)

        pen.setColor(mask_color)
        pen.setWidthF(1)
        brush.setColor(mask_color)

        try:
            brush.setStyle(qt_api.QtCore.Qt.BrushStyle[self.mask['pattern']])
        except KeyError:
            brush.setStyle(qt_api.QtCore.Qt.BrushStyle.BDiagPattern)

        painter.setPen(pen)
        painter.setBrush(brush)

        path: QtGui.QPainterPath = qt_api.QtGui.QPainterPath()
        path.addRect(plot_area)
        painter.fillPath(path, brush)

        painter.restore()


class GuiApp:
    """General class for the main GUI."""

    DEFAULT_MASK_PROPERTIES = {
        "name": "Mask",
        "description": "",
        "start": 0,
        "end": 0,
        "color": "red",
        "pattern": "SolidPattern",
        "alpha": 0.2,
        "enabled": True,
        "global": False,
        "linked": None,
        "selected": False,
        "highlighted": False
    }

    DEFAULT_SPEC_INFO = {
        "redshift": 0,
        "quality_flag": 0,
        "cmap": "viridis",
        "vmin": None,
        "vmax": None,
        "lines": {
            'list': [],
            'redshifts': []
        }
    }

    def __init__(self) -> None:
        self.qapp: QtWidgets.QApplication = get_qapp()
        self.qapp.setApplicationName("Redmost")
        self.qapp.setOrganizationName("DAddona")
        self.qapp.setOrganizationDomain(
            "github.com/mauritiusdadd/redmost"
        )

        self.splash = splashscreen.SplashScreen()
        self.splash.setValue(0)
        self.splash.showMessage(self.qapp.tr("Initializing..."))

        self.settings: QtCore.QSettings = qt_api.QtCore.QSettings()

        self.open_spectra: Dict[uuid.UUID, Spectrum] = {}
        self.open_spectra_2d: Dict[uuid.UUID, loaders.Spectrum2D] = {}
        self.open_spectra_items: Dict[
            uuid.UUID, QtWidgets.QListWidgetItem
        ] = {}

        self.spectral_masks: Dict[
            uuid.UUID, Dict[uuid.UUID, Dict[str, Any]]
        ] = {}

        self.current_uuid: Optional[uuid.UUID] = None
        self.object_state_dict: Dict[uuid.UUID, Any] = {}
        self.current_project_file_path: Optional[str] = None
        self.current_open_dir: Union[None, str] = None
        self.current_zcat_open_dir: Union[None, str] = None
        self.current_image_open_dir: Union[None, str] = None
        self.current_mask_open_dir: Union[None, str] = None
        self.xcor_tree_leaf_items: List[QtWidgets.QTreeWidgetItem] = []
        self.xcorr_templates: Dict[str, Spectrum] = {}
        self.cur_xcorr_leaf_item: Optional[QtWidgets.QTreeWidgetItem] = None
        self.use_gpu: Union[str, bool] = False
        self.compute_device: Union[str, None] = None
        self.compute_device_id: int = -1
        self.templates_to_plot: Dict[str, Any] = {}

        self.global_state: Enum = GlobalState.READY
        self._updateXcorrTemplateTreeState_lock = False
        self._rebuild_xcorr_template_list = True
        self._copy_mask: Union[Dict[str, Any], None] = None

        self.qf_color: Dict[int, str] = {
            0: "#FFFFFF",
            1: "#55940a00",
            2: "#55945400",
            3: "#55176601",
            4: "#55176601",
            9: "#559100b5"
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

        self.settings_wnd.gpu_backend_combo_box.currentIndexChanged.connect(
            self.setGpuBackend
        )

        self.settings_wnd.gpu_device_combo_box.currentIndexChanged.connect(
            self.setGpuDevice
        )

        self.settings_wnd.accepted.connect(
            self.acceptSettings
        )
        self.settings_wnd.rejected.connect(
            self.restoreSettings
        )

        self.settings_wnd.template_dirs_list_widget.addItem(
            backends.xcorr.TEMPLATE_DIR
        )

        self.settings_wnd.add_template_dir_push_button.clicked.connect(
            self.doAddNewTemplateDir
        )

        self.settings_wnd.remove_template_dir_push_button.clicked.connect(
            self.doRemoveCurrentTemplateDir
        )

        self.control_mapping_wnd = cast(
            ControlMappingDialog,
            qt_api.loadUiWidget("control_mapping_dialog.ui")
        )

        self.main_wnd.setWindowTitle("redmost")
        self.main_wnd.setWindowIcon(
            qt_api.QtGui.QIcon(utils.get_icon("redmost.png"))
        )
        self.qapp.setWindowIcon(
            qt_api.QtGui.QIcon(utils.get_icon("redmost.png"))
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

        # Fill colormap combobox
        if HAS_MPL:

            icon_w = 50
            icon_h = 10
            _icon_gray_data = np.repeat(
                np.linspace(start=0, stop=1, num=icon_w)[None,...],
                repeats=icon_h,
                axis=0
            )
            self.main_wnd.cmap_combo_box.setEnabled(True)
            self.main_wnd.cmap_combo_box.setToolTip(
                "Set the current colormap for 2D/3D spectra"
            )
            item: QtWidgets.QListWidgetItem
            for cmap_name in list(colormaps):
                cmap = colormaps.get(cmap_name)
                _rgb_icon_data: np.ndarray = (
                        255 * cmap(_icon_gray_data)[..., 0:3]
                ).astype(np.uint8).copy()
                icon: QtGui.QIcon = qt_api.QtGui.QIcon(
                    qt_api.QtGui.QPixmap(
                        qt_api.QtGui.QImage(
                            _rgb_icon_data.data,
                            icon_w, icon_h, 3 * icon_w,
                            qt_api.QtGui.QImage.Format.Format_RGB888
                        )
                    )
                )

                # NOTE: remember, QImage does not hold the data, it just reads
                #       the memory location pointed at by the input. If no
                #       reference is made for the input array it will be
                #       collected by the garbage collector eventually, and the
                #       QImage will try to read from an invalid memory segment.
                #       To prevent this,we just make sure to keep a reference
                #       of the input array
                icon._data = _rgb_icon_data

                self.main_wnd.cmap_combo_box.addItem(
                    icon, cmap_name, cmap_name
                )
            self.main_wnd.cmap_combo_box.setIconSize(
                qt_api.QtCore.QSize(icon_w, icon_h)
            )
        else:
            self.main_wnd.cmap_combo_box.setEnabled(False)
            self.main_wnd.cmap_combo_box.setToolTip(
                "Please, install the matplotlib package to enable colormaps"
            )

        self.splash.showMessage(self.qapp.tr("Loading backends..."))
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

        # Check for X-Corr plugin.
        if not backends.HAS_XCORR:
            self.main_wnd.tab_xcorr.setEnabled(False)
            self.xcorr_handler = None
        else:
            self.main_wnd.tab_xcorr.setEnabled(True)
            self.main_wnd.xcorr_wrange_checkbox.setEnabled(True)
            self.main_wnd.xcorr_wrange_checkbox.setChecked(False)
            if utils.HAS_TORCH or utils.HAS_TENSORFLOW:
                self.main_wnd.xcorr_gpu_checkbox.setEnabled(True)
            else:
                self.main_wnd.xcorr_gpu_checkbox.setEnabled(False)

            self.main_wnd.xcorr_run_button.clicked.connect(
                self.doStartXCorr
            )

            self.main_wnd.xcorr_csub_preview_checkbox.clicked.connect(
                self.redrawCurrentSpec
            )

            self.main_wnd.xcorr_csub_combobox.currentIndexChanged.connect(
                self.redrawCurrentSpec
            )

            self.main_wnd.xcorr_csub_chebyshev_order_spinbox.valueChanged.connect(
                self.redrawCurrentSpec
            )

            self.main_wnd.xcorr_csub_fft_sigma_dspinbox.valueChanged.connect(
                self.redrawCurrentSpec
            )

            self.xcorr_handler = QXCorrHandler()
            self.xcorr_handler.reset.connect(
                self.pbar.reset
            )
            self.xcorr_handler.maximum.connect(
                self.pbar.setMaximum
            )
            self.xcorr_handler.progress.connect(
                self.pbar.setValue
            )
            self.xcorr_handler.message.connect(
                self.statusbar.showMessage
            )
            self.xcorr_handler.finished.connect(
                self.collectXCorrResults
            )

        self.setXcorrTreeWidgetColumnWidth([200, 20])

        # Dialog for importing zcat

        self.zcat_mapping_dialog = ImportZcatDialog()
        self.zcat_mapping_dialog.finished.connect(self.getZcatColumnMapping)

        # QChartView widget for flux

        self.flux_chart_view: SpectrumQChartView = SpectrumQChartView(
            self.main_wnd.flux_tabwidget
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
        self.main_wnd.flux_widget_layout_1d.addWidget(self.flux_chart_view)

        self.flux_chart_view_2d: SpectrumQChartView2D = SpectrumQChartView2D(
            self.main_wnd.flux_tabwidget
        )
        self.flux_chart_view_2d.master = False
        self.flux_chart_view_2d.setObjectName("flux_chart_view_2d")
        self.flux_chart_view_2d.setRenderHint(
            qt_api.QtGui.QPainter.RenderHint.Antialiasing
        )
        self.flux_chart_view_2d.setHorizontalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.flux_chart_view_2d.setVerticalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.flux_chart_view_2d.setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view_2d.chart().setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view_2d.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.flux_chart_view_2d.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

        self.main_wnd.flux_widget_layout_2d.addWidget(self.flux_chart_view_2d)

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

        # QChartView widget for XCorr

        self.xcorr_chart_view: XCorrQChartView = XCorrQChartView(
            self.main_wnd.other_charts_tab_widget
        )
        self.xcorr_chart_view.vertical_lock = True
        self.xcorr_chart_view.setObjectName("var_chart_view")
        self.xcorr_chart_view.setRenderHint(
            qt_api.QtGui.QPainter.RenderHint.Antialiasing
        )
        self.xcorr_chart_view.setHorizontalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.xcorr_chart_view.setVerticalScrollBarPolicy(
            qt_api.QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.xcorr_chart_view.setContentsMargins(0, 0, 0, 0)
        self.xcorr_chart_view.chart().setContentsMargins(0, 0, 0, 0)
        self.xcorr_chart_view.chart().layout().setContentsMargins(0, 0, 0, 0)
        self.xcorr_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.main_wnd.xcorr_widget_layout.addWidget(self.xcorr_chart_view)

        # Add siblings to main flux QChartView

        self.flux_chart_view.addSibling(self.var_chart_view)
        self.flux_chart_view.addSibling(self.wdisp_chart_view)
        self.flux_chart_view.addSibling(self.sky_chart_view)
        self.flux_chart_view.addSibling(self.flux_chart_view_2d)

        # Set QSplitters initial sizes

        self.main_wnd.splitter_main.setSizes([200, 1000, 200])
        self.main_wnd.splitter_plots.setSizes([500, 300])

        # Fill mask pattert combo-box
        self.main_wnd.mask_fill_combobox.clear()
        for x in list(qt_api.QtCore.Qt.BrushStyle):
            if x.value >= 15:
                break
            self.main_wnd.mask_fill_combobox.addItem(
                x.name, x.value
            )

        # Connect signals
        self.main_wnd.cmap_combo_box.currentIndexChanged.connect(
            self.doUpdateColorMap
        )
        self.main_wnd.cmap_min_dspinbox.valueChanged.connect(
            self.doUpdateColorMap
        )
        self.main_wnd.cmap_max_dspinbox.valueChanged.connect(
            self.doUpdateColorMap
        )

        self.main_wnd.mask_add_button.clicked.connect(
            self.doAddMask
        )
        self.main_wnd.mask_name_textbox.textChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_desc_textbox.textChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_start_spinbox.valueChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_end_spinbox.valueChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_enabled_checkbox.checkStateChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_apply_all_checkbox.checkStateChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_color_textbox.textChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_fill_combobox.currentIndexChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_trasparency_spinbox.valueChanged.connect(
            self.maskPropChanged
        )
        self.main_wnd.mask_load_button.clicked.connect(
            self.doLoadMasks
        )
        self.main_wnd.mask_save_button.clicked.connect(
            self.doSaveMasks
        )

        self.main_wnd.mask_delete_button.clicked.connect(
            self.doDeleteMask
        )

        self.main_wnd.mask_deleteall_button.clicked.connect(
            self.doDeleteAllMask
        )

        self.main_wnd.mask_list_widget.currentItemChanged.connect(
            self.currentMaskChanged
        )

        self.main_wnd.mask_list_widget.itemChanged.connect(
            self.maskChanged
        )

        self.main_wnd.mask_apply_button.clicked.connect(
            self.updateCurrentMask
        )

        self.main_wnd.mask_cancel_button.clicked.connect(
            self.restoreCurrentMask
        )

        self.main_wnd.action_controls_layout.triggered.connect(
            self.control_mapping_wnd.open
        )

        self.main_wnd.log_y_check_box.toggled.connect(
            self.redrawCurrentSpec
        )

        self.main_wnd.plot_templates_check_box.toggled.connect(
            self.doPlotTemplates
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
        self.flux_chart_view.maskSelectedIndexChanged.connect(
            self.main_wnd.mask_list_widget.setCurrentRow
        )

        self.flux_chart_view_2d.maskSelectedIndexChanged.connect(
            self.main_wnd.mask_list_widget.setCurrentRow
        )

        self.var_chart_view.onMouseMoveSeries.connect(
            self._updateMouseLabelFromEvent
        )
        self.var_chart_view.maskSelectedIndexChanged.connect(
            self.main_wnd.mask_list_widget.setCurrentRow
        )

        self.xcorr_chart_view.onMouseMoveSeries.connect(
            self._updateMouseLabelXCorrFromEvent
        )

        self.main_wnd.smoothing_1d_check_box.stateChanged.connect(
            self.redrawCurrentSpec
        )
        self.main_wnd.smoothing_1d_dspinbox.valueChanged.connect(
            self.redrawCurrentSpec
        )

        self.main_wnd.smoothing_2d_check_box.stateChanged.connect(
            self.redrawCurrentSpec
        )
        self.main_wnd.smoothing_2d_dspinbox.valueChanged.connect(
            self.redrawCurrentSpec
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

        self.main_wnd.show_legend_check_box.toggled.connect(
            self.redrawCurrentSpec
        )

        self.main_wnd.show_lines_check_box.toggled.connect(
            self.flux_chart_view_2d.setLinesVisible
        )

        self.main_wnd.z_dspinbox.valueChanged.connect(
            self.setCurrentObjectRedshift
        )

        self.main_wnd.qflag_spinbox.valueChanged.connect(
            self.currentQualityFlagChanged
        )

        self.main_wnd.show_lines_combo_box.currentIndexChanged.connect(
            self.setShowLinesType
        )

        self.main_wnd.action_online_user_manual.triggered.connect(
            _open_online_doc
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

        self.main_wnd.xcorr_preview_button.clicked.connect(
            self.doPreviewXcorrTemplate
        )

        self.main_wnd.xcorr_templates_tree_widget.itemChanged.connect(
            self.updateXcorrTemplateTreeState
        )

        self.main_wnd.xcorr_templates_tree_widget.itemDoubleClicked.connect(
            self.setCurrentObjectRedshiftFromXcorr
        )

        self.main_wnd.xcorr_templates_tree_widget.itemDoubleClicked.connect(
            self.plotXcorrFunction
        )

        self.main_wnd.xcorr_templates_tree_widget.itemDoubleClicked.connect(
            self.plotXcorrTemplate
        )

        self.main_wnd.xcorr_use_z_button.clicked.connect(
            self.doSetCurrentObjectRedshiftFromXcorr
        )

        self.main_wnd.xcorr_use_z_button.clicked.connect(
            self.doPlotXcorrFunction
        )

        self.main_wnd.xcorr_use_z_button.clicked.connect(
            self.doPlotXcorrTemplate
        )

        self.main_wnd.xcorr_wav_max_spinbox.valueChanged.connect(
            self.redrawCurrentSpec
        )

        self.main_wnd.xcorr_wav_min_spinbox.valueChanged.connect(
            self.redrawCurrentSpec
        )

        self.main_wnd.xcorr_wrange_checkbox.checkStateChanged.connect(
            self.redrawCurrentSpec
        )

        self.main_wnd.action_mask_copy.triggered.connect(
            self.doCopyMask
        )

        self.main_wnd.action_mask_paste.triggered.connect(
            self.doPasteMask
        )

        # Final steps

        # This dictionary contains the global settings that should be saved
        # upon closing, and restored when the application is started again.
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
            "plot_templates": (
              self.main_wnd.plot_templates_check_box.checkState,
              self.main_wnd.plot_templates_check_box.setCheckState,
              qt_api.QtCore.Qt.CheckState
            ),
            "smoothing_1d_state": (
                self.main_wnd.smoothing_1d_check_box.checkState,
                self.main_wnd.smoothing_1d_check_box.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "smoothing_1d_value": (
                self.main_wnd.smoothing_1d_dspinbox.value,
                self.main_wnd.smoothing_1d_dspinbox.setValue,
                float,
            ),
            "smoothing_2d_state": (
                self.main_wnd.smoothing_2d_check_box.checkState,
                self.main_wnd.smoothing_2d_check_box.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "smoothing_2d_value": (
                self.main_wnd.smoothing_2d_dspinbox.value,
                self.main_wnd.smoothing_2d_dspinbox.setValue,
                float,
            ),
            "show_legend": (
                self.main_wnd.show_legend_check_box.checkState,
                self.main_wnd.show_legend_check_box.setCheckState,
                qt_api.QtCore.Qt.CheckState
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
            "xcorr_csub_combobox": (
                self.main_wnd.xcorr_csub_combobox.currentIndex,
                self.main_wnd.xcorr_csub_combobox.setCurrentIndex,
                int
            ),
            "xcorr_csub_chebyshev_order_spinbox": (
                self.main_wnd.xcorr_csub_chebyshev_order_spinbox.value,
                self.main_wnd.xcorr_csub_chebyshev_order_spinbox.setValue,
                int
            ),
            "xcorr_csub_fft_sigma_dspinbox": (
                self.main_wnd.xcorr_csub_fft_sigma_dspinbox.value,
                self.main_wnd.xcorr_csub_fft_sigma_dspinbox.setValue,
                float
            ),
            "xcorr_csub_preview_checkbox": (
                self.main_wnd.xcorr_csub_preview_checkbox.checkState,
                self.main_wnd.xcorr_csub_preview_checkbox.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "xcorr_gpu_checkbox": (
                self.main_wnd.xcorr_gpu_checkbox.checkState,
                self.main_wnd.xcorr_gpu_checkbox.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "xcorr_wrange_checkbox": (
                self.main_wnd.xcorr_wrange_checkbox.checkState,
                self.main_wnd.xcorr_wrange_checkbox.setCheckState,
                qt_api.QtCore.Qt.CheckState
            ),
            "xcorr_wav_min": (
                self.main_wnd.xcorr_wav_min_spinbox.value,
                self.main_wnd.xcorr_wav_min_spinbox.setValue,
                int
            ),
            "xcorr_wav_max": (
                self.main_wnd.xcorr_wav_max_spinbox.value,
                self.main_wnd.xcorr_wav_max_spinbox.setValue,
                int
            ),
            "xcorr_z_min": (
                self.main_wnd.xcorr_z_min_dspinbox.value,
                self.main_wnd.xcorr_z_min_dspinbox.setValue,
                float
            ),
            "xcorr_z_max": (
                self.main_wnd.xcorr_z_max_dspinbox.value,
                self.main_wnd.xcorr_z_max_dspinbox.setValue,
                float
            ),
            "xcorr_z_step": (
                self.main_wnd.xcorr_z_step_dspinbox.value,
                self.main_wnd.xcorr_z_step_dspinbox.setValue,
                float
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
            ),
            "gpu_backend": (
                self.getGpuBackend,
                self.setGpuBackend,
                str
            ),
            "gpu_device": (
                self.getGpuDeviceId,
                self.setGpuDeviceId,
                str
            ),
            "xcorr_template_dirs": (
                self.getXcorrTemplateDirs,
                self.setXcorrTemplateDirs,
                lambda x: x if isinstance(x, list) else [x,]
            ),
            "xcorr_tree_widget_column_widths": (
                self.getXcorrTreeWidgetColumnWidth,
                self.setXcorrTreeWidgetColumnWidth,
                lambda x: x if isinstance(x, list) else [int(x),]
            )
        }

        self.splash.showMessage(
            self.qapp.tr("Loading settings and templates...")
        )
        self.loadSettings(prog_callback=self.splash.setValue)
        self.newProject()

        self.splash.finish(self.main_wnd)
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
            'comment': str(self.main_wnd.z_comment_textbox.text()),
            'quality_flag': int(self.main_wnd.qflag_spinbox.value()),
            'cmap': self.main_wnd.cmap_combo_box.currentData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole,
            ),
            'vmin': self.main_wnd.cmap_min_dspinbox.value(),
            'vmax': self.main_wnd.cmap_max_dspinbox.value(),
            'lines': {
                'list': lines_list,
                'redshifts': redshifts_form_lines
            }
        }

        self.object_state_dict[self.current_uuid] = obj_state
        self.global_state = GlobalState.READY

    def _lock(self, *args, **kwargs) -> None:
        self.main_wnd.spec_group_box.setEnabled(False)
        self.main_wnd.redshift_tools_tab.setEnabled(False)
        self.main_wnd.masking_tab.setEnabled(False)
        self.main_wnd.info_tab_widget.setEnabled(False)
        self.flux_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.NoRubberBand
        )
        self.flux_chart_view_2d.setRubberBand(
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
        self.main_wnd.action_mask_copy.setIcon(
            qt_api.get_qicon("copy", theme)
        )
        self.main_wnd.action_mask_copy.setIcon(
            qt_api.get_qicon("copy", theme)
        )
        self.main_wnd.action_mask_paste.setIcon(
            qt_api.get_qicon("clipboard", theme)
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
        self.main_wnd.xcorr_run_button.setIcon(
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

        try:
            z_comment = old_state['comment']
        except KeyError:
            z_comment = ""

        if z_comment is not None:
            self.main_wnd.z_comment_textbox.setText(z_comment)

        try:
            vmin = old_state['vmin']
        except KeyError:
            vmin = 0
        if vmin is not None:
            self.main_wnd.cmap_min_dspinbox.setValue(vmin)

        try:
            vmax = old_state['vmax']
        except KeyError:
            vmax = 1
        if vmax is not None:
            self.main_wnd.cmap_max_dspinbox.setValue(vmax)

        try:
            cmap_name = old_state['cmap']
        except KeyError:
            cmap_name = self.DEFAULT_SPEC_INFO['cmap']
        if cmap_name is not None:
            self.main_wnd.cmap_combo_box.setCurrentText(cmap_name)

        self.main_wnd.qflag_spinbox.setValue(int(quality_flag))

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
        self.main_wnd.redshift_tools_tab.setEnabled(True)
        self.main_wnd.info_tab_widget.setEnabled(True)
        self.main_wnd.plot_group_box.setEnabled(True)
        self.main_wnd.masking_tab.setEnabled(True)
        self.flux_chart_view.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.RectangleRubberBand
        )
        self.flux_chart_view_2d.setRubberBand(
            qt_api.QtCharts.QChartView.RubberBand.RectangleRubberBand
        )

    def _update_spec_item_qf(self, item_uuid: uuid.UUID, qf: int) -> None:
        item: QtWidgets.QListWidgetItem
        item = self.open_spectra_items[item_uuid]
        try:
            item.setBackground(qt_api.QtGui.QColor(self.qf_color[qf]))
        except KeyError:
            item.setBackground(qt_api.QtGui.QColor("#00000000"))

    def _updateMouseLabelFromEvent(self, *args) -> None:
        self._updateMouseLabel(args[0][0])

    def _updateMouseLabelXCorrFromEvent(self, *args) -> None:
        self._updateMouseLabelXCorr(args[0][0])

    def _updateMouseLabel(self, mouse_pos: QtCore.QPointF) -> None:
        self.mousePosLabel.setText(f"\u03BB = {mouse_pos.x():.2f}")

    def _updateMouseLabelXCorr(self, mouse_pos: QtCore.QPointF) -> None:
        self.mousePosLabel.setText(f"z = {mouse_pos.x():.4f}")

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

    def addNewTemplateDir(self, dir_path, update=True) -> None:
        dir_path = pathlib.Path(str(dir_path)).resolve()

        for j in range(self.settings_wnd.template_dirs_list_widget.count()):
            item = self.settings_wnd.template_dirs_list_widget.item(j)
            existing_path = pathlib.Path(item.text()).resolve()
            if dir_path == existing_path:
                break
        else:
            self._rebuild_xcorr_template_list = True
            self.settings_wnd.template_dirs_list_widget.addItem(
                str(dir_path)
            )
            if update:
                self.updateXcorrTemplateList()

    def clearXcorrTreeWidgetZ(self) -> None:
        for item in self.xcor_tree_leaf_items:
            item.setText(1, "")

        self.cur_xcorr_leaf_item = None
        self.plotXcorrFunction()

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
                info_dict = copy.deepcopy(self.DEFAULT_SPEC_INFO)
                self.object_state_dict[obj_uuid] = info_dict

            info_dict['redshift'] = row['z']
            if row['zwarn'] != 0:
                self._update_spec_item_qf(obj_uuid, 1)
        self._restore_object_state(self.current_uuid)
        self.global_state = GlobalState.READY
        self._unlock()

    def collectXCorrResults(self) -> None:
        if self.xcorr_handler is None:
            return

        self.showXcorrZ()

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
            self.main_wnd.xcorr_current_radio.setEnabled(False)
            self.updateMaskList()
            return
        else:
            self.main_wnd.redrock_current_radio.setEnabled(True)
            self.main_wnd.xcorr_current_radio.setEnabled(True)
            self.main_wnd.mask_add_button.setEnabled(True)
            self.main_wnd.mask_load_button.setEnabled(True)

        if self.global_state != GlobalState.READY:
            return

        self._unlock()

        spec_uuid: uuid.UUID = new_item.data(
            qt_api.QtCore.Qt.ItemDataRole.UserRole
        )

        self.showXcorrZ(spec_uuid)
        self.plotXcorrFunction(self.cur_xcorr_leaf_item)

        self.showObjectInfo(spec_uuid)

        flux_chart: QtCharts.QChart = self.flux_chart_view.chart()
        flux_chart.removeAllSeries()

        flux_chart_2d: QtCharts.QChart = self.flux_chart_view_2d.chart()
        flux_chart_2d.removeAllSeries()

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
            self.updateMaskList()

            for ax in flux_chart.axes():
                flux_chart.removeAxis(ax)

        try:
            mask_list = list(self.spectral_masks[self.current_uuid].values())
        except KeyError:
            mask_list = []

        self.flux_chart_view.setMasks(mask_list)
        self.flux_chart_view_2d.setMasks(mask_list)
        self.var_chart_view.setMasks(mask_list)

        for ax in var_chart.axes():
            var_chart.removeAxis(ax)
        for ax in wd_chart.axes():
            wd_chart.removeAxis(ax)
        for ax in sky_chart.axes():
            sky_chart.removeAxis(ax)

        sp: Spectrum = self.open_spectra[spec_uuid]

        try:
            sp_2d: loaders.Spectrum2D = sp.meta['linked']['2D']
        except KeyError:
            pass
        else:
            self.flux_chart_view_2d.setSpectralAxis(
                sp_2d.spectral_axis.to(units.angstrom).value
            )

            if self.main_wnd.smoothing_2d_check_box.isChecked():
                sigma_2d = self.main_wnd.smoothing_2d_dspinbox.value()
                if sigma_2d > 0:
                    flux_2d_image = gaussian_filter(
                        sp_2d.flux.value, sigma=sigma_2d
                    )
                else:
                    flux_2d_image = sp_2d.flux.value
            else:
                flux_2d_image = sp_2d.flux.value

            self.flux_chart_view_2d.setSpectrumImage(flux_2d_image)

            self.flux_chart_view_2d.start_1d = sp_2d.meta['header'].get(
                '1D_START'
            )
            self.flux_chart_view_2d.end_1d = sp_2d.meta['header'].get(
                '1D_END'
            )

        wav: np.ndarray = sp.spectral_axis.value
        wav_unit: units.Unit = sp.spectral_axis.unit

        flux: np.ndarray = sp.flux.value

        # NOTE: We need this because QChart refuses to handle correctly ticks
        #       placement when the axis range is very small
        mean_flux = 10 ** utils.find_exp(np.nanmean(flux))
        flux = flux / mean_flux
        flux_unit: units.Unit = units.Unit(sp.flux.unit * mean_flux)

        var: Union[np.ndarray, None] = None
        var_unit: Union[units.Unit, None] = None

        wdisp: Union[np.ndarray, None] = None
        wdisp_unit: Union[units.Unit, None] = None

        sky: Union[np.ndarray, None] = None
        sky_unit: Union[units.Unit, None] = None

        if isinstance(sp.uncertainty, VarianceUncertainty):
            var = sp.uncertainty.array / (mean_flux ** 2)
            var_unit = units.Unit(sp.uncertainty.unit * (mean_flux ** 2))
        elif isinstance(sp.uncertainty, InverseVariance):
            var = 1 / (sp.uncertainty.array * (mean_flux ** 2))
            var_unit = units.Unit(1 / (sp.uncertainty.unit / (mean_flux ** 2)))
        elif isinstance(sp.uncertainty, StdDevUncertainty):
            var = (sp.uncertainty.array / mean_flux) ** 2
            var_unit = units.Unit((sp.uncertainty.unit * mean_flux) ** 2)

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


        flux_2d_series = values2series(
            wav, flux, self.qapp.tr("Flux")
        )
        flux_chart_2d.addSeries(flux_2d_series)

        if not flux_chart_2d.axes():
            flux_2d_axis_x = qt_api.QtCharts.QValueAxis()
            flux_2d_axis_y = qt_api.QtCharts.QValueAxis()

            flux_chart_2d.addAxis(
                flux_2d_axis_x, qt_api.QtCore.Qt.AlignmentFlag.AlignBottom
            )
            flux_chart_2d.addAxis(
                flux_2d_axis_y, qt_api.QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            flux_2d_axis_x = flux_chart_2d.axes()[0]
            flux_2d_axis_y = flux_chart_2d.axes()[1]

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
        flux_axis_x.setTitleText(wav_unit.to_string())

        flux_axis_y.setLabelFormat("%.2f")
        flux_axis_y.setTitleText(flux_unit.to_string())

        flux_series.attachAxis(flux_axis_x)
        flux_series.attachAxis(flux_axis_y)

        flux_2d_axis_x.setTickInterval(500)
        flux_2d_axis_x.setLabelFormat("%.2f")
        flux_2d_axis_x.setTitleText(wav_unit.to_string())

        flux_2d_axis_y.setLabelFormat("%.2f")
        flux_2d_axis_y.setTitleText('pixels')

        flux_2d_series.attachAxis(flux_2d_axis_x)
        flux_2d_series.attachAxis(flux_2d_axis_y)
        flux_2d_series.setOpacity(0.2)

        if self.main_wnd.smoothing_1d_check_box.isChecked():
            smoothing_factor = self.main_wnd.smoothing_1d_dspinbox.value()

            flux_series.setOpacity(0.2)
            smoothing_sigma = len(flux) / (1 + 2 * smoothing_factor)
            smoothed_flux = utils.smooth_fft(
                flux,
                sigma=smoothing_sigma,
                use_gpu=self.use_gpu,
                device=self.compute_device
            )
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

        if self.main_wnd.show_legend_check_box.isChecked():
            flux_chart.legend().show()
        else:
            flux_chart.legend().hide()

        flux_chart_2d.setContentsMargins(0, 0, 0, 0)
        flux_chart_2d.setBackgroundRoundness(0)
        flux_chart_2d.legend().hide()

        var_chart.setContentsMargins(0, 0, 0, 0)
        var_chart.setBackgroundRoundness(0)
        var_chart.legend().hide()

        wd_chart.setContentsMargins(0, 0, 0, 0)
        wd_chart.setBackgroundRoundness(0)
        wd_chart.legend().hide()

        sky_chart.setContentsMargins(0, 0, 0, 0)
        sky_chart.setBackgroundRoundness(0)
        sky_chart.legend().hide()

        self.plotXcorrCsub(flux_scale=mean_flux)
        self.plotTemplates()

    def doAddNewLine(self, *args: Any, **kwargs: Any) -> None:
        """Tell the main app to identify a new line."""
        # Do nothing if no spectrum is currently selected
        if self.current_uuid is None:
            return
        self._lock()
        self.main_wnd.spec_group_box.setEnabled(False)
        self.global_state = GlobalState.SELECT_LINE_MANUAL
        self.qapp.setOverrideCursor(qt_api.QtCore.Qt.CursorShape.CrossCursor)

    def doAddNewTemplateDir(self, *args: Any, **kwargs: Any) -> None:

        url = qt_api.QtWidgets.QFileDialog.getExistingDirectoryUrl(
            self.settings_wnd,
            self.qapp.tr("Select a directory"),
        )

        if not url.isValid():
            return

        self.addNewTemplateDir(url.path())
        self.updateXcorrTemplateList()

    def addMask(
        self,
        spec_uuid: uuid.UUID,
        mask_properties: Dict[str, Any],
        mask_uuid: Optional[uuid.UUID] = None
    ) -> uuid.UUID:
        if mask_uuid is None:
            mask_uuid = uuid.uuid4()

        if spec_uuid not in self.spectral_masks:
            self.spectral_masks[spec_uuid] = {}

        self.spectral_masks[spec_uuid][mask_uuid] = mask_properties

        return mask_uuid

    def deleteMask(self, spec_uuid: uuid.UUID, mask_uuid: uuid.UUID) -> None:
        try:
            mask_properties = self.spectral_masks[spec_uuid][mask_uuid]
        except KeyError:
            pass
        else:
            if not mask_properties['global']:
                self.spectral_masks[spec_uuid].pop(mask_uuid)
        self.updateMaskList()

    def currentMaskItem(self) -> Union[
        Tuple[uuid.UUID, QtWidgets.QListWidgetItem],
        None
    ]:
        if self.current_uuid is None:
            return None

        mask_item = self.main_wnd.mask_list_widget.currentItem()

        if mask_item is None:
            return None

        mask_uuid = mask_item.data(
            qt_api.QtCore.Qt.ItemDataRole.UserRole
        )

        return mask_uuid, mask_item

    def doAddMask(self) -> None:
        if self.current_uuid is None:
            return

        mask_properties = copy.deepcopy(self.DEFAULT_MASK_PROPERTIES)


        custom_widget: MaskListWidgetItem = MaskListWidgetItem(
            self.main_wnd.mask_list_widget
        )
        custom_widget.setName(mask_properties['name'])

        new_item = qt_api.QtWidgets.QListWidgetItem()
        new_item.setSizeHint(custom_widget.sizeHint())
        new_item.setCheckState(qt_api.QtCore.Qt.CheckState.Checked)
        new_item.setData(
            qt_api.QtCore.Qt.ItemDataRole.UserRole,
            self.addMask(self.current_uuid, mask_properties)
        )

        self.main_wnd.mask_list_widget.addItem(new_item)
        self.main_wnd.mask_list_widget.setItemWidget(new_item, custom_widget)
        self.main_wnd.mask_list_widget.setCurrentItem(new_item)
        self.currentMaskChanged(new_item)
        # self.redrawCurrentSpec()

    def doCopyMask(self) -> None:
        mask_uuid, _ = self.currentMaskItem()

        try:
            self._copy_mask = self.spectral_masks[self.current_uuid][mask_uuid]
        except KeyError:
            self._copy_mask = None

    def doPasteMask(self) -> None:
        if (self._copy_mask is None) or (self.current_uuid is None):
            return

        if self.current_uuid not in self.spectral_masks:
            self.spectral_masks[self.current_uuid] = {}

        mask_uuid = uuid.uuid4()
        self.spectral_masks[self.current_uuid][mask_uuid] = copy.deepcopy(
            self._copy_mask
        )
        self.updateMaskList()
        self.redrawCurrentSpec()

    def doDeleteMask(self) -> None:
        current_mask = self.currentMaskItem()
        if current_mask is None:
            return

        if current_mask[0] is not None:
            self.deleteMask(self.current_uuid, current_mask[0])
        self.redrawCurrentSpec()

    def doDeleteAllMask(self) -> None:
        if self.current_uuid is None:
            return

        try:
            mask_dict = self.spectral_masks[self.current_uuid]
        except KeyError:
            return

        for mask_uuid in list(mask_dict.keys()):
            self.deleteMask(self.current_uuid, mask_uuid)

        self.redrawCurrentSpec()

    def doLoadMasks(self) -> bool:
        if self.current_uuid is None:
            return False

        if self.current_mask_open_dir is not None:
            cur_dir = self.current_mask_open_dir
        elif self.current_project_file_path is not None:
            cur_dir = self.current_project_file_path
        elif self.current_open_dir is not None:
            cur_dir = self.current_open_dir
        else:
            cur_dir = '.'

        mask_file_path, _ = qt_api.QtWidgets.QFileDialog.getOpenFileName(
            self.main_wnd,
            self.qapp.tr("Load masks from file"),
            cur_dir,
            (
                f"{self.qapp.tr('JSON file')} (*.json);;"
                f"{self.qapp.tr('All files')} (*.*)"
            ),
            f"{self.qapp.tr('JSON file')} (*.json)"
        )

        if not mask_file_path:
            return False

        try:
            serialized_masks = []

            with open(mask_file_path, 'r') as f:
                serialized_masks = json.load(f)

            for mask_properties in serialized_masks:
                for key, val in self.DEFAULT_MASK_PROPERTIES.items():
                    if key not in mask_properties:
                        mask_properties[key] = copy.copy(val)

                mask_properties['linked'] = None
                mask_properties['global'] = False
                mask_properties['enabled'] = True

        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                 self.qapp.tr(
                     "An error has occurred while reading masks."
                 )
            )
            self.msgBox.setInformativeText(
                traceback.format_exc()
            )
            self.msgBox.setDetailedText(str(exc))
            self.msgBox.setIcon(
                qt_api.QtWidgets.QMessageBox.Icon.Critical
            )
            self.msgBox.setStandardButtons(
                qt_api.QtWidgets.QMessageBox.StandardButton.Ok
            )
            self.msgBox.exec()
            return False

        for mask_properties in serialized_masks:
            self.addMask(self.current_uuid, mask_properties)
        self.updateMaskList()
        self.redrawCurrentSpec()

        self.statusbar.showMessage(self.qapp.tr("Mask file loaded"))
        self.current_mask_open_dir = os.path.dirname(mask_file_path)
        return True

    def doSaveMasks(self) -> bool:
        if self.current_uuid is None:
            return False

        if self.current_mask_open_dir is not None:
            cur_dir = self.current_mask_open_dir
        elif self.current_project_file_path is not None:
            cur_dir = self.current_project_file_path
        elif self.current_open_dir is not None:
            cur_dir = self.current_open_dir
        else:
            cur_dir = '.'

        open_list = qt_api.QtWidgets.QFileDialog.getSaveFileName(
            self.main_wnd,
            self.qapp.tr("Export masks to file"),
            os.path.join(cur_dir, 'masks.json'),
            (
                f"{self.qapp.tr('JSON file')} (*.json);;"
            ),
            f"{self.qapp.tr('JSON file')} (*.json)"
        )

        try:
            dest_file_path, file_type = open_list
            serialized_masks = []

            for x in self.spectral_masks[self.current_uuid].values():
                mask_properties = copy.deepcopy(x)
                for p in ['linked', 'global', 'enabled']:
                    try:
                        mask_properties.pop(p)
                    except KeyError:
                        continue
                serialized_masks.append(mask_properties)

            with open(dest_file_path, 'w') as f:
                json.dump(serialized_masks, f, indent=2)
        except Exception as exc:
            self.msgBox.setWindowTitle(self.qapp.tr("Error"))
            self.msgBox.setText(
                 self.qapp.tr(
                     "An error has occurred while saving masks."
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

        self.statusbar.showMessage(self.qapp.tr("Mask file saved"))
        self.current_mask_open_dir = os.path.dirname(dest_file_path)
        return True

    def updateMaskList(self) -> None:
        self.main_wnd.mask_list_widget.clear()

        if self.current_uuid is None:
            self.main_wnd.mask_add_button.setEnabled(False)
            self.main_wnd.mask_delete_button.setEnabled(False)
            self.main_wnd.mask_deleteall_button.setEnabled(False)
            self.main_wnd.mask_load_button.setEnabled(False)
            self.main_wnd.mask_save_button.setEnabled(False)
            return
        else:
            self.main_wnd.mask_add_button.setEnabled(True)
            self.main_wnd.mask_load_button.setEnabled(True)

        try:
            masks = self.spectral_masks[self.current_uuid]
        except KeyError:
            return
        else:
            if len(masks) > 0:
                self.main_wnd.mask_delete_button.setEnabled(True)
                self.main_wnd.mask_deleteall_button.setEnabled(True)
                self.main_wnd.mask_save_button.setEnabled(True)

        for mask_uuid, mask_properties in masks.items():
            try:
                mask_enabled = mask_properties["enabled"]
            except KeyError:
                mask_enabled = True

            new_item = qt_api.QtWidgets.QListWidgetItem()

            if mask_enabled:
                new_item.setCheckState(qt_api.QtCore.Qt.CheckState.Checked)
            else:
                new_item.setCheckState(qt_api.QtCore.Qt.CheckState.Unchecked)
            new_item.setData(qt_api.QtCore.Qt.ItemDataRole.UserRole, mask_uuid)

            custom_widget: MaskListWidgetItem = MaskListWidgetItem()
            custom_widget.setName(mask_properties["name"])
            custom_widget.setPreviewMask(mask_properties)

            if mask_properties['global']:
                custom_widget.setFlagText('[ALL]')
            else:
                custom_widget.setFlagText('')

            new_item.setSizeHint(custom_widget.sizeHint())

            self.main_wnd.mask_list_widget.addItem(new_item)
            self.main_wnd.mask_list_widget.setItemWidget(
                new_item, custom_widget
            )

    def updateCurrentMask(self) -> None:
        mask = self.currentMaskItem()
        if mask is None:
            return

        mask_uuid = mask[0]
        mask_item = mask[1]

        try:
            mask_properties = self.spectral_masks[self.current_uuid][mask_uuid]
        except KeyError:
            mask_properties = {}
            self.spectral_masks[self.current_uuid][mask_uuid] = mask_properties

        mask_properties['name'] = self.main_wnd.mask_name_textbox.text()
        mask_properties['description'] = self.main_wnd.mask_desc_textbox.text()
        mask_properties['start'] = self.main_wnd.mask_start_spinbox.value()
        mask_properties['end'] = self.main_wnd.mask_end_spinbox.value()
        mask_properties['color'] = self.main_wnd.mask_color_textbox.text()
        mask_properties['alpha'] = 1 - (self.main_wnd.mask_trasparency_spinbox.value()/100)
        mask_properties['pattern'] = self.main_wnd.mask_fill_combobox.currentText()
        mask_properties['enabled'] = self.main_wnd.mask_enabled_checkbox.isChecked()
        mask_properties['global'] = self.main_wnd.mask_apply_all_checkbox.isChecked()

        if mask_properties['enabled']:
            mask_item.setCheckState(qt_api.QtCore.Qt.CheckState.Checked)
        else:
            mask_item.setCheckState(qt_api.QtCore.Qt.CheckState.Unchecked)

        item_widget: MaskListWidgetItem = self.main_wnd.mask_list_widget.itemWidget(mask_item)
        item_widget.setName(mask_properties['name'])
        item_widget.setPreviewMask(mask_properties)

        if mask_properties['global']:
            item_widget.setFlagText('[ALL]')
            mask_properties['linked'] = self.current_uuid
            for spec_uuid in self.open_spectra.keys():
                if spec_uuid == self.current_uuid:
                    continue
                self.addMask(spec_uuid, mask_properties, mask_uuid)
        else:
            item_widget.setFlagText('')
            if mask_properties['linked']:
                for spec_uuid in self.open_spectra.keys():
                    if spec_uuid == mask_properties['linked']:
                        continue
                    self.deleteMask(spec_uuid, mask_uuid)
                mask_properties['linked'] = None

        self.main_wnd.mask_delete_button.setEnabled(
            not mask_properties['global']
        )
        self.main_wnd.mask_deleteall_button.setEnabled(
            not mask_properties['global']
        )
        self.main_wnd.mask_apply_button.setEnabled(False)
        self.main_wnd.mask_cancel_button.setEnabled(False)
        self.redrawCurrentSpec()

    def restoreCurrentMask(self) -> None:
        if self.current_uuid is None:
            return

        mask_item = self.main_wnd.mask_list_widget.currentItem()
        if mask_item is None:
            return

        self.currentMaskChanged(mask_item)

    def maskChanged(
        self,
        mask_item: Optional[QtWidgets.QListWidgetItem],
        *args: Any,
        **kwargs: Any
    ) -> None:
        if (self.current_uuid is None) or (mask_item is None):
            return

        mask_uuid = mask_item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)

        try:
            mask_properties = self.spectral_masks[self.current_uuid][mask_uuid]
        except KeyError:
            mask_properties = {}
            self.spectral_masks[self.current_uuid][mask_uuid] = mask_properties

        for key, val in self.DEFAULT_MASK_PROPERTIES.items():
            if key not in mask_properties:
                mask_properties[key] = copy.copy(val)

        mask_properties['enabled'] = mask_item.checkState() == qt_api.QtCore.Qt.CheckState.Checked
        self.redrawCurrentSpec()

    def currentMaskChanged(
        self,
        new_item: Optional[QtWidgets.QListWidgetItem],
        *args: Any,
        **kwargs: Any
    ) -> None:

        if (self.current_uuid is None) or (new_item is None):
            self.main_wnd.mask_properties_groupbox.setEnabled(False)
            self.main_wnd.mask_delete_button.setEnabled(False)
            return

        for mask in self.spectral_masks[self.current_uuid].values():
            mask["highlighted"] = False
            mask["selected"] = False

        mask_uuid = new_item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)
        mask_properties = self.spectral_masks[self.current_uuid][mask_uuid]
        mask_properties["selected"] = True

        for key, val in self.DEFAULT_MASK_PROPERTIES.items():
            if key not in mask_properties:
                mask_properties[key] = copy.copy(val)

        self.main_wnd.mask_name_textbox.setText(mask_properties['name'])
        self.main_wnd.mask_desc_textbox.setText(mask_properties['description'])
        self.main_wnd.mask_start_spinbox.setValue(mask_properties['start'])
        self.main_wnd.mask_end_spinbox.setValue(mask_properties['end'])
        self.main_wnd.mask_color_textbox.setText(mask_properties['color'])
        self.main_wnd.mask_trasparency_spinbox.setValue(
            int(100 * (1 - mask_properties['alpha']))
        )
        self.main_wnd.mask_fill_combobox.setCurrentText(
            mask_properties['pattern']
        )
        self.main_wnd.mask_enabled_checkbox.setChecked(
            mask_properties['enabled']
        )
        self.main_wnd.mask_apply_all_checkbox.setChecked(
            mask_properties['global']
        )

        self.main_wnd.mask_properties_groupbox.setEnabled(True)
        self.main_wnd.mask_apply_button.setEnabled(False)
        self.main_wnd.mask_cancel_button.setEnabled(False)

        self.main_wnd.mask_delete_button.setEnabled(
            not mask_properties['global']
        )
        self.main_wnd.mask_deleteall_button.setEnabled(
            not mask_properties['global']
        )
        self.redrawCurrentSpec()

    def maskPropChanged(self) -> None:
        self.main_wnd.mask_apply_button.setEnabled(True)
        self.main_wnd.mask_cancel_button.setEnabled(True)

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
        elif self.current_image_open_dir is not None:
            cur_dir = self.current_image_open_dir
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
            self.current_image_open_dir = os.path.dirname(dest_file_path)
            self.statusbar.showMessage(
                self.qapp.tr("Picture saved")
            )

    def doExportZcat(self, *args, **kwargs) -> None:
        if self.current_zcat_open_dir is not None:
            cur_dir = self.current_zcat_open_dir
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
            self.current_zcat_open_dir = os.path.dirname(dest_file_path)
            self.statusbar.showMessage(
                self.qapp.tr("Redshift catalogue saved")
            )

    def doIdentifyLines(self,  *args: Any, **kwargs: Any) -> None:
        """Automagically identify lines in the current spectrum."""
        if self.current_uuid is None:
            return

        sp: Spectrum = self.open_spectra[self.current_uuid]

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
            cur_dir = self.current_zcat_open_dir
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

        self.current_zcat_open_dir = os.path.dirname(catalogue_file)
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
        exception_tracker: List[Tuple[str, str]] = []

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

            if not os.path.isfile(file):
                exception_tracker.append((
                    file, "not found!"
                ))
                continue

            item_uuid = uuid.uuid4()

            self.pbar.setValue(j + 1)
            self.statusbar.showMessage(
                self.qapp.tr("Loading file") + f"{j + 1:d}/{n_files}..."
            )
            self.qapp.processEvents()

            # Check for a possible collision, even if it should never happen
            while item_uuid in self.open_spectra.keys():
                item_uuid = uuid.uuid4()

            try:
                sp: Union[Spectrum, loaders.Spectrum2D] = loaders.read(file)
            except Exception as exc:
                exception_tracker.append((file, str(exc)))
                continue
            else:
                if not al_least_one:
                    self.current_open_dir = os.path.dirname(file)
                    al_least_one = True

            sp.meta['uuid'] = item_uuid

            try:
                sp_header = sp.meta['header']
            except KeyError:
                sp_header = None

            new_item: Union[QtWidgets.QListWidgetItem, None] = None
            sp_1d: Union[Spectrum, None] = None

            if isinstance(sp, Spectrum):
                item_name = '[1D] ' + os.path.basename(file)
                sp_1d = sp
                if sp_header and 'SPEC_2D' in sp_header:
                    for _uuid, _sp in self.open_spectra.items():
                        if not _sp.meta.get('is_dummy'):
                            continue

                        try:
                            sp_2d = _sp.meta['linked']['2D']
                        except KeyError:
                            continue

                        if sp_2d.meta['file_name'] == sp_header['SPEC_2D']:
                            sp_2d.meta['linked'] = {'1D': sp_1d}
                            sp_1d.meta['linked'] = {'2D': sp_2d}
                            new_item = self.open_spectra_items[_uuid]
                            new_item.setText(
                                '[1D|2D] ' + os.path.basename(file)
                            )
                            self.open_spectra[_uuid] = sp_1d
                            break

            elif isinstance(sp,loaders.Spectrum2D):
                sp_2d = sp
                item_name = '[2D] ' + os.path.basename(file)
                if sp_header and 'SPEC_1D' in sp_header:
                    for _uuid, _sp in self.open_spectra.items():
                        if _sp.meta['file_name'] == sp_header['SPEC_1D']:
                            sp_1d = _sp
                            sp_2d.meta['linked'] = {'1D': sp_1d}
                            sp_1d.meta['linked'] = {'2D': sp_2d}
                            new_item = self.open_spectra_items[_uuid]
                            new_item.setText(
                                '[1D|2D] ' + os.path.basename(file)
                            )
                            break
                if sp_1d is None:
                    flux_from_2d = np.nansum(
                        sp_2d.flux.value,
                        axis=0
                    )
                    sp_1d = Spectrum(
                        flux=flux_from_2d * sp_2d.flux.unit,
                        spectral_axis=sp_2d.spectral_axis,
                    )
                    sp_1d.meta['file_name'] = None
                    sp_1d.meta['full_path'] = None
                    sp_1d.meta['is_dummy'] = True
                    sp_1d.meta['linked'] = {'2D': sp_2d}
            else:
                exception_tracker.append((
                    file, "Not a supported spectrum type"
                ))
                continue

            self.qapp.processEvents()
            if new_item is not None:
                continue

            new_item = qt_api.QtWidgets.QListWidgetItem(item_name)
            new_item.setCheckState(qt_api.QtCore.Qt.CheckState.Checked)
            new_item.setToolTip(file)
            new_item.setData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole, item_uuid
            )

            info_dict: Dict[str, Any] = copy.deepcopy(self.DEFAULT_SPEC_INFO)
            info_dict['vmin'] = float(np.nanmin(sp_1d.flux.value))
            info_dict['vmax'] = float(np.nanmax(sp_1d.flux.value))

            self.open_spectra_items[item_uuid] = new_item
            self.object_state_dict[item_uuid] = info_dict
            self.main_wnd.spec_list_widget.addItem(new_item)
            self.open_spectra[item_uuid] = sp_1d

        self.cancel_button.hide()
        self.pbar.hide()
        if len(self.open_spectra) > 0:
            self._unlock()
        self.global_state = GlobalState.READY

        self.redrawCurrentSpec()

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
                        f"file: {fname}\n"
                        f"reason: {reason_str}"
                        for fname, reason_str in exception_tracker
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

    def removeSpecItem(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return

        item_uuid = item.data(qt_api.QtCore.Qt.ItemDataRole.UserRole)
        item_row = self.main_wnd.spec_list_widget.row(item)
        self.main_wnd.spec_list_widget.takeItem(item_row)

        self.open_spectra_items.pop(item_uuid)

        if self.xcorr_handler.best_z is not None:
            try:
                self.xcorr_handler.best_z.pop(item_uuid)
            except KeyError:
                pass

        if self.xcorr_handler.corr_data is not None:
            try:
                self.xcorr_handler.best_z.pop(item_uuid)
            except KeyError:
                pass

        try:
            self.object_state_dict.pop(item_uuid)
        except KeyError:
            pass

        del item

        if self.main_wnd.spec_list_widget.count() == 0:
            self.newProject()

    def doRemoveCurrentSpecItems(self) -> None:
        if self.main_wnd.spec_list_widget.count() == 0:
            return

        current_row = self.main_wnd.spec_list_widget.currentRow()
        item = self.main_wnd.spec_list_widget.item(current_row)
        self.removeSpecItem(item)

    def doRemoveSelectedSpecItems(self) -> None:
        if self.main_wnd.spec_list_widget.count() == 0:
            return

        for row in range(self.main_wnd.spec_list_widget.count(), 0, -1):
            item = self.main_wnd.spec_list_widget.item(row - 1)
            self.removeSpecItem(item)


    def doRemoveCurrentTemplateDir(self, *args: Any, **kwargs: Any) -> None:
        item = self.settings_wnd.template_dirs_list_widget.takeItem(
            self.settings_wnd.template_dirs_list_widget.currentRow()
        )
        del item
        self._rebuild_xcorr_template_list = True
        self.updateXcorrTemplateList()

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

        for_rr: Dict[uuid.UUID, Spectrum] = {}
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

        self.redrock_handler.run(for_rr)


    def doStartXCorr(self, *args, **kwargs) -> None:
        if self.xcorr_handler is None:
            # Should never happen since we disable the button to start this
            # action if the xcorr backend is not found
            return

        self.statusbar.showMessage("Running xcorr backend...")

        for_xcorr: Dict[uuid.UUID, Spectrum] = {}

        if self.main_wnd.xcorr_all_radio.isChecked():
            for_xcorr = self.open_spectra

        elif self.main_wnd.xcorr_selected_radio.isChecked():

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
                    for_xcorr[item_uuid] = self.open_spectra[item_uuid]
        else:
            if self.current_uuid is None:
                return
            for_xcorr[self.current_uuid] = self.open_spectra[self.current_uuid]

        self.xcorr_handler.z_min = self.main_wnd.xcorr_z_min_dspinbox.value()
        self.xcorr_handler.z_max = self.main_wnd.xcorr_z_max_dspinbox.value()
        self.xcorr_handler.z_step = self.main_wnd.xcorr_z_step_dspinbox.value()

        self.xcorr_handler.templates = self.getXcorrSelectedTemplates()

        if self.main_wnd.xcorr_wrange_checkbox.isChecked():
            self.xcorr_handler.w_min_ang = float(
                self.main_wnd.xcorr_wav_min_spinbox.value()
            )
            self.xcorr_handler.w_max_ang = float(
                self.main_wnd.xcorr_wav_max_spinbox.value()
            )
        else:
            self.xcorr_handler.w_min_ang = None
            self.xcorr_handler.w_max_ang = None

        if self.main_wnd.xcorr_gpu_checkbox.isChecked():
            self.xcorr_handler.setGpuBackend(self.use_gpu)
            self.xcorr_handler.setGpuDevice(self.compute_device)
        else:
            self.xcorr_handler.setGpuBackend(False)
            self.xcorr_handler.setGpuDevice(None)

        self.xcorr_handler.setContinuumSubtraction(
            self.main_wnd.xcorr_csub_combobox.currentIndex()
        )

        self.xcorr_handler.setFFTSmoothing(
            self.main_wnd.xcorr_csub_fft_sigma_dspinbox.value()
        )

        self.xcorr_handler.setChebyshevOrder(
            self.main_wnd.xcorr_csub_chebyshev_order_spinbox.value()
        )

        w_masks = {}
        for spec_uuid in for_xcorr.keys():
            try:
                mask_dict = self.spectral_masks[spec_uuid]
            except KeyError:
                w_masks[spec_uuid] = []
            else:
                w_masks[spec_uuid] = list(
                    mask
                    for mask in mask_dict.values()
                    if mask['enabled']
                )

        self.pbar.show()
        self.xcorr_handler.run(for_xcorr, w_masks)
        self.pbar.hide()

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

    def doUpdateColorMap(self, *args, **kwargs) -> None:
        if not HAS_MPL:
            return

        if self.main_wnd.flux_tabwidget.currentIndex() == 1:
            target = self.flux_chart_view_2d
        else:
            return

        cmap_name = self.main_wnd.cmap_combo_box.currentData(
                qt_api.QtCore.Qt.ItemDataRole.UserRole
            )
        cmap = colormaps.get(cmap_name)

        if cmap is None:
            return

        vmin = self.main_wnd.cmap_min_dspinbox.value()
        vmax = self.main_wnd.cmap_max_dspinbox.value()
        self.object_state_dict[self.current_uuid]['cmap'] = cmap_name
        self.object_state_dict[self.current_uuid]['vmin'] = vmin
        self.object_state_dict[self.current_uuid]['vmax'] = vmax
        target.setColorMap(cmap, update=False)
        target.setVmin(vmin, update=False)
        target.setVmax(vmax, update=True)

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
            names=['INDEX', 'SPEC_FILE', 'Z', 'QF', 'UUID', 'COMMENT'],
            dtype=[int, str, float, int, str, str]
        )

        for j, (o_uuid, o_sp) in enumerate(self.open_spectra.items()):
            spec_file = o_sp.meta['file_name']

            try:
                info_dict = self.object_state_dict[o_uuid]
            except KeyError:
                redshift = -99.0
                quality_flag = 0
                sanitized_comment = ""
            else:
                if info_dict['redshift'] is None:
                    redshift = -99.0
                else:
                    redshift = info_dict['redshift']

                if (
                    ('comment' not in info_dict) or
                    (info_dict['comment'] is None)
                ):
                    comment = ""
                else:
                    comment = info_dict['comment']

                sanitized_comment = (
                    '_'.join(comment.split())
                    .replace('"', '')
                    .replace("'", "")
                )

                quality_flag = info_dict['quality_flag']

            new_row = (
                j, spec_file, redshift, quality_flag, o_uuid.hex,
                sanitized_comment
            )
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

    def getGpuBackend(self):
        if self.use_gpu:
            return self.use_gpu
        else:
            return  "none"

    def getGpuDevice(self):
        return self.compute_device

    def getGpuDeviceId(self):
        return self.compute_device_id

    def getXcorrSelectedTemplates(self) -> Dict[str, specutils.Spectrum]:

        selected_templates = {}
        for item in self.xcor_tree_leaf_items:
            if item.checkState(0) == qt_api.QtCore.Qt.CheckState.Checked:
                template_id = item.data(
                    0, qt_api.QtCore.Qt.ItemDataRole.UserRole
                )
                selected_templates[template_id] = self.xcorr_templates[template_id]
        return selected_templates



    def getXcorrTemplateDirs(self) -> List[str]:
        xcorr_template_dir_list = []
        for j in range(self.settings_wnd.template_dirs_list_widget.count()):
            item = self.settings_wnd.template_dirs_list_widget.item(j)
            xcorr_template_dir_list.append(
                str(pathlib.Path(item.text()).resolve())
            )

        return xcorr_template_dir_list

    def getXcorrTreeWidgetColumnWidth(self) -> List[int]:
        widths = [
            int(self.main_wnd.xcorr_templates_tree_widget.columnWidth(col))
            for col in range(
                self.main_wnd.xcorr_templates_tree_widget.columnCount()
            )
        ]
        return widths


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

            for obj_uuid, obj_sp in self.open_spectra.items():
                if obj_uuid in matched_uuids:
                    continue
                elif obj_sp.meta['file_name'] == spec_file:
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

    def getCurrentTheme(self) -> str:
        return str(self.settings_wnd.icon_theme_combo_box.currentText())

    def loadIcons(self) -> None:
        self._load_icons(self.getCurrentTheme())

    def loadSettings(self, prog_callback: Optional[Callable] = None) -> None:
        st_items = self._global_setting.items()
        for j, (key, (_, set_func, cast_func)) in enumerate(st_items):
            if prog_callback is not None:
                prog_callback(int(100*j / len(st_items)))

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
        self.open_spectra = {}
        self.object_state_dict = {}
        self.spectral_masks = {}
        self.current_uuid = None
        self.current_project_file_path = None
        self.current_open_dir = None
        self.current_zcat_open_dir = None
        self.current_image_open_dir = None
        self.current_mask_open_dir = None

        self.main_wnd.spec_list_widget.clear()
        self.main_wnd.lines_match_list_widget.clear()
        self.main_wnd.lines_table_widget.setRowCount(0)
        self.main_wnd.obj_prop_table_widget.setRowCount(0)
        self.updateMaskList()

        self.xcorr_handler.clear()
        self.clearXcorrTreeWidgetZ()

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

            if file_info['path'] is None:
                file_path = None
                is_dummy = True
            else:
                file_path = os.path.realpath(
                    os.path.join(
                        os.path.dirname(file_name),
                        file_info['path']
                    )
                )
                is_dummy = False

            if not os.path.exists(file_path):
                if os.path.exists(file_info['path']):
                    file_path = file_info['path']
                else:
                    exception_tracker[item_uuid] = (
                        file_path,
                        f"File {file_path} does not exist!"
                    )
                    continue

            sp: Union[Spectrum, loaders.Spectrum2D]
            try:
                sp = loaders.read(file_path)
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

            sp.meta['file_name'] = os.path.basename(file_path)
            sp.meta['full_path'] = file_path
            sp.meta['is_dummy'] = is_dummy

            try:
                serialized_links = file_info['linked']
            except KeyError:
                pass
            else:
                links_dict = {}
                try:
                    for key, sp_path in serialized_links.items():
                        linked_file_path = os.path.realpath(
                            os.path.join(
                                os.path.dirname(file_name),
                                sp_path
                            )
                        )
                        try:
                            linked_sp = loaders.read(linked_file_path)
                        except Exception as exc:
                            exception_tracker[item_uuid] = (
                                file_path,
                                f"Cannot open linked {key} file {sp_path}: " +
                                str(exc)
                            )
                        else:
                            links_dict[key] = linked_sp
                            if isinstance(sp, Spectrum):
                                linked_sp.meta['linked'] = {'1D': sp}
                            elif isinstance(sp, loaders.Spectrum2D):
                                linked_sp.meta['linked'] = {'2D': sp}
                except AttributeError as exc:
                    exception_tracker[item_uuid] = (
                        file_path, f"Corruped project file: {str(exc)}"
                    )
                    continue
                else:
                    sp.meta['linked'] = links_dict

            self.main_wnd.spec_list_widget.addItem(new_item)
            self.open_spectra[item_uuid] = sp
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
                    'checked': qt_api.QtCore.Qt.CheckState(
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

            try:
                z_comment = obj_info['comment']
            except KeyError:
                z_comment = ""

            self._update_spec_item_qf(obj_uuid, quality_flag)
            self.object_state_dict[obj_uuid] = {
                'redshift': redshift,
                'comment': z_comment,
                'quality_flag': quality_flag,
                'lines': {
                    'list': lines_list,
                    'redshifts': z_list,
                }
            }

        self.spectral_masks = {}
        try:
            spectral_masks = serialized_dict['spectral_masks']
        except KeyError:
            pass
        else:
            for obj_uuid_str, serialized_masks in spectral_masks.items():
                obj_uuid = uuid.UUID(obj_uuid_str)
                self.spectral_masks[obj_uuid] = {}
                for mask_uuid_str, properties in serialized_masks.items():
                    mask_uuid = uuid.UUID(mask_uuid_str)
                    mask_properties: Dict[str, Any] = copy.deepcopy(
                        self.DEFAULT_MASK_PROPERTIES
                    )

                    for key, val in properties.items():
                        mask_properties[key] = val

                    try:
                        if properties['linked'] is not None:
                            mask_properties['linked'] = uuid.UUID(
                                str(properties['linked'])
                            )
                    except KeyError:
                        pass

                    self.spectral_masks[obj_uuid][mask_uuid] = mask_properties

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

    def doPlotXcorrFunction(self, *args, **kwargs) -> None:
        self.plotXcorrFunction()

    def plotXcorrFunction(
            self,
            item: Optional[QtWidgets.QTreeWidgetItem] = None,
            col: int = 0
    ) -> None:
        self.cur_xcorr_leaf_item = item

        xcorr_chart: QtCharts.QChart = self.xcorr_chart_view.chart()
        xcorr_chart.removeAllSeries()
        for ax in xcorr_chart.axes():
            xcorr_chart.removeAxis(ax)

        if item is None:
            sel = self.main_wnd.xcorr_templates_tree_widget.selectedItems()

            if not sel:
                return

            try:
                item = sel[0]
            except IndexError:
                return

        try:
            object_corr = self.xcorr_handler.corr_data[self.current_uuid]
            object_ze = self.xcorr_handler.best_z[self.current_uuid]
        except KeyError:
            return
        else:
            template_id = item.data(
                0, qt_api.QtCore.Qt.ItemDataRole.UserRole
            )

        try:
            vec_z, xcorr = object_corr[template_id]
            best_z = object_ze[template_id]
        except KeyError:
            return

        self.xcorr_chart_view.setRedshift(best_z)
        xcorr_chart = self.xcorr_chart_view.chart()
        xcorr_chart.removeAllSeries()

        for ax in xcorr_chart.axes():
            xcorr_chart.removeAxis(ax)

        xcorr_series = values2series(vec_z, xcorr, self.qapp.tr("X-Corr"))
        xcorr_chart.addSeries(xcorr_series)

        if not xcorr_chart.axes():
            xcorr_axis_x = qt_api.QtCharts.QValueAxis()

            if self.main_wnd.log_y_check_box.isChecked():
                xcorr_axis_y = qt_api.QtCharts.QLogValueAxis()
            else:
                xcorr_axis_y = qt_api.QtCharts.QValueAxis()

            xcorr_chart.addAxis(
                xcorr_axis_x, qt_api.QtCore.Qt.AlignmentFlag.AlignBottom
            )
            xcorr_chart.addAxis(
                xcorr_axis_y, qt_api.QtCore.Qt.AlignmentFlag.AlignLeft
            )
        else:
            xcorr_axis_x = xcorr_chart.axes()[0]
            xcorr_axis_y = xcorr_chart.axes()[1]

        xcorr_axis_x.setTickInterval(500)
        xcorr_axis_x.setLabelFormat("%.2f")
        xcorr_axis_x.setTitleText(str('redshift'))

        xcorr_axis_y.setLabelFormat("%.2f")
        xcorr_axis_y.setTitleText(str('xcorr value'))

        xcorr_series.attachAxis(xcorr_axis_x)
        xcorr_series.attachAxis(xcorr_axis_y)

        xcorr_chart.setContentsMargins(0, 0, 0, 0)
        xcorr_chart.setBackgroundRoundness(0)
        xcorr_chart.legend().hide()

    def doPlotXcorrTemplate(self, *args, **kwargs) -> None:
        self.plotXcorrTemplate()

    def plotXcorrTemplate(
            self,
            item: Optional[QtWidgets.QTreeWidgetItem] = None,
            col: int = 0,
            redshift: Optional[float] = None
    ) -> None:
        if item is None:
            sel = self.main_wnd.xcorr_templates_tree_widget.selectedItems()

            if not sel:
                return

            try:
                item = sel[0]
            except IndexError:
                return

        template_id = item.data(
            0, qt_api.QtCore.Qt.ItemDataRole.UserRole
        )
        self.templates_to_plot['xcorr'] = template_id
        self.drawXcorrTemplate()

    def doPlotTemplates(self, *args, **kwargs) -> None:
        self.plotTemplates()

    def plotTemplates(self, redshift: Optional[float] = None) -> None:
        if redshift is None:
            redshift = self.main_wnd.z_dspinbox.value()

        if not self.main_wnd.plot_templates_check_box.isChecked():
            # remove templates from flux chart
            flux_chart: QtCharts.QChart = self.flux_chart_view.chart()
            for ser in flux_chart.series():
                if (
                    ser.name().startswith('xcorr/') or
                    ser.name().startswith('rrock/')
                ):
                    flux_chart.removeSeries(ser)
            return

        for t_kind, t_id in self.templates_to_plot.items():
            if t_kind == 'xcorr':
                self.drawXcorrTemplate(t_id, redshift)

    def drawXcorrTemplate(
            self,
            template_id: Optional[str] = None,
            redshift: Optional[float] = None
    ) -> None:
        if not self.main_wnd.plot_templates_check_box.isChecked():
            return

        if template_id is None:
            if self.templates_to_plot['xcorr'] is None:
                return

            template_id = self.templates_to_plot['xcorr']

        template_name = f"xcorr/{os.path.basename(template_id)}"

        try:
            template_sp = self.xcorr_templates[template_id]
        except KeyError:
            # NOTE: this should never happen, unless the program state is
            #       horribly broken...
            return

        if redshift is None:
            try:
                object_ze = self.xcorr_handler.best_z[self.current_uuid]
                best_z = object_ze[template_id]
            except KeyError:
                redshift = self.main_wnd.z_dspinbox.value()
            else:
                redshift = best_z

        template_wav: np.ndarray = template_sp.spectral_axis.value
        template_wav = template_wav * (1 + redshift)

        template_flux: np.ndarray = template_sp.flux.value
        mean_template_flux = 10 ** utils.find_exp(np.nanmean(template_flux))
        template_flux = template_flux / mean_template_flux

        flux_chart: QtCharts.QChart = self.flux_chart_view.chart()

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

        # min_y and max_y are used to rescale the template to the current
        # 1D spectrum flux range
        min_y: Union[float, None] = None
        max_y: Union[float, None] = None

        ser: QtCharts.QXYSeries
        for ser in flux_chart.series():
            if ser.name().startswith('xcorr/'):
                flux_chart.removeSeries(ser)
            elif ser.name().lower() == 'flux':
                ser_points = ser.points()
                if (ser_points is not None) and (len(ser_points) > 5):
                    p_y_list = [p.y() for p in ser_points]
                    p_x_list = [p.x() for p in ser_points]
                    min_y = np.percentile(p_y_list, 20)
                    max_y = np.percentile(p_y_list, 80)
                    min_x = np.min(p_x_list)
                    max_x = np.max(p_x_list)

        # Rescale the template to match the min/max flux values of the
        # currently plotted spectrum
        if (min_y is not None) and (max_y is not None):
            t_wav_mask = (template_wav >= min_x) & (template_wav <= max_x)
            if any(t_wav_mask):
                template_flux -= np.percentile(template_flux[t_wav_mask], 20)
                template_flux /= np.percentile(template_flux[t_wav_mask], 80)
            else:
                template_flux -= np.percentile(template_flux, 20)
                template_flux /= np.percentile(template_flux, 80)

            template_flux *= (max_y - min_y)
            template_flux += min_y

        template_series: QtCharts.QLineSeries = values2series(
            template_wav, template_flux, template_name
        )
        
        flux_chart.addSeries(template_series)
        template_series.attachAxis(flux_axis_x)
        template_series.attachAxis(flux_axis_y)
        pen: QtGui.QPen = template_series.pen()
        pen.setColor(qt_api.QtGui.QColor("red"))
        pen.setWidth(2)
        template_series.setPen(pen)
        template_series.setOpacity(0.2)

    def plotXcorrCsub(self, flux_scale: float = 1.0) -> None:
        if self.xcorr_handler is None:
            # Should never happen since we disable the button to start this
            # action if the xcorr backend is not found
            return

        flux_chart: QtCharts.QChart = self.flux_chart_view.chart()
        for ser in flux_chart.series():
            if ser.name().startswith('continuum/'):
                flux_chart.removeSeries(ser)

        if not self.main_wnd.xcorr_csub_preview_checkbox.isChecked():
            return

        sub_cont = self.main_wnd.xcorr_csub_combobox.currentIndex()
        if sub_cont == 0:
            return
        elif sub_cont == 1:
            continuum_sub = 'chebyshev'
        else:
            continuum_sub = 'fft'

        if self.main_wnd.xcorr_gpu_checkbox.isChecked():
            use_gpu = self.use_gpu
            device = self.compute_device
        else:
            use_gpu = False
            device = None

        fft_sigma = self.main_wnd.xcorr_csub_fft_sigma_dspinbox.value()

        cheb_order = self.main_wnd.xcorr_csub_chebyshev_order_spinbox.value()

        curr_sp = self.open_spectra[self.current_uuid]

        spectral_axis_ang = curr_sp.spectral_axis.to(units.angstrom).value
        mask = ~np.isfinite(curr_sp.flux.value)

        try:
            mask_dict = self.spectral_masks[self.current_uuid]
        except KeyError:
            pass
        else:
            for mask_properties in mask_dict.values():
                if not mask_properties['enabled']:
                    continue

                mask[
                    (spectral_axis_ang >= mask_properties['start']) &
                    (spectral_axis_ang <= mask_properties['end'])
                ] = True

        if self.main_wnd.xcorr_wrange_checkbox.isChecked():
            w_min = float(self.main_wnd.xcorr_wav_min_spinbox.value())
            w_max = float(self.main_wnd.xcorr_wav_max_spinbox.value())
            mask[spectral_axis_ang <= w_min] = True
            mask[spectral_axis_ang >= w_max] = True

        cont, c_sub = utils.separate_continuum(
            curr_sp.flux.value / flux_scale,
            fit_mode=continuum_sub,
            fft_sigma=fft_sigma,
            chebyshev_order=cheb_order,
            mask=mask,
            use_gpu=use_gpu,
            device=device
        )

        continuum_series: QtCharts.QLineSeries = values2series(
            spectral_axis_ang,
            cont,
            name=f'continuum/{continuum_sub}'
        )

        pen: QtGui.QPen = continuum_series.pen()
        pen.setColor(qt_api.QtGui.QColor("green"))
        pen.setWidth(2)
        continuum_series.setPen(pen)
        continuum_series.setOpacity(0.5)

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

        flux_chart.addSeries(continuum_series)
        continuum_series.attachAxis(flux_axis_x)
        continuum_series.attachAxis(flux_axis_y)

    def previewSeries(self, series: list[QtCharts.QLineSeries]):

        preview_dialog: QtWidgets.QDialog = qt_api.QtWidgets.QDialog(
            self.main_wnd
        )
        preview_dialog.setMinimumWidth(800)
        preview_dialog.setMinimumHeight(400)

        dlg_layout: QtWidgets.QVBoxLayout = qt_api.QtWidgets.QVBoxLayout()
        preview_dialog.setLayout(dlg_layout)

        temp_sp_view = SpectrumQChartView()
        dlg_layout.addWidget(temp_sp_view)
        flux_chart: QtCharts.QChart = temp_sp_view.chart()

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

        flux_axis_y.setTitleText("Flux [arbitrary units]")
        flux_axis_x.setTitleText("Wavelength [Angstrom]")

        for ser in series:
            flux_chart.addSeries(ser)
            ser.attachAxis(flux_axis_x)
            ser.attachAxis(flux_axis_y)

        preview_dialog.exec()

    def doPreviewXcorrTemplate(self, *args, **kwargs):
        self.previewXcorrTemplate()

    def previewXcorrTemplate(
            self,
            item: Optional[QtWidgets.QTreeWidgetItem] = None,
            col: int = 0
    ):
        """
        Preview a xcorr template in a new dialog window.
        """
        if item is None:
            sel = self.main_wnd.xcorr_templates_tree_widget.selectedItems()

            if not sel:
                return

            try:
                item = sel[0]
            except IndexError:
                return

        template_id = item.data(
            0, qt_api.QtCore.Qt.ItemDataRole.UserRole
        )

        try:
            template_sp = self.xcorr_templates[template_id]
        except KeyError:
            # NOTE: this should never happen, unless the program state is
            #       horribly broken...
            return

        template_wav_ang: np.ndarray = template_sp.spectral_axis.to("AA").value
        template_flux: np.ndarray = template_sp.flux.value

        template_series: QtCharts.QLineSeries = values2series(
            template_wav_ang,
            template_flux,
            name=str(template_id)
        )

        pen: QtGui.QPen = template_series.pen()
        pen.setColor(qt_api.QtGui.QColor("black"))
        pen.setWidth(2)
        template_series.setPen(pen)
        template_series.setOpacity(0.5)

        self.previewSeries([template_series, ])

    def redrawCurrentSpec(self, *args, **kwargs) -> None:
        """Redraw the charts."""
        self.currentSpecItemChanged(
            self.main_wnd.spec_list_widget.currentItem()
        )

    def requestCancelCurrentOperation(self) -> None:
        """Set the global state of the program to REUQUEST_CANCEL."""
        self.global_state = GlobalState.REUQUEST_CANCEL

    def restoreSettings(self) -> None:
        self._rebuild_xcorr_template_list = False
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

            spec_meta = self.open_spectra[item_uuid].meta


            file_info: Dict[str, Any] = {
                'index': k,
                'uuid': item_uuid.hex,
                'text': item.text(),
                'checked': int(item.checkState().value),
            }

            try:
                is_dummy = bool(spec_meta['is_dummy'])
            except KeyError:
                is_dummy = False

            if is_dummy or (spec_meta['full_path'] is None):
                file_info['path'] = None
            else:
                file_info['path'] = os.path.relpath(
                    spec_meta['full_path'], proj_path
                )

            try:
                serialized_links: Dict[str, Any] = {
                    sp_type: os.path.relpath(
                        sp.meta['full_path'], proj_path
                    )
                    for sp_type, sp in spec_meta['linked'].items()
                }
            except KeyError:
                pass
            else:
                file_info['linked'] = serialized_links

            serialized_open_file_list.append(file_info)

        # Serialize objects info for json dumping
        serialized_object_info_dict: Dict[str, Any] = {}
        serialized_object_mask_dict: Dict[str, Any] = {}
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

            serialized_info_dict: Dict[str, Any] = copy.deepcopy(obj_info)
            serialized_info_dict['lines'] = {
                'list': serialized_lines_list,
                'redshifts': serialized_z_list,
            }

            serialized_object_info_dict[obj_uuid.hex] = serialized_info_dict

            try:
                masks = self.spectral_masks[obj_uuid]
            except KeyError:
                pass
            else:
                serialized_masks: Dict[str, Any] = {}
                for mask_uuid, masks_properties in masks.items():
                    serialized_masks[mask_uuid.hex] = copy.deepcopy(
                        masks_properties
                    )
                    try:
                        linked_uuid = masks_properties['linked']
                    except KeyError:
                        serialized_link = None
                    else:
                        try:
                            serialized_link = linked_uuid.hex
                        except Exception:
                            serialized_link = None
                    serialized_masks[mask_uuid.hex]['linked'] = serialized_link

                serialized_object_mask_dict[obj_uuid.hex] = serialized_masks

        if self.current_uuid is None:
            serialized_current_uuid = None
        else:
            serialized_current_uuid = self.current_uuid.hex

        project_dict = {
            'current_uuid': serialized_current_uuid,
            'open_files': serialized_open_file_list,
            'objects_properties': serialized_object_info_dict,
            'spectral_masks': serialized_object_mask_dict,
        }

        with open(file_name, 'w') as f:
            json.dump(project_dict, f, indent=2)

    def saveSettings(self) -> None:
        for key, (get_func, _, _) in self._global_setting.items():
            val = get_func()
            self.settings.setValue(key, val)

    def setGpuBackend(self, backend_index: Union[int, str, None]):
        self.settings_wnd.gpu_device_combo_box.clear()
        if backend_index == 0 or backend_index is None:
            self.settings_wnd.gpu_device_combo_box.setEnabled(False)
            self.use_gpu = False
        else:
            self.settings_wnd.gpu_device_combo_box.setEnabled(True)
            if backend_index == 1 or backend_index == 'tf':
                self.use_gpu = 'tf'
                self.settings_wnd.gpu_backend_combo_box.setCurrentIndex(1)
            elif  backend_index == 2 or backend_index == 'torch':
                self.use_gpu = 'torch'
                self.settings_wnd.gpu_backend_combo_box.setCurrentIndex(2)
            else:
                self.settings_wnd.gpu_device_combo_box.clear()
                self.settings_wnd.gpu_device_combo_box.setEnabled(False)
                self.use_gpu = False
                return

            for device_name, device in utils.DEVICE_DICT[self.use_gpu].items():
                self.settings_wnd.gpu_device_combo_box.addItem(
                    device_name
                )

            if self.settings_wnd.gpu_device_combo_box.count() > 0:
                self.settings_wnd.gpu_device_combo_box.setCurrentIndex(0)
            else:
                self.compute_device = None

    def setGpuDevice(self, device_index: int):
        device_id = self.settings_wnd.gpu_device_combo_box.itemText(
            device_index
        )

        try:
            self.compute_device = utils.DEVICE_DICT[self.use_gpu][device_id]
        except KeyError:
            self.compute_device_id = -1
            self.setGpuBackend(None)
            return
        else:
            self.compute_device_id = device_id

    def setGpuDeviceId(self, device_id):
        for k in range(self.settings_wnd.gpu_device_combo_box.count()):
            id = self.settings_wnd.gpu_device_combo_box.itemText(k)
            if id == device_id:
                self.setGpuDevice(k)
                break
        else:
            self.setGpuBackend(None)

    def setCurrentObjectRedshift(self, redshift: Optional[float]) -> None:
        if redshift is None:
            redshift = 0

        self.main_wnd.z_dspinbox.setValue(redshift)

        self.flux_chart_view.setRedshift(redshift=redshift)
        self.flux_chart_view_2d.setRedshift(redshift=redshift)
        self.plotTemplates(redshift=redshift)

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

    def doSetCurrentObjectRedshiftFromXcorr(self, *args, **kwargs) -> None:
        self.setCurrentObjectRedshiftFromXcorr()

    def setCurrentObjectRedshiftFromXcorr(
            self,
            item: Optional[QtWidgets.QTreeWidgetItem] = None,
            col: int = 0
    ) -> None:
        if item is None:
            sel = self.main_wnd.xcorr_templates_tree_widget.selectedItems()

            if not sel:
                return

            try:
                item = sel[0]
            except IndexError:
                return

        try:
            object_ze = self.xcorr_handler.best_z[self.current_uuid]
        except KeyError:
            return
        else:
            template_id = item.data(
                0, qt_api.QtCore.Qt.ItemDataRole.UserRole
            )

        try:
            template_z = object_ze[template_id]
        except KeyError:
            return

        self.setCurrentObjectRedshift(template_z)

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
            self.flux_chart_view_2d.setLinesType('')
        elif type_index == 1:
            self.flux_chart_view.setLinesType('E')
            self.flux_chart_view_2d.setLinesType('E')
        elif type_index == 2:
            self.flux_chart_view.setLinesType('A')
            self.flux_chart_view_2d.setLinesType('A')

    def setXcorrTemplateDirs(self, dir_list: List[str]) -> None:
        for dir_path in dir_list:
            self.addNewTemplateDir(dir_path, update=False)
        self.updateXcorrTemplateList()

    def setXcorrTreeWidgetColumnWidth(self, widths: List[int]) -> None:
        for col, col_width in enumerate(widths):
            self.main_wnd.xcorr_templates_tree_widget.setColumnWidth(
                col, int(col_width)
            )

    def showObjectInfo(self, object_uuid: uuid.UUID) -> None:
        sp: Spectrum = self.open_spectra[object_uuid]

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

    def showXcorrZ(self, object_uuid: Optional[uuid.UUID] = None) -> None:
        if self.xcorr_handler.best_z is None:
            return

        if object_uuid is None:
            if self.current_uuid is None:
                return
            object_uuid = self.current_uuid

        try:
            object_ze = self.xcorr_handler.best_z[object_uuid]
        except KeyError:
            self.clearXcorrTreeWidgetZ()
        else:
            for item in self.xcor_tree_leaf_items:
                template_id = item.data(
                    0, qt_api.QtCore.Qt.ItemDataRole.UserRole
                )

                try:
                    template_z = object_ze[template_id]
                except KeyError:
                    item.setText(1, "None")
                else:
                    item.setText(1, str(template_z))

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

    def run(self) -> None:
        """Run the main Qt Application."""
        self.main_wnd.show()
        sys.exit(self.qapp.exec())

    def updateXcorrTemplateTreeState(self, *args: Any, **kwargs: Any) -> None:
        if self._updateXcorrTemplateTreeState_lock:
            return

        def _set_children_state(
            item: QtWidgets.QTreeWidgetItem,
            state: QtCore.Qt.CheckState
        ) -> None:
            n_children = item.childCount()

            item.setCheckState(0, state)
            if n_children > 0:
                for j in range(item.childCount()):
                    _set_children_state(item.child(j), state)

        def _set_parents_state(item: QtWidgets.QTreeWidgetItem):
            parent = item.parent()

            if parent is None:
                return
            elif isinstance(parent, qt_api.QtWidgets.QTreeWidget):
                return

            sibling_count = parent.childCount()
            checked_count = 0
            unchecked_count = 0
            for j in range(sibling_count):
                check_state = parent.child(j).checkState(0)
                if check_state == qt_api.QtCore.Qt.CheckState.Checked:
                    checked_count += 1
                elif check_state == qt_api.QtCore.Qt.CheckState.Checked:
                    unchecked_count += 1

            if checked_count == sibling_count:
                parent_state = qt_api.QtCore.Qt.CheckState.Checked
            elif unchecked_count == sibling_count:
                parent_state = qt_api.QtCore.Qt.CheckState.Unchecked
            else:
                parent_state = qt_api.QtCore.Qt.CheckState.PartiallyChecked

            parent.setCheckState(0, parent_state)

            _set_parents_state(parent)

        self._updateXcorrTemplateTreeState_lock = True
        _set_children_state(args[0], args[0].checkState(0))
        _set_parents_state(args[0])
        self._updateXcorrTemplateTreeState_lock = False

    def updateXcorrTemplateList(self) -> None:

        if not self._rebuild_xcorr_template_list:
            return

        folder_icon = qt_api.get_qicon("folder", self.getCurrentTheme())
        book_icon = qt_api.get_qicon("book", self.getCurrentTheme())

        def _build_folder_tree(
                parts: List[str],
                parent: QtWidgets.QTreeWidgetItem
        ) -> Union[None, QtWidgets.QTreeWidgetItem]:
            if len(parts) <= 1:
                return None

            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.text(0) == parts[0]:
                    break
            else:
                child = qt_api.QtWidgets.QTreeWidgetItem(parent)
                child.setText(0, parts[0])
                child.setCheckState(0, qt_api.QtCore.Qt.CheckState.Checked)
                if len(parts) == 2:
                    child.setIcon(0, book_icon)
                    child.setData(
                        0, qt_api.QtCore.Qt.ItemDataRole.UserRole,
                        parts[1]
                    )
                    return child
                else:
                    child.setIcon(0, folder_icon)

            return _build_folder_tree(parts[1:], child)

        self.main_wnd.xcorr_templates_tree_widget.clear()
        self.main_wnd.xcorr_templates_tree_widget.setStyleSheet(
            "QTreeView::item:!has-children:hover{background-color:#8cc2ff;}"
        )
        self.xcor_tree_leaf_items = []
        self.xcorr_templates = {}

        self.settings_wnd.template_files_list_widget.clear()
        for dir_path in self.getXcorrTemplateDirs():
            template_dict = backends.xcorr.load_templates(dir_path)

            self.xcorr_templates[dir_path] = template_dict

            top_level_item = qt_api.QtWidgets.QTreeWidgetItem(
                self.main_wnd.xcorr_templates_tree_widget
            )
            top_level_item.setText(0, dir_path)
            top_level_item.setIcon(0, folder_icon)

            self.qapp.processEvents()

            for local_template_id, template_sp in template_dict.items():
                self.settings_wnd.template_files_list_widget.addItem(
                    local_template_id
                )
                self.qapp.processEvents()
                template_id = dir_path + '::' + local_template_id
                path_parts = list(pathlib.PurePath(local_template_id).parts)
                path_parts.append(template_id)

                self.xcorr_templates[template_id] = template_sp

                self.xcor_tree_leaf_items.append(
                    _build_folder_tree(
                        path_parts,
                        parent=top_level_item
                    )
                )
        self.main_wnd.xcorr_templates_tree_widget.expandAll()
        self._rebuild_xcorr_template_list = False


def values2series(
    x_values: Union[List[float], np.ndarray],
    y_values: Union[List[float], np.ndarray],
    name: str
) -> QtCharts.QLineSeries:
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
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        series.append(x, y)

    return series


def main() -> None:
    """Run the main GUI application using PySide."""
    myapp: GuiApp = GuiApp()
    myapp.run()


if __name__ == '__main__':
    main()
