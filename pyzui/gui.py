#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:33:40 2023.

@author: daddona
"""
from __future__ import annotations

import os
import sys
import json
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

        for sibling in self.siblings:
            sibling.setAxesRange(x_min, y_min, x_max, y_max)
        self._sibling_locked = False

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
        [
            'READY',
            'WAITING',
            'SELECT_LINE_MANUAL',
            'SAVE_OBJECT_STATE',
            'LOAD_OBJECT_STATE',
            'REUQUEST_CANCEL',
        ]
    )

    def __init__(self, qt_backend: str) -> None:
        self.qapp: QtWidgets.QApplication = getQApp()

        self.open_spectra: Dict[uuid.UUID, Spectrum1D] = {}
        self.current_redshift: Union[float, None] = None
        self.current_spec_uuid: uuid.UUID | None = None
        self.object_state_dict: Dict[uuid.UUID, Any] = {}

        self.global_state: Enum = self.GlobalState.READY

        self.main_wnd: QtWidgets.QMainWindow = loadUiWidget(
            "main_window.ui", qt_backend=qt_backend
        )

        # Status Bar
        self.mousePosLabel: QtWidgets.QLabel = QtWidgets.QLabel("")

        self.cancel_button: QtWidgets.QPushButton = QtWidgets.QPushButton()
        self.cancel_button.setText("Cancel")
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(
            self.requestCancelCurrentOperation
        )

        self.pbar: QtWidgets.QProgressBar = QtWidgets.QProgressBar()
        self.pbar.hide()

        self.statusbar: QtWidgets.QStatusBar = self.main_wnd.statusBar()
        if self.statusbar is None:
            self.statusbar = QtWidgets.QStatusBar()
            self.main_wnd.setStatusBar(self.statusbar)

        self.statusbar.addPermanentWidget(self.pbar)
        self.statusbar.addPermanentWidget(self.cancel_button)
        self.statusbar.addPermanentWidget(self.mousePosLabel)

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

        self.main_wnd.deleteLinePushButton.clicked.connect(
            self.doDeleteCurrentLine
        )

        self.main_wnd.deleteLinesPushButton.clicked.connect(
            self.doDeleteAllLines
        )

        self.main_wnd.matchLinesPushButton.clicked.connect(
            self.doGetRedshiftFromLines
        )

    def _backup_current_object_state(self) -> None:
        """
        Backup the program state for the current object

        Returns
        -------
        None

        """
        if self.current_spec_uuid is None:
            return

        self.global_state = self.GlobalState.SAVE_OBJECT_STATE
        smooting_dict = {
            'state': self.main_wnd.smoothingCheckBox.checkState(),
            'value': self.main_wnd.smoothingDoubleSpinBox.value()
        }

        lines_list = []
        for row_index in range(self.main_wnd.linesTableWidget.rowCount()):
            w_item = self.main_wnd.linesTableWidget.item(row_index, 0)
            line_info = {
                'row': row_index,
                'data': float(w_item.data(QtCore.Qt.ItemDataRole.UserRole)),
                'text': w_item.text(),
                'checked': w_item.checkState()
            }
            lines_list.append(line_info)

        redshifts_form_lines = []
        for row_index in range(self.main_wnd.linesMatchListWidget.count()):
            z_item = self.main_wnd.linesMatchListWidget.item(row_index)
            z_info = {
                'row': row_index,
                'text': z_item.text(),
                'data': z_item.data(QtCore.Qt.ItemDataRole.UserRole)
            }
            redshifts_form_lines.append(z_info)

        obj_state = {
            'smoothing': smooting_dict,
            'redshift': self.current_redshift,
            'lines': {
                'list': lines_list,
                'redshifts': redshifts_form_lines
            }
        }

        self.object_state_dict[self.current_spec_uuid] = obj_state
        self.global_state = self.GlobalState.READY

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
        self.global_state = self.GlobalState.LOAD_OBJECT_STATE

        # Clear current content of the widgets
        self.main_wnd.linesTableWidget.setRowCount(0)
        self.main_wnd.linesMatchListWidget.clear()

        # If object has not been saved yes, use dafult values
        if obj_uuid not in self.object_state_dict.keys():
            self.main_wnd.smoothingCheckBox.setCheckState(
                QtCore.Qt.CheckState.Unchecked
            )
            self.main_wnd.smoothingDoubleSpinBox.setValue(3)
            self.global_state = self.GlobalState.READY
            return

       # otherwise restore any old previously saved data
        old_state = self.object_state_dict[obj_uuid]
        self.main_wnd.smoothingCheckBox.setCheckState(
            old_state['smoothing']['state']
        )
        self.main_wnd.smoothingDoubleSpinBox.setValue(
            old_state['smoothing']['value']
        )

        lines = old_state['lines']['list']
        self.main_wnd.linesTableWidget.setRowCount(len(lines))
        for line_info in lines:
            w_item = QtWidgets.QTableWidgetItem(line_info['text'])
            w_item.setData(QtCore.Qt.ItemDataRole.UserRole, line_info['data'])
            w_item.setCheckState(line_info['checked'])
            self.main_wnd.linesTableWidget.setItem(line_info['row'], 0, w_item)

        redshifts_form_lines = old_state['lines']['redshifts']
        for z_info in redshifts_form_lines:
            z_item = QtWidgets.QListWidgetItem(z_info['text'])
            z_item.setData(QtCore.Qt.ItemDataRole.UserRole, z_info['data'])
            self.main_wnd.linesMatchListWidget.insertItem(
                z_info['row'], z_item
            )

        self.global_state = self.GlobalState.READY

    def _lock(self, *args, **kwargs) -> None:
        self.main_wnd.specGroupBox.setEnabled(False)
        self.main_wnd.redGroupBox.setEnabled(False)
        self.main_wnd.infoGroupBox.setEnabled(False)
        self.main_wnd.plotGroupBox.setEnabled(False)
        self.fluxQChartView.setRubberBand(
            QtCharts.QChartView.RubberBand.NoRubberBand
        )

    def _unlock(self, *args, **kwargs) -> None:
        self.main_wnd.specGroupBox.setEnabled(True)
        self.main_wnd.redGroupBox.setEnabled(True)
        self.main_wnd.infoGroupBox.setEnabled(True)
        self.main_wnd.plotGroupBox.setEnabled(True)
        self.fluxQChartView.setRubberBand(
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
        new_item = QtWidgets.QTableWidgetItem(f"{wavelength:.2f} A")
        new_item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsSelectable |
            QtCore.Qt.ItemFlag.ItemIsEnabled |
            QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        new_item.setCheckState(QtCore.Qt.CheckState.Checked)
        new_item_row: int = self.main_wnd.linesTableWidget.rowCount()
        new_item.setData(QtCore.Qt.ItemDataRole.UserRole, wavelength)
        self.main_wnd.linesTableWidget.setRowCount(new_item_row + 1)
        self.main_wnd.linesTableWidget.setItem(new_item_row, 0, new_item)

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
        if self.current_spec_uuid is None:
            return
        self._lock()
        self.main_wnd.specGroupBox.setEnabled(False)
        self.global_state = self.GlobalState.SELECT_LINE_MANUAL
        self.qapp.setOverrideCursor(QtCore.Qt.CursorShape.CrossCursor)

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
        if self.current_spec_uuid is None:
            return

        self.main_wnd.linesTableWidget.removeRow(
            self.main_wnd.linesTableWidget.currentRow()
        )

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
        self.main_wnd.linesTableWidget.setRowCount(0)

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
        if self.current_spec_uuid is None:
            return

        sp: Spectrum1D = self.open_spectra[self.current_spec_uuid]

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
                QtCore.Qt.ItemFlag.ItemIsEnabled |
                QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            new_item.setData(QtCore.Qt.ItemDataRole.UserRole, w)
            new_item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.main_wnd.linesTableWidget.setItem(j, 0, new_item)

    def doSaveCurrentProject(self):
        raise NotImplementedError()

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

        self._lock()
        self.global_state = self.GlobalState.WAITING

        n_files = len(file_list)

        self.pbar.setMaximum(n_files)
        self.pbar.show()
        self.cancel_button.show()
        for j, file in enumerate(file_list):
            if self.global_state == self.GlobalState.REUQUEST_CANCEL:
                break

            self.pbar.setValue(j + 1)
            self.statusbar.showMessage(
                f"Loading file {j + 1:d} of {n_files}..."
            )
            self.qapp.processEvents()

            item_uuid: uuid.UUID = uuid.uuid4()
            sp: Spectrum1D = utils.loadSpectrum(file)

            new_item = QtWidgets.QListWidgetItem(f"{sp.obj_id}")
            new_item.setCheckState(QtCore.Qt.CheckState.Checked)
            new_item.setToolTip(file)
            new_item.setData(QtCore.Qt.ItemDataRole.UserRole, item_uuid)

            self.open_spectra[item_uuid] = sp
            self.main_wnd.specListWidget.addItem(new_item)
            self.qapp.processEvents()

        self.cancel_button.hide()
        self.pbar.hide()
        self._unlock()
        self.global_state = self.GlobalState.READY

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

    def doGetRedshiftFromLines(self, *args, **kwargs) -> None:
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
        line_table: QtWidgets.QTableWidget = self.main_wnd.linesTableWidget
        z_list: QtWidgets.QListWidget = self.main_wnd.linesMatchListWidget
        tol: float = self.main_wnd.linesMatchingTolDoubleSpinBox.value()
        z_min: float = self.main_wnd.zMinDoubleSpinBox.value()
        z_max: float = self.main_wnd.zMaxDoubleSpinBox.value()


        # Build a list of selected lines to be used.
        # If no lines are selected, then use all lines.
        lines_lam = []
        for row_index in range(line_table.rowCount()):
            item: QtWidgets.QTableWidgetItem = line_table.item(row_index, 0)

            if item.checkState() != QtCore.Qt.CheckState.Checked:
                # Ignore lines that are not selected
                continue

            lines_lam.append(float(item.data(QtCore.Qt.ItemDataRole.UserRole)))

        best_z_values, best_z_probs = lines.get_redshift_from_lines(
            lines_lam, z_min=z_min, z_max=z_max, tol=tol
        )

        z_list.clear()
        for z, prob in zip(best_z_values, best_z_probs):
            new_z_item = QtWidgets.QListWidgetItem(f"z={z:.4f} (p={prob:.4f})")
            z_list.addItem(new_z_item)


    def doZoomReset(self, *arg, **kwargs) -> None:
        self.fluxQChartView.zoomReset()

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
        if self.global_state == self.GlobalState.SELECT_LINE_MANUAL:
            self.addLine(data_pos.x())
            self.global_state = self.GlobalState.READY
            self.qapp.restoreOverrideCursor()
            self._unlock()

    def requestCancelCurrentOperation(self) -> None:
        self.global_state = self.GlobalState.REUQUEST_CANCEL

    def toggleSmothing(self, show_smoothing: int) -> None:
        self.redrawCurrentSpec()

    def setSmoothingFactor(self, smoothing_value: float) -> None:
        self.redrawCurrentSpec()

    def redrawCurrentSpec(self, *args, **kwargs) -> None:
        self.currentSpecItemChanged(
            self.main_wnd.specListWidget.currentItem()
        )

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
        if self.global_state != self.GlobalState.READY:
            return

        self._unlock()

        spec_uuid: uuid.UUID = new_item.data(
            QtCore.Qt.ItemDataRole.UserRole
        )

        flux_chart = self.fluxQChartView.chart()
        flux_chart.removeAllSeries()

        var_chart = self.varQChartView.chart()
        var_chart.removeAllSeries()

        self._backup_current_object_state()
        if spec_uuid != self.current_spec_uuid:
            # If we actually change the spectrum, then reset the view
            self._restore_object_state(spec_uuid)
            self.current_spec_uuid = spec_uuid

            for ax in flux_chart.axes():
                flux_chart.removeAxis(ax)
            for ax in var_chart.axes():
                var_chart.removeAxis(ax)

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

        smoothing_check_state = self.main_wnd.smoothingCheckBox.checkState()
        if smoothing_check_state == QtCore.Qt.CheckState.Checked:
            smoothing_factor = self.main_wnd.smoothingDoubleSpinBox.value()

            flux_series.setOpacity(0.2)
            smoothing_sigma = len(flux) / (1 + 2 * smoothing_factor)
            smoothed_flux = utils.smooth_fft(flux, sigma=smoothing_sigma)
            smoothed_flux_series = values2series(
                wav, smoothed_flux, "Smoothed flux"
            )

            pen: QtGui.QPen = smoothed_flux_series.pen()
            pen.setColor(QtGui.QColor("orange"))
            pen.setWidth(2)
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
