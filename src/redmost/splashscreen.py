# Copyright (c) 2023-2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
#
# This file is part of Redmost.
#
# Redmost is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Redmost is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Redmost.  If not, see <https://www.gnu.org/licenses/>.
"""Splash screen implementation for Redmost application."""

from pathlib import Path
from typing import TYPE_CHECKING
from redmost.qt_compat import qt_api, get_qapp
from redmost.utils import get_data_file


if TYPE_CHECKING:
    # NOTE: These imports will be actually executed only by the type-checking
    #       programs, like mypy.
    from redmost.qt_compat import QtCore, QtGui, QtWidgets, QtCharts
    from redmost.qt_compat import Signal


class SplashScreen(qt_api.QtCore.QObject):

    def __init__(self):

        super(qt_api.QtCore.QObject, self).__init__()

        self.qapp = get_qapp()

        splashfile = get_data_file(
            str(Path("pics") / "splashscreen.jpg")
        )

        self._msg = ""
        self._pxm = qt_api.QtGui.QPixmap(splashfile)
        self._qss = qt_api.QtWidgets.QSplashScreen(
            self._pxm,
            qt_api.QtCore.Qt.WindowType.WindowStaysOnTopHint |
            qt_api.QtCore.Qt.WindowType.X11BypassWindowManagerHint
        )
        self._progress = qt_api.QtWidgets.QProgressBar(self._qss)

        h = self._pxm.height()
        w = self._pxm.width()

        self._progress.setMaximum(100)
        self._progress.setMinimum(0)
        self._progress.setGeometry(10, h-50, w-20, 20)
        self._progress.setStyleSheet("""
        QSplashScreen QProgressBar {
            border: 1px solid gray;
            border-radius: 0px;
            background-color: black;
            color: white;
            height: 1em;
        }

        QSplashScreen QProgressBar::chunk {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                       stop: 0 #0A0AFF, stop: 1 #0A0A9B);
            width: 20px;
        }
        """)

        self._qss.show()
        self.processEvents()

    def close(self):
        self.update()
        self._qss.close()

    def setMaximum(self, val):
        self._progress.setMaximum(val)

    def setMinimum(self, val):
        self._progress.setMainimum(val)

    def setValue(self, val):
        self._progress.setValue(val)
        self.update()

    def maximum(self):
        return self._progress.maximum()

    def minimum(self):
        return self._progress.mainimum()

    def value(self):
        return self._progress.value()

    def message(self):
        return self._msg

    def showMessage(self, msg):
        self._msg = msg
        self._progress.setFormat("    " + str(msg) + " %p%")

    def update(self):
        self._qss.update()
        self.processEvents()

    def finish(self, qwid):
        self._qss.finish(qwid)

    def processEvents(self):
        if self.qapp is not None:
            self.qapp.processEvents()