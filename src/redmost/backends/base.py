# Copyright (c) 2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Base classes and interfaces for Redmost backend plugins."""

from __future__ import annotations

from typing import Optional, Any
from redmost.qt_compat import qt_api


class BackendPlugin(qt_api.QtCore.QObject):
    """Abstract base class for Redmost backend plugins.

    Each backend exposes its execution interface, its main widget to be
    docked in the main window tabs, and an optional settings widget to be
    embedded into the application settings dialog.
    """

    started = qt_api.Signal()
    finished = qt_api.Signal(object)
    cancelled = qt_api.Signal()
    progress = qt_api.Signal(int)
    maximum = qt_api.Signal(int)
    message = qt_api.Signal(str)
    templates_changed = qt_api.Signal()

    backend_id: str = ""
    backend_name: str = ""

    def __init__(self, parent: Optional[qt_api.QtCore.QObject] = None) -> None:
        """Initialize the backend plugin."""
        super().__init__(parent)

    def is_available(self) -> bool:
        """Return True if backend dependencies are installed and available."""
        return True

    def get_main_widget(
        self, parent: Optional[qt_api.QtWidgets.QWidget] = None
    ) -> qt_api.QtWidgets.QWidget:
        """Return the main GUI widget to display in the main window tab."""
        raise NotImplementedError

    def get_run_button(self) -> Optional[qt_api.QtWidgets.QPushButton]:
        """Return the main Run/Cancel QPushButton if present."""
        return None

    def is_running(self) -> bool:
        """Return True if the backend is currently executing a job."""
        return False

    def run(self, targets: list[Any], **kwargs: Any) -> None:
        """Execute the backend computation on the given targets."""
        raise NotImplementedError

    def cancel(self) -> None:
        """Cancel the currently running computation."""
        pass

    def disable_controls(self) -> None:
        """Disable internal controls while a job is running."""
        pass

    def restore_controls(self) -> None:
        """Restore internal controls when a job completes or is cancelled."""
        pass

    def get_settings_widget(
        self, parent: Optional[qt_api.QtWidgets.QWidget] = None
    ) -> Optional[qt_api.QtWidgets.QWidget]:
        """Return the widget to display in the Settings dialog, or None."""
        return None

    def load_settings(self, settings: qt_api.QtCore.QSettings) -> None:
        """Load settings from the persistent QSettings storage."""
        pass

    def apply_settings(self, settings: qt_api.QtCore.QSettings) -> None:
        """Save settings to the persistent QSettings storage."""
        pass

    def setup_gui(self, app: Any) -> None:
        """Register widgets and wire up signals with the main GuiApp."""
        pass
