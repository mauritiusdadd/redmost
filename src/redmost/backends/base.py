# Copyright (c) 2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
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
