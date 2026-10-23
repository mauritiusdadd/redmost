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

"""Backends package providing modular redshift estimation plugins."""

from __future__ import annotations

from typing import Any, Optional, Type
from redmost.backends.base import BackendPlugin
from redmost.backends.redrock import RedrockPlugin, rrock, HAS_REDROCK

try:
    from redmost.backends.xcorr import XCorrPlugin, xcorr, HAS_XCORR
except (ImportError, ModuleNotFoundError):
    XCorrPlugin = None  # type: ignore[assignment,misc]
    xcorr = None  # type: ignore[assignment]
    HAS_XCORR = False

# Registry of available backend plugin classes
BACKEND_PLUGINS: list[Type[BackendPlugin]] = [RedrockPlugin]
if XCorrPlugin is not None:
    BACKEND_PLUGINS.append(XCorrPlugin)

__all__ = [
    "BackendPlugin",
    "BACKEND_PLUGINS",
    "RedrockPlugin",
    "XCorrPlugin",
    "rrock",
    "xcorr",
    "HAS_REDROCK",
    "HAS_XCORR",
]

