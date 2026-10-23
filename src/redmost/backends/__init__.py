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

