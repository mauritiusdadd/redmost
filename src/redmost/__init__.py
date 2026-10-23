#!/usr/bin/env python3
"""
@author: Maurizio D'Addona <mauritiusdadd@gmail.com>
"""

try:
    from redmost._version import version as __version__  # type: ignore
    from redmost._version import version_tuple  # type: ignore
except ImportError:
    __version__ = "unknown"
    version_tuple = (0, 0, 0, "unknown")

GITHUB_REPO_URL = "https://github.com/mauritiusdadd/redmost"
ONLINE_DOC_URL = "https://redmost.readthedocs.io/en/latest"
PYPI_REPO_PAGE = "https://pypi.org/project/redmost/"
PYPI_REPO_API_URL = "https://pypi.python.org/pypi/redmost/json"
ICON_PACKAGE_URL = "https://github.com/feathericons/feather"

ABOUT_TEXT = """

ACKNOWLEDGEMENTS

If you use this software for your work, please consider to cite

- <a href="https://zenodo.org/records/10818017">zenodo.org/records/10818017</a>

Also remember to acknowledge:

- astropy: <a href=https://www.astropy.org/acknowledging.html>astropy.org</a>
- specutils: <a href=https://github.com/astropy/specutils/blob/main/specutils/CITATION>github.com/astropy/specutils</a>
- redrock: <a href=https://github.com/desihub/redrock>github.com/desihub/redrock</a> (if you use the redrock backend)

This program uses icons derived from the following themes:

- feather: <a href=https://github.com/feathericons/feather>github.com/feathericons/feather</a>

LICENSE

LICENSE

GNU General Public License v3.0 or later (GPL-3.0-or-later)

Copyright (C) 2023-2026 Maurizio D'Addona <mauritiusdadd@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

