#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

BSD 3-Clause License

Copyright (c) 2023-2024, Maurizio D'Addona <mauritiusdadd@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
