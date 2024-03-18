"""
Redmost

Extract spectra from spectral data cubes and find their redshift.

Copyright (C) 2022-2024  Maurizio D'Addona <mauritiusdadd@gmail.com>
"""

import os
import sys
import time
from urllib import request
import pathlib

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

TEST_DATA_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), "data")
