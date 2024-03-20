[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10817885.svg)](https://doi.org/10.5281/zenodo.10817885)
[![DOCS](https://readthedocs.org/projects/redmost/badge/?version=latest)](https://redmost.readthedocs.io/en/latest/) [![Coverage Status](https://coveralls.io/repos/github/mauritiusdadd/redmost/badge.svg?branch=main)](https://coveralls.io/github/mauritiusdadd/redmost?branch=main) [![Testing PyQt6](https://github.com/mauritiusdadd/redmost/actions/workflows/test_linux_pyqt6.yml/badge.svg)](https://github.com/mauritiusdadd/redmost/actions/workflows/test_linux_pyqt6.yml) [![Testing PySide6](https://github.com/mauritiusdadd/redmost/actions/workflows/test_linux_pyside6.yml/badge.svg)](https://github.com/mauritiusdadd/redmost/actions/workflows/test_linux_pyside6.yml)
# REDMOST 

<b>RED</b>shift <b>M</b>easurement <b>O</b>f <b>S</b>pec<b>T</b>ra is a Qt6 Graphical User Interface (GUI) to do redshift measurements on 1D spectra.

If [redrock][1] is correctly installed, it cat be used as a backend to measure the redshift.

# Installation

This is a python program based on Qt6 and supports both PyQt6 and PySide6 backends. It also support PyQt5 backend but it is not fully tested and may not work as expected.
Since it is a good practice to not mess up the system-wide python environment, you should install this program in a virtual environment. If you don't have a virtual environment yet, you can create one with the command

```python -m venv env_name```

For example, to create a virtual environment called "astro", you can use the command

```python -m venv astro```

and you can activate it with

```. astro/bin/activate```


## From this GIT repository
To install the bleeding edge version, first clone this repository
 
```
git clone https://github.com/mauritiusdadd/redmost.git
cd redmost
```

and then run pip specifying which Qt backend you want to use:

- for PyQt6: ```pip install .[pyqt6]```
- for PySide6: ```pip install .[pyside6]```
- for PySide5: ```pip install .[pyqt5]``` (only for compatibility, may not work as expected)

## From PyPi

just use pip, like for any other packages. Keep in mind that packaged releases may be older than the git version and could lack of some newly implemented functionalities.

- for PyQt6: ```pip install redmost[pyqt6]```
- for PySide6: ```pip install redmost[pyside6]```
- for PyQt5: ```pip install redmost[pyqt5]```

After the installation, to update redmost to the most recent release, use

```pip install redmost --upgrade```

## Install third party backends

Redmost can use modular backends to measure the redshift, although only redrock is currently supported. Please check and follow the installation instructions of the single packages!

- redrock backend: [https://github.com/desihub/redrock][1]

# Run

To run the program just run the command ```redmost``` in a terminal. If you have both PyQt and Pyside installed, you can force the program to use a specific backend using the environment variable ``QT_API``, for example:

```QT_API="pyside6" redmost```

# Docs and tutorials

The full documentation is available at: https://redmost.readthedocs.io/en/latest

# Acknowledgements

If you use this software for your work, please consider to include a citation to [10.5281/zenodo.10817884][4].

Also remember to acknowledge:

- astropy: [https://www.astropy.org/acknowledging.html][3]
- specutils: [https://github.com/astropy/specutils/blob/main/specutils/CITATION][2]
- redrock: [https://github.com/desihub/redrock][1] (if you use the redrock backend)

This program uses icons derived from the following themes:

- feather: [https://github.com/feathericons/feather][5]

[1]: https://github.com/desihub/redrock
[2]: https://www.astropy.org/acknowledging.html
[3]: https://github.com/astropy/specutils/blob/main/specutils/CITATION
[4]: https://zenodo.org/records/10818017
[5]: https://github.com/feathericons/feather
