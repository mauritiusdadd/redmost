

# REDMOST

<b>RED</b>shift <b>M</b>easurement <b>O</b>f <b>S</b>pec<b>T</b>ra is a Qt6 Graphical User Interface (GUI) to do spectroscopic redshift measurements for 1D spectra.

If [redrock][1] is correcly installed, it cat be used as a backend to measure the redshift.

# Installation

This is a python program based on Qt6 and supports both PyQt6 and PySide6 backends.
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

and then run pip specifyng which Qt backend you want to use:

- for PyQt6: ```pip install .[pyqt6]```
- for PySide6: ```pip install .[pyside6]```

## Install third party backends

Redmost can use modular backends to measure the redshift, although only redrock is currently supported. Please check and follow the installation instructions of the single packages!

- redrock backend: [https://github.com/desihub/redrock][1]

# Run

To run the program just run the command ```redmost``` in a terminal

# Aknowledgemnts

If you use this software for your work, please consider to include a citation to 10.5281/zenodo.10817884
Also remember to aknowledge:

- astropy: [https://www.astropy.org/acknowledging.html][3]
- specutils: [https://github.com/astropy/specutils/blob/main/specutils/CITATION][2]
- redrock: [https://github.com/desihub/redrock][1] (if you use the redrock backend)


[1]: https://github.com/desihub/redrock
[2]: https://www.astropy.org/acknowledging.html
[3]: https://github.com/astropy/specutils/blob/main/specutils/CITATION
