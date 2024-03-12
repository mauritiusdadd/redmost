
# pyzui

A Qt6 GUI interface to do spectroscopic redshift measurements.
If redrock is correcly installed, it cat be used as a backend to measure the redshift.

# Installation

This is a python program based on Qt6 and supports both PyQt6 and PySide6 backends.

### From this GIT repository
 To install the bleeding edge version, first clone this repository
 
```
git clone https://github.com/mauritiusdadd/pyzui.git
cd pyzui
```

and then run pip specifyng which Qt backend you want to use:

- for PyQt6: ```pip install .[pyqt6]```
- for PySide6: ```pip install .[pyside6]```
