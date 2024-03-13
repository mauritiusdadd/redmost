
# pyzui

A Qt6 GUI interface to do spectroscopic redshift measurements.
If redrock is correcly installed, it cat be used as a backend to measure the redshift.

# Installation

This is a python program based on Qt6 and supports both PyQt6 and PySide6 backends.
Since it is a good practice to not mess up the system-wide python environment, you should install this program in a virtual environment. If you don't have a virtual environment yet, you can create one with the command

```python -m venv env_name```

For example, to create a virtual environment called "astro", you can use the command

```python -m venv astro```

and you can activate it with

```. astro/bin/activate```


### From this GIT repository
To install the bleeding edge version, first clone this repository
 
```
git clone https://github.com/mauritiusdadd/pyzui.git
cd pyzui
```

and then run pip specifyng which Qt backend you want to use:

- for PyQt6: ```pip install .[pyqt6]```
- for PySide6: ```pip install .[pyside6]```

# Run

To run the program just run the command ```pyzui``` in a terminal
