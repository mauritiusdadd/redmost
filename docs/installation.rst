.. |github_mark| image:: pics/github-mark.png
   :height: 1em
   :target: github_repo

Installing a Packaged Release
=============================

The simplest way to install Redmost is using ``pip`` and, since it is a good practice to not mess up the system-wide python environment, you should install this program in a virtual environment. If you don't have a virtual environment yet, you can create one with the command

.. code-block:: bash

    python -m venv env_name

For example, to create a virtual environment called "astro", you can use the command

.. code-block:: bash

    python -m venv astro

and you can activate it with

.. code-block:: bash

    source astro/bin/activate

Redmost supports supports both PyQt6 and PySide6 backends, with precedence to PyQt6 if both are installed. PyQt5 backend is also supported for compatibility but it is not fully tested and may not work as expected. You can tell pip to install the appropriate dependencies using the commands:

.. code-block:: bash

    pip install .[pyqt6]

for PyQt6 backend or

.. code-block:: bash

    pip install .[pyside6]

for PySide6 backend, or

.. code-block:: bash

    pip install .[pyqt5]

Installing from GitHub
======================

If you like to use the bleeding-edge version, you can install Specex directly from the |github_mark| `GitHub repository <https://github.com/mauritiusdadd/redmost>`_

.. code-block:: bash

    git clone 'https://github.com/mauritiusdadd/redmost.git'
    cd redmost

and then run pip specifyng which Qt backend you want to use either

.. code-block:: bash

    pip install .[pyqt6]

or

.. code-block:: bash

    pip install .[pyside6]

or

.. code-block:: bash

    pip install .[pyqt5]

Updating
========

After the installation, to update redmost to the most recent release, use

.. code-block:: bash

    pip install redmost --upgrade

Install third party backends
============================

Redmost uses modular backends to measure the redshift, although only redrock is currently supported. Please check and follow the installation instructions of the single packages!

- redrock backend: `<https://github.com/desihub/redrock>`_

Running Redmost
===============

To run the program just run the command ``redmost`` in a terminal. If you have both PyQt and PySide you can force the program to use a particular backend using the environment variable ``QT_API``

.. code-block:: bash

    QT_API="pyside6" redmost

.. _references_installation:

References
----------

#. `Redrock <https://github.com/desihub/redrock>`_
