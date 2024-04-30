.. _tutorial_1:

Tutorial 1 - Basic operations
=============================

What do you need for this tutorial
----------------------------------

    .. _item_a:

a) A working installation of redmost (see :ref:`Installation Guide<installation_guide>`)

    .. _item_b:

b) Test files: `https://github.com/mauritiusdadd/redmost/tree/main/test/data <https://github.com/mauritiusdadd/redmost/tree/main/test/data>`_

Running Redmost
---------------

As described in the installation guide, you can run redmost by executing it in
a terminal ``redmost`` and the main window should appear
like in :numref:`fig_main_ui`. Since no spectra have been loaded yet, most of
the controls are disabled, so the first step is to load at least one spectrum.

.. _fig_main_ui:
.. figure:: ../pics/main_ui_annotated.png
   :figwidth: 100 %
   :alt: Redmost main window.

   This is the main window of Redmost. The labels indicate:
   the menu bar [1]; the toolbar [2]; the open spectra list widget [3];
   the current spectrum header info [4]; the current spectrum flux plot
   widget [5]; the current spectrum variance, wavelegth dispersion, and sky
   plots widgets [6]; The plot options tools [7]; The current spectrum
   redshift value and quality flag selector [8]; The available redshift
   measurement tools [9].

Load spectra
------------

.. |import_spectra_ico| image:: ../pics/icons/file-plus.svg

Spectra can be loaded using the button |import_spectra_ico| on the tool bar
(:numref:`fig_main_ui` [2]) or the menù "File" > "Import spectra"
(:numref:`fig_menu_file`). A dialog will appear where you can select one or
multiple files to load. For this tutorial we will use the test spectra that are
in "test/data" folder of the github repository, so make sure to download all
the files from (:ref:`b <item_b>`) into a dedicated folder. Then, in the
"Import Spectra" dialog select all the FITS files whose names begin with
"spec" and click on the "Open" button (:numref:`fig_dialog_load_spectra`).

.. _fig_menu_file:
.. figure:: ../pics/menu_file.png
   :figwidth: 50 %
   :alt: File menù

   The "File" menù.

.. _fig_dialog_load_spectra:
.. figure:: ../pics/dialog_load_spectra.png
   :figwidth: 100 %
   :alt: Open file dialog

   Dialog windows used to select the files to load.

Spectrum plot widgets
---------------------

.. |zoom_in_ico| image:: ../pics/icons/zoom-in.svg
.. |zoom_out_ico| image:: ../pics/icons/zoom-out.svg
.. |zoom_fit_ico| image:: ../pics/icons/maximize.svg

The open spectra list widget (:numref:`fig_main_ui` [2]) should be now
populated with the files opened in the previous step. If you clik on one of
the element of this list, the corresponding spectrum is shown in the flux plot
widgets (:numref:`fig_main_ui` [5]). If variance, wavelength dispersion or sky
spectra are available, they will be shown in the corresponding plot widgets
(:numref:`fig_main_ui` [6]). In the "Object info" panel, the header of the
primary fits extension is also shown. Smoothing can be enabled using the
corresponding check-box in the "Plot Options" panel
(:numref:`fig_main_ui` [5]), in this case the smoothed spectrum will be plotted
in orange on top of the original spectrum (:numref:`fig_main_ui_spectrum`).
The check-box "Show lines" toggles the visualization of the most common
amission/absorption lines at the redshift of the object and you can also select
which type of line to show using the combo-box on the right.

You can zoom-in on a region of the spectrum by clicking with the left mouse
button and then dragging to select the area you want to zoom in. You can also
control the zoom using the mouse wheel + Ctrl.
Another way to control the zoom is by using the buttons |zoom_in_ico|,
|zoom_out_ico|, and |zoom_fit_ico| to respectively zoom-in, zoom-out and
fit the spectrum in the widget area.

.. _fig_main_ui_spectrum:
.. figure:: ../pics/main_ui_spectrum.png
   :figwidth: 100 %
   :alt: Main window showing a spectrum.

   The main window when a spectrum is selected from the open spectra list.



Save and load projects
----------------------

.. |save_project_ico| image:: ../pics/icons/save.svg
.. |load_project_ico| image:: ../pics/icons/folder.svg

Now that we have loaded a bunch of spectra, we can save the current project
using the button |save_project_ico| on the tool bar (:numref:`fig_main_ui` [2])
or the menù "File" > "Save" (:numref:`fig_menu_file`). To load a previously
saved project, use the button |load_project_ico| on the tool bar or the menù
"File" > "Open Project" (:numref:`fig_menu_file`).