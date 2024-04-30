.. _tutorial_2:

Tutorial 2 - Redshift measurements
==================================

What do you need for this tutorial
----------------------------------

    .. _item_a:

a) A working installation of redmost (see :ref:`Installation Guide<installation_guide>`)

    .. _item_b:

b) Test files: `https://github.com/mauritiusdadd/redmost/tree/main/test/data <https://github.com/mauritiusdadd/redmost/tree/main/test/data>`_


Manually with the lines tool
----------------------------

First of all, load the test spectra (:ref:`b <item_b>`), as described in
:ref:`Tutorial 1<tutorial_1>`, and select the one named  "spec_95.fits". This
is the spectrum of a passive galaxy with clear and easily identifiable CaII H&K
lines. Let's first zoom-in on these lines, then move to the "Lines" tab in the
"Redshift tools" panel (:numref:`fig_main_ui` [9]) and then click the button
"Add line". Now most of the controls are disabled and the mouse cursor has
changed to indicate that the application is waiting for you to select where the
line is: just move the mouse cursor on the peak of the absorption of the
CaII K, like in :numref:`fig_main_ui_add_line`, and then click with the left
mouse button. A new entry has been added to the "Identification" list with the
wavelength we just picked up with the mouse. We can now associate to this
entry the restframe wavelength of the CaII K by first selecting it from the
list with the mouse and then selecting "CaII_K - 3933.70A" from the combo-box
"Set current line to", like shown in :numref:`fig_main_ui_identification`.
This operation automatically compute the redshift and updates the restframe
wavelength for other identified lines. The redshift value appears in the top
panel of "Redshift Tools" and can also be changed manually.

A quality flag for the redshift estimation can be set using the "quality flag"
combo-box. Setting a quality flag also changes the color of the corresponding
element in the open spectra list (:numref:`fig_main_ui` [3]), in this way it is
easy to distinguish the objects for which you already measured the redshift
from those you have not looked at yet.

.. _fig_main_ui_add_line:
.. figure:: ../pics/main_ui_add_line.png
   :figwidth: 100 %
   :alt: Main window showing how to add a line.

   The main window when a spectrum is selected from the open spectra list.

.. _fig_main_ui_identification:
.. figure:: ../pics/main_ui_identification.png
   :figwidth: 50 %
   :alt: Main window - identification tool.

   Identification tool.

Multi-lines matching
--------------------

If you are able to identify more that one line, but you are not sure about what
line they are, you can try to use the multi-lines matching tool. Just click
on the button "Match selected" and a list of possible redshift will appear,
like in :numref:`fig_main_ui_multi_lines`. Just click on one of the
redshifts to set it as the redshift value for the object.


.. _fig_main_ui_multi_lines:
.. figure:: ../pics/main_ui_multi_lines.png
   :figwidth: 50 %
   :alt: Main window - multi-lines matching tool.

   Three lines (CaII H&K and NaD) have been matched using the multi-lines
   matching tools and a list of possible redshift has been produced.
   A pseudo-probability (p) is associated to each redshift, where p=1.0
   indicates a perfect match.

Using Redrock backend
---------------------

.. |play_ico| image:: ../pics/icons/play.svg

If you have redrock configured and installed, then you can move to the tab
"Redrock" in the "Redshift tools" (:numref:`fig_main_ui` [9]). In the panel
"Options" you can specify whether to measure the redshift for all the loaded
spectra (default), only for the selected spectra, or for the current object
only. Once you have set the desired working mode, just clik on the
button "|play_ico| Run" to run redrock. The output of redrock will be printed
out in the "Status" text-box, so you can check what is going on.
Once redrock has finished, redmost will automatically collect the best redshift
estimations for each objects.

.. _fig_main_ui_redrock:
.. figure:: ../pics/main_ui_redrock.png
   :figwidth: 50 %
   :alt: Main window - redrock backend.

   The redrock backend interface.