Visualization
=============

The :mod:`~m3sh.vis` module provides wrapper functions and classes around
low-level VTK functionality. It can also be used as a stand-alone OBJ
viewer:

>>> python vis.py file.obj --edges

opens a graphics window and displays the contents of `file.obj`. Omitting
the '--edges' argument will not render mesh edges.

.. currentmodule:: m3sh

.. autosummary::
   :toctree: api
   :template: module-toc-vtk.rst

   vis


.. note::

   This module requires a recent VTK installation (version 9.1 or higher is
   recommended).