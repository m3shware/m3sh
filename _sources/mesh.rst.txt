Halfedge data structure
=======================

The :mod:`~m3sh.hds` module provides a general purpose halfedge data 
structure :class:`~m3sh.hds.Mesh` for 2-manifold meshes with polygonal 
faces. The :mod:`~m3sh.iterators` and :mod:`~m3sh.traits` modules build
on the facilities provided by :mod:`~m3sh.hds` and provide convenient
access to both combinatorial and geometric characteristics of a mesh.

.. currentmodule:: m3sh

.. autosummary:: 
   :toctree: api
   :template: module-toc.rst

   hds
   iterators
   traits
   flags

   
.. note::

   The mesh class and many mesh items provide :term:`special method`
   implementations, so called double underscore methods or *dunder* 
   methods for short.
   
