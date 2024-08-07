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

   Support for mesh item flags as defined in :mod:`~m3sh.flags` is still
   experimental and subject to change.
   
   
Special methods
---------------

The mesh class and many mesh items provide :term:`special method`
implementations, so called double underscore methods or dunder 
methods for short.


Mesh
++++

.. autosummary::
   :toctree: api
   :template: function.rst
   
   ~m3sh.hds.Mesh.__iter__
   ~m3sh.hds.Mesh.__copy__
   ~m3sh.hds.Mesh.__deepcopy__
   
.. note::

   The iterator returned by :meth:`~m3sh.hds.Mesh.__iter__` 
   intentionally hides deleted faces of a mesh.
      

Vertex
++++++

The vertex class provides convenient ways for alternative access
to the vertex :attr:`~m3sh.hds.Vertex.index` attribute.

.. autosummary::
   :toctree: api
   :template: function.rst
    
   ~m3sh.hds.Vertex.__index__
   ~m3sh.hds.Vertex.__int__
   
   
Halfedge
++++++++

.. autosummary::
   :toctree: api
   :template: function.rst
    
   ~m3sh.hds.Halfedge.__iter__
   ~m3sh.hds.Halfedge.__getitem__
   ~m3sh.hds.Halfedge.__contains__
   
   
Face
++++

Just like vertices, faces provide alternative access to their
:attr:`~m3sh.hds.Face.index` attribute.

.. autosummary::
   :toctree: api
   :template: function.rst
    
   ~m3sh.hds.Face.__index__
   ~m3sh.hds.Face.__int__
   ~m3sh.hds.Face.__len__
   ~m3sh.hds.Face.__iter__
   ~m3sh.hds.Face.__getitem__
   ~m3sh.hds.Face.__contains__

The vertices of a face can be visited in several ways. While using
:meth:`~m3sh.hds.Face.__len__` and :meth:`~m3sh.hds.Face.__getitem__`

.. code-block:: python

   for i in range(len(f)):
       print(f[i])
       
is equivalent to using :meth:`~m3sh.hds.Face.__iter__`

.. code-block:: python

   for v in f:
       print(v)
       
the latter is much more efficient and preferred.

.. note::

   Vertices of a face are always visited in positive orientation, i.e.,
   counter-clockwise.
