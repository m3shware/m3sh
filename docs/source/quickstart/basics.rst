.. _mesh-rep-label:

Mesh basics
===========

.. image:: ../_static/dragon_full.png

|

We define a **polygonal mesh** by specifying two sequences: a sequence
:math:`V = (\mathbf{v}_i)_{i=0}^{n-1}` of geometric vertices such that
:math:`\mathbf{v}_i \in \mathbb{R}^3` and a sequence 
:math:`F = (f_j)_{j=0}^{m-1}` of combinatorial face definitions.

Faces :math:`f \in F` are sequences themselves. A :math:`k`-tuple
:math:`f = (i_0, \dots, i_{k-1})` of integers defines a face :math:`|f|` of
valence
:math:`k` with vertices :math:`\mathbf{v}_{i_0}, \dots, \mathbf{v}_{i_{k-1}}`.
Geometric faces :math:`|f|` are not required to be planar or convex. 
If :math:`k` is equal to three for all faces, the pair :math:`(V, F)` defines
a **triangle mesh**.

.. note::

   We can identify the ordered sequence of vertices with a matrix
   :math:`V \in \mathbb{R}^{n \times 3}`. Replacing :math:`V` with
   :math:`V' \in  \mathbb{R}^{n \times 3}` yields a mesh with identical
   **combinatorics** but different vertex positions.


.. _manifold-mesh-label:

Manifold meshes
---------------

A pair :math:`(V,F)` as just defined can describe arbitrary collections of
polygonal faces. For simplicity we only consider triangular faces in the
following definitions. A triangle mesh is **2-manifold** if the faces incident
to a vertex either form a closed or an open triangle fan.

|

.. image:: ../_static//manifold_fan.png
   :width: 70 %
   :align: center
 
|  
   
The **orientation** of a face is defined by the cyclic ordering of its incident
vertices as specified in its combinatorial definition. The orientation
of two adjacent faces is **compatible**, if the two vertices of the common edge
appear in opposite order. A manifold mesh is **orientable** if any two adjacent
faces have compatible orientation.

.. note::

   Both, manifoldness and orientability are determind by the mesh
   combinatorics :math:`F` and do **not** depend on a concrete geometric
   realization (an embedding defined by specifying :math:`V`).


.. _halfedge-label:

Halfedge representation
-----------------------

Any orientable 2-manifold mesh can be represented using halfedges [1]_ [2]_. 
Conceptually one splits each edge of a mesh into two so called halfedges. 
Each halfedge is oriented according to the orientation of its incident face. 
In this way adjacent faces give rise to oppositely oriented halfedges:

|

.. image:: ../_static//halfedge_all.png
   :width: 90 %
   :align: center

|

The :class:`~m3sh.hds.Mesh` class provides a generic halfedge data structure
for orientable 2-manifold meshes. The combinatorics of a mesh is defined via
its halfedges and their attributes. Each halfedge is aware of its incident
:attr:`~m3sh.hds.Halfedge.face`, its :attr:`~m3sh.hds.Halfedge.origin` and
:attr:`~m3sh.hds.Halfedge.target` vertex, the neighboring halfedge
:attr:`~m3sh.hds.Halfedge.pair`, as well as its successor
:attr:`~m3sh.hds.Halfedge.next` and predecessor halfedge
:attr:`~m3sh.hds.Halfedge.prev` in a face defining loop of halfedges.

.. note::

   The explicit representation of a face as a list of its vertices can
   be reconstructed from the set of halfedges. It is sufficient to know 
   one halfedge per face to compute the face defining loop of halfedges 
   (or vertices).


References
----------
.. [1] K. Crane: "A Survey of Efficient Structures for Digital Geometry
       Processing", 2006. 

.. [2] H. Brönnimann: "Designing and Implementing a General Purpose Halfedge
       Data Structure", Proceedings of the 5th International Workshop on 
       Algorithm Engineering, 2001.


