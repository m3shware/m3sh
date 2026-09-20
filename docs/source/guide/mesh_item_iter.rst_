Accessing vertices, faces, and halfedges
========================================

Once a mesh has been constructed, its items (vertices, halfedges, and
faces) can be accessed via properties :attr:`~Mesh.vertices`,
:attr:`~Mesh.halfedges`, and :attr:`~Mesh.faces`. The vertex coordinate
array is exposed via :attr:`~Mesh.points`. Note the distinction between
vertices and points in :math:`\mathbb{R}^3` -- vertices are topological
entities that have a location attribute :attr:`~m3sh.hds.Vertex.point`,
vertex coordinates can change without changing mesh topology.

.. note::

   Elements of the vertex list and rows of the coordinate array correspond
   by index. Elements are stored in insertion order. The same is true for
   the face list.


Modifying vertex coordinates
++++++++++++++++++++++++++++

We want to project the mesh onto the :math:`xy`-plane. There are several
equivalent ways to do this. We can use the vertex list of a mesh and access
vertex coordinates via the :attr:`~m3sh.hds.Vertex.point` property of a
vertex:

.. code-block:: python

    for v in mesh.vertices:                     # mesh.vertices is a list
        v.point[2] = 0.0                        # set the z-coordinate to zero

We can directly access the vertex coordinate array. The correspondence of
vertices to coordinates is given by index but not needed for this task:

.. code-block:: python

    for p in mesh.points:                       # mesh.points is of type ndarray
        p[2] = 0.0                              # set the z-coordinate to zero

Since the vertex coordinate array is of type :class:`~numpy.ndarray` we can
use slicing to set all :math:`z`-coordinates:

.. code-block:: python

    mesh.points[:, 2] = 0.0                     # set all z-coordinates to zero


Computing edge midpoints
++++++++++++++++++++++++

The :attr:`~Mesh.halfedges` dictionary maps pairs of :class:`Vertex` objects to
:class:`Halfedge` objects. The following loop will compute (half)edge midpoints:

.. code-block:: python

    for h in mesh.halfedges.values():
        m = 0.5 * (h.origin.point + h.target.point)


Extracting face definitions
+++++++++++++++++++++++++++

As already explained the face defining sequence of vertices of a face :math:`f`
can be extracted from any incident halfedge of :math:`f` by following its
:attr:`~m3sh.hds.Halfedge.next` attribute:

.. code-block:: python

    v = [f.halfedge.origin.index]
    h = f.halfedge.next

    while h is not f.halfedge:
        v.append(h.origin.index)
        h = h.next

The above loop is equivalent to the following list comprehension:

.. code-block:: python

   v = [x for x in f]

.. note::

   The :attr:`~m3sh.hds.Mesh.vertices` and :attr:`~m3sh.hds.Mesh.faces`
   attributes of a mesh expose deleted mesh items. Applying one of the
   corresponding predefined iterators :func:`~m3sh.iterators.verts` and
   :func:`~m3sh.iterators.faces` to a mesh will skip those items.
