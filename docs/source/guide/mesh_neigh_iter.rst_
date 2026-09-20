Local neighborhood traversal
============================

The following recipe will visit adjacent vertices of a vertex :math:`v` in
counter-clockwise order:

.. code-block:: python

    # Start with the halfedge stored as attribute of a vertex v
    h = v.halfedge

    # Visit the adjacent vertices of v in ccw-order.
    while True:
        # Get the target vertex of h and do something with it.
        w = h.target
        ...
        # Rotate the halfedge counter-clockwise.
        h = h.prev.pair

        # Check if we have reached the starting halfedge again.
        if h is v.halfedge:
            break


The :mod:`~m3sh.iterators` module provides several generic iterators. The most
basic ones being :func:`~m3sh.iterators.verts`, :func:`~m3sh.iterators.halfs`,
:func:`~m3sh.iterators.edges`, and :func:`~m3sh.iterators.faces`. The behavior
of an iterator depends on the type of the provided argument. Using the
:func:`~m3sh.iterators.verts` iterator, the above recipe simplifies to

.. code-block:: python

    import m3sh.iterators as it

    # Visit the adjacent vertices of v in ccw-order.
    for w in it.verts(v):
        # Do something with w ...
        ...


Vertex neighborhood iterators
+++++++++++++++++++++++++++++

When applied to a vertex instance, the :func:`~m3sh.iterators.verts` iterator
can be used to visit the 1-ring neighbors of a vertex in counter-clockwise
order as induced by the mesh orientation:

.. code-block:: python

    import m3sh.iterators as it

    # Visit all vertices of a mesh in the order they were added.
    for v in mesh.vertices:
        print(f'1-ring neighbors of vertex {int(v)}:')
        print('\t', end='')

        # Visit all vertices adjacent to vertex v, i.e., all vertices
        # connected to v via an edge.
        for w in it.verts(v):
            print(int(w), end=' ')

        print()


When applied to a vertex :math:`v`, the :func:`~m3sh.iterators.faces` iterator
will traverse all faces :math:`f` with :math:`v \in f` in counter-clockwise
order:

.. code-block:: python

    import m3sh.iterators as it

    for v in mesh.vertices:
        print(f'Faces incident to vertex {int(v)}:')
        print('\t', end='')

        for f in it.faces(v):
            print(int(f), end=' ')

        print()


Face neighborhood iterators
+++++++++++++++++++++++++++

Two facs are adjacent if they share a common edge. When applied to a face
:math:`f`, the :func:`~m3sh.iterators.face` iterator visits all adjacent
faces in counter-clockwise order:

.. code-block:: python

    import m3sh.iterators as it

    for x in it.faces(f):
        print(x)
