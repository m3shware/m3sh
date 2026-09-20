# Copyright 2024-26, m3shware developers
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

r""" Combinatorial mesh iterators.

The halfedge data structure facilitates efficient neighborhood traversal
on a mesh. This module provides generic implementations of the most common
traversal schemes using Python's iterator protocol. Whenever possible and
meanigful, adjacent/incident mesh items are visited in counter-clockwise
order as determined by the mesh orientation

Example
-------
To compute the average of the 1-ring neighbors for each vertex of a mesh
one can do

.. code-block::

    import itertools as it

    for v in it.verts(mesh):
        avg = sum(w.point for w in it.verts(v))

An equivalent implementation using the interface provided by the halfedge
data structure would be

.. code-block::

    for v in mesh.vertices:
        h = v.halfedge
        avg = 0.0

        while True:
            avg += h.target.point
            h = h.prev.pair

            if h is v.halfedge:
                break

Relation to simplical complexes
-------------------------------
Some basics forms of the provided iterators model notions from algebraic
topology, see [1]_. A triangle mesh is a simplical complex ...

The star operator
~~~~~~~~~~~~~~~~~
Let :math:`K` be a simplicial complex. The (open) star of a simplex
:math:`\sigma \in K` is defined as

.. math::

   \operatorname{st}(\sigma) = \{ \tau \in K | \sigma \subset \tau \}

...

The link operator
~~~~~~~~~~~~~~~~~
The link is defined as the boundary of the star

.. math::

   \operatorname{lnk}(\sigma) = \overline{\operatorname{st}(\sigma)}
        \setminus \operatorname{st}(\sigma)

...

References
----------
.. [1] Allen Hatcher: *Algebraic Topology*, 2001.
"""

from collections import deque

import numpy as np

from m3sh.hds import Face
from m3sh.heap import Heap


def verts(obj):
    """ Vertex iterator.

    Iterator visiting the adjacent/incident vertices of `obj`. For a
    vertex this results in a counter-clockwise traversal of adjacent
    vertices (vertices connected by an edge). For a face, the incident
    vertices are visited in the order determined by the mesh orientation.

    Parameters
    ----------
    obj : Vertex or Face or Mesh
        The base object.

    Yields
    ------
    Vertex
        Next vertex in a traversal of adjacent/incident vertices.
        When applied to a mesh, this iterator will skip any deleted
        vertices present in its vertex list

    Examples
    --------
    The iterator skips deleted mesh items. Hence, applying it to a mesh

    .. code-block::

        for v in verts(mesh):
            ...

    is equivalent to

    .. code-block::

        for v in mesh.vertices:
            if not v.deleted:
                ...
    """
    # Does not apply to halfedges, they do not provide a _viter() method.
    return obj._viter()


def verts_bfs(item, stop=None, start=1):
    """ Breadth-first vertex neighborhood iterator.

    Breadth-first traversal of vertex neighborhood of a mesh item. Incident
    vertices of a halfedge or face are considered neighbors at distance zero.

    Parameters
    ----------
    item : Vertex or Halfedge or Face
        The seed item. Halfedges and faces are treated as vertex containers,
        their incident vertices define seed vertices.
    stop : int, optional
        All vertices at edge distance less or equal to `stop` are visited.
        If :obj:`None`, the search will continue until all vertices of a
        connected component are visited.
    start : int, optional
        Vertex reporting starts at the given distance level. Seed vertices
        are not reported by default.

    Yields
    ------
    Vertex
        Next vertex in breadth-first search.
    int
        Distance to `item` measured as the number of traversed edges.
    """
    # Initialization only makes sense for the listed input types. Meshes
    # are iterable but this iterator yields faces, not vertices! But note
    # that a list of vertices would be a valid input!
    try:
        seeds = [v for v in item]
    except TypeError:
        seeds = [item]

    queue = deque(seeds)
    level = dict.fromkeys(seeds, 0)

    while queue:
        v = queue.popleft()
        d = level[v]

        # Stop when all vertices at distance stop (i.e., number of edges
        # traversed) have been found.
        if stop is not None and d > stop:
            return

        if start <= d:
            yield v, d

        for w in v._viter():
            # Vertices with assigned level information are either in the
            # queue right now or have been removed earlier.
            if w not in level:
                queue.append(w)
                level[w] = d + 1
            # else:
            #     assert level[w] <= d + 1


def verts_dij(item, stop=None, start=0.0):
    """ Dijkstra based vertex neighborhood iterator.

    Visit the vertex neighborhood of a mesh item in a Dijkstra like
    fashion (distance based breadth-first search).

    Parameters
    ----------
    item : Vertex or Halfedge or Face
        The seed item.
    stop : float, optional
        All vertices at distance less or equal to `stop` are visited.
    start : float, optional
        Vertex reporting starts at the given distance.

    Yields
    ------
    Vertex
        Next vertex according to distance.
    float
        Distance to `item`, i.e., the length of the shortest edge path
        that connects the returned vertex to the seed item.
    """
    # The dictionary of predecessors is generated but not used.
    # prev = dict()

    # Initialization only makes sense for the listed input types. Meshes
    # are iterable but this iterator yields faces, not vertices! But note
    # that a list of vertices would be a valid input!
    try:
        seeds = [v for v in item]
    except TypeError:
        seeds = [item]

    # Seed the priority queue with all source vertices. Priorities are
    # distance values. Smaller distance means higher priority.
    queue = Heap((seed, 0.0) for seed in seeds)
    dist = dict.fromkeys(seeds, 0.0)

    # Initialize predecessor values for all seed vertices. This replaces
    # corresponding dictionary entries if already present.
    # prev.update((v, None) for v in seeds)

    while queue:
        v, d = queue.pop()

        # Stop once a vertex with larger distance value than eps is
        # popped. All remaining vertices have larger distance values.
        if stop is not None and d > stop:
            return

        if start <= d:
            yield v, d

        # Update distance value of v's neighbors w if the path via v
        # and the edge vw is shorter than the current shortest path.
        for w in v._viter():
            edge_len = np.linalg.norm(w.point - v.point)

            if d + edge_len < dist.get(w, np.inf):
                dist[w] = d + edge_len
                # prev[w] = v
                queue.push(w, d + edge_len)


def _verts_frozen(obj):
    """ Vertex iterator.
    """
    return iter(list(obj._viter()))


def halfs(obj):
    """ Halfedge iterator

    Traverse the incident halfedges of `obj`. For vertex input the
    halfedges with `obj` as origin are visited in a counter-clockwise
    traversal. For a face, the face defining loop of halfedges is
    traversed in counter-clockwise order.

    Parameters
    ----------
    obj : Vertex or Face or Mesh
        The base object.

    Yields
    ------
    Halfedge
        Next halfedge in a traversal of incident halfedges. For mesh
        input those halfedges are in no particular order.

    Examples
    --------
    When applied to a mesh

    .. code-block::

        import itertools as it

        for h in it.halfs(mesh)
            ...

    is equivalent to

    .. code-block::

        for h in mesh.halfedges.values():
            ...
    """
    return obj._hiter()


def _halfs_frozen(obj):
    """ Halfedge iterator.
    """
    return iter(list(obj._hiter()))


def edges(mesh):
    """ Edge iterator.

    An undirected edge is a pair of oppositely oriented halfedges. This
    iterator visits exactly one of the two halfedge representatives of
    an edge.

    Parameters
    ----------
    mesh : Mesh
        Mesh instance.

    Yields
    ------
    Halfedge
        Next representative of an edge.
    """
    return mesh._eiter()


def _edges_frozen(mesh):
    """ Edge iterator.
    """
    return iter(list(mesh._eiter()))


def faces(obj):
    """ Face iterator.

    Iterator visiting the adjacent/incident faces of `obj`. Faces are
    adjacent if they share a common edge.

    Parameters
    ----------
    obj : Vertex or Face or Mesh
        The base object.

    Yields
    ------
    Face
        Next face in a traversal of all adjacent/incident faces. When
        applied to a mesh, this iterator skips any deleted faces still
        present in its face list.

    Examples
    --------
    For vertex input all incident faces are visited in counter-clockwise
    order. For non-boundary vertices this is equivalent to

    .. code-block::

        h = v.halfedge

        while True:
            f = h.face
            h = h.prev.pair

            if h is v.halfedge:
                break
    """
    # Does not apply to halfedges, they do not provide a _fiter() method.
    return obj._fiter()


def faces_bfs(item, stop=None, start=1):
    """ Breadth-first face neighborhood iterator.

    Breadth-first traversal of the face neighborhood of a mesh item. A
    face and a mesh item (a vertex, a halfedge, or another face) are
    considered neighbors is they share a vertex or an edge.

    Parameters
    ----------
    item : Vertex or Halfedge or Face
        The seed item.
    stop : int, optional
        All faces at distance less or equal to `stop` are visited.
        If :obj:`None`, the search will continue until all faces of a
        connected component are visited.
    start : int, optional
        Face reporting starts at the given distance level. Seed faces
        are not reported by default.

    Yields
    ------
    Face
        The next face in breadth-first search.
    int
        Distance to `item`.

    Notes
    -----
    The neighborhood relation used in breadth-first search is different
    for the one used by :func:`faces` where faces are considered neighbors
    if they share an edge. For instance, the faces at distance 1 to a seed
    face are all faces that share an edge or a vertex with the seed. This
    can be imagined as a ring of faces around the seed face. Consequently,
    the list

    >>> [f for f in faces(seed)]

    is different from

    >>> [f for f in faces_bfs(seed, stop=1)]

    Use :func:`fdual_bfs` to restrict the neighborhood relation to having
    a common edge.
    """
    # Initialization only makes sense for the listed input types. Meshes
    # are iterable but this iterator yields faces, not vertices!
    try:
        # The seeds container always holds vertices. In constrast, level
        # assigns a level to visited vertices and faces!
        seeds = [v for v in item]
        level = dict.fromkeys(seeds, 0)

        # Edges and faces can be seen as vertex containers. It would be
        # fine to use an explicit list of vertices as input. But note
        # that using a list of the three vertices of a face will make
        # this face a level 1 face, not level 0. The level of all other
        # faces does not change.
        if isinstance(item, Face):
            # This is the only way in which a face can be of level zero.
            # In general, the following rule applies. Visited a level d
            # vertex will assign level d + 1 to all incident faces that
            # have not yet been visited.
            level[item] = 0

            # Also, if reporting starts a level zero, yield the initial
            # face .
            if start == 0:
                yield item
    except TypeError:
        seeds = [item]
        level = dict.fromkeys(seeds, 0)

    queue = deque(seeds)

    while queue:
        v = queue.popleft()

        # Once a stop level vertex is visited in a breadth-first traversal,
        # all faces of lower levels are exhausted.
        if stop is not None and level[v] >= stop:
            return

        for w in v._viter():
            if w not in level:
                queue.append(w)
                level[w] = level[v] + 1

        for f in v._fiter():
            if f not in level:
                level[f] = level[v] + 1

                if start <= level[v]:
                    yield f, level[f]


def fdual_bfs(*seeds, stop=None, start=1):
    """ Dual breadth-first face neighborhood iterator.

    Breadth-first search using dual mesh combinatorics.

    Parameters
    ----------
    *seeds
        At least one but otherwise arbitrary number of faces.
    stop : int, optional
        All faces at dual edge distance less or equal to `stop` are visited.
        If :obj:`None`, the search continues until all faces of a connected
        component are visited.
    start : int, optional
        Face reporting starts at the given distance level. Seed faces are
        not reported by default.

    Yields
    ------
    Face
        The next face in a breadth-first search.
    int
        Distance
    """
    queue = deque(seeds)
    level = dict.fromkeys(seeds, 0)

    while queue:
        f = queue.popleft()
        d = level[f]

        if stop is not None and d > stop:
            return

        if start <= d:
            yield f, d

        for g in f._fiter():
            if g not in level:
                queue.append(g)
                level[g] = d + 1


def faces_lnk(item):
    """ Face iterator.

    Counter-clockwise traversal of all faces that share an edge or a
    vertex with `item`.

    Parameters
    ----------
    item : Halfedge or Face
        The base item.

    Yields
    ------
    Face
        Next face in a counter-clockwise traversal of faces.

    Notes
    -----
    For a given seed face the lists

    >>> [f for f in faces_lnk(seed)]

    and

    >>> [f for f in faces_bfs(seed, stop=1)]

    are the same.
    """
    return item._fiter_lnk()


def _faces_frozen(obj):
    """ Face iterator.
    """
    return iter(list(obj._fiter()))
