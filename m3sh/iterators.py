# Copyright 2024, m3shware
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

""" Combinatorial mesh item neighborhood iterators.

Adjacent/incident mesh items are visited in counter-clockwise order as
determined by the mesh orientation (whenever it makes sense to consider
oriented item traversal).

Note
----
When applied to a :class:`~m3sh.hds.Mesh` instance, the iterators
:func:`verts` and :func:`faces` will **skip** deleted mesh items.
This is an alternative to iteration over the mesh item
containers :attr:`~m3sh.hds.Mesh.vertices` and :attr:`~m3sh.hds.Mesh.faces`.
"""

from collections import deque

import numpy as np

from m3sh.hds import Face
from m3sh.heap import MinHeap


def verts(obj):
    """ Vertex iterator.

    The returned iterator traverses adjacent/incident vertices
    of `obj` depending on its type:

    .. table::
       :width: 100%
       :widths: 20, 80

       =============== ================================================
       :class:`Vertex` \u21ba traversal of adjacent vertices
       --------------- ------------------------------------------------
       :class:`Face`   \u21ba traversal of incident vertices
       --------------- ------------------------------------------------
       :class:`Mesh`   in-order traversal of :attr:`~Mesh.vertices`
       =============== ================================================

    Parameters
    ----------
    obj : Vertex or Face or Mesh
        The base object.

    Yields
    ------
    Vertex

    Note
    ----
    When applied to a mesh, this iterator will skip any deleted vertices
    present in its vertex list.
    """
    return obj._viter()


def _verts_bfs(item, stop=None, start=0):
    """ Breadth-first vertex neighborhood iterator.

    Breadth-first traversal of vertex neighborhood of a mesh item.
    Incident vertices are considered neighbors at distance zero.

    Parameters
    ----------
    item : Vertex or Halfedge or Face
        The seed item.
    stop : int, optional
        All vertices at edge distance less or equal to `stop` are visited.
    start : int, optional
        Vertex reporting starts at the given distance level.

    Yields
    ------
    Vertex
        Next vertex in breadth-first search.
    int
        Distance to `item`.
    """
    # The try block could be replaced by explicit type checking using
    # isinstance.
    try:
        seeds = [v for v in item]           # face and edge are iterable
    except TypeError:
        seeds = [item]                      # vertex case, single seed

    queue = deque(seeds)
    level = dict.fromkeys(seeds, 0)

    while queue:
        v = queue.popleft()
        d = level[v]

        # Stop when all vertices at distance stop (i.e., number of
        # edges traversed) have been found.
        if stop is not None and d > stop:
            return

        if start <= d:
            yield v, d

        for w in v._viter():
            # Vertices with assigned level information are either
            # in the queue right now or have been removed earlier.
            if w not in level:
                queue.append(w)
                level[w] = d + 1
            else:
                assert level[w] <= d + 1


def _verts_dij(item, stop=None, start=0.0):
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
        Distance to `item`.

    Note
    ----
    Distance is measured as the length of the shortest edge path that
    connects two vertices.
    """
    # The dictionary of predecessors is generated but not used.
    # prev = dict()

    # The try block could be replaced by explicit type checking using
    # isinstance.
    try:
        seeds = [v for v in item]           # face and edge are iterable
    except TypeError:
        seeds = [item]                      # vertex case, single seed

    # Seed the priority queue with all source vertices. Priorities are
    # distance values. Smaller distance means higher priority.
    queue = MinHeap((seed, 0.0) for seed in seeds)
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

    The returned iterator traverses incident halfedges of `obj`
    depending on its type:

    .. table::
       :width: 100%
       :widths: 20, 80

       =============== ================================================
       :class:`Vertex` \u21ba traversal of outward pointing halfedges
       --------------- ------------------------------------------------
       :class:`Face`   \u21ba traversal of incident halfedges
       --------------- ------------------------------------------------
       :class:`Mesh`   traversal of :attr:`~Mesh.halfedges`
       =============== ================================================

    Parameters
    ----------
    obj : Vertex or Face or Mesh
        The base object.

    Yields
    ------
    Halfedge

    Note
    ----
    This iterator is equivalent to ``iter(mesh.halfedges.values())``
    when applied to a mesh.
    """
    return obj._hiter()


def _halfs_frozen(obj):
    """ Halfedge iterator.
    """
    return iter(list(obj._hiter()))


def edges(mesh):
    """ Edge iterator.

    Formally, an undirected edge is defined as a pair of oppositely
    oriented halfedges. To visit the edges of a mesh, this iterator
    yields exactly one of the two halfedge representatives of an edge.

    Parameters
    ----------
    mesh : Mesh
        Mesh instance.

    Yields
    ------
    Halfedge
    """
    return mesh._eiter()


def _edges_frozen(mesh):
    """ Edge iterator.
    """
    return iter(list(mesh._eiter()))


def faces(obj):
    """ Face iterator.

    A vertex :math:`v` and a face :math:`f` are incident if
    :math:`v \in f`. Two faces are incident if they share a common edge.
    The returned iterator visits the incident faces of `obj` depending
    on its type:

    .. table::
       :width: 100%
       :widths: 20, 80

       =============== ================================================
       :class:`Vertex` \u21ba traversal of incident faces
       --------------- ------------------------------------------------
       :class:`Face`   \u21ba traversal of incident faces
       --------------- ------------------------------------------------
       :class:`Mesh`   in-order traversal of :attr:`~Mesh.faces`
       =============== ================================================

    Parameters
    ----------
    obj : Vertex or Face or Mesh
        The base object.

    Yields
    ------
    Face

    Note
    ----
    When applied to a mesh, this iterator will skip any deleted faces
    still present in its face list.
    """
    return obj._fiter()


def _faces_bfs(item, stop=None, start=0):
    """ Breadth-first face neighborhood iterator.

    Breadth-first traversal of the face neighborhood of a mesh
    item. The 0-ring face neighborhood of a vertex is empty.

    Parameters
    ----------
    item : Vertex or Halfedge or Face
        The seed item.
    stop : int, optional
        All faces at distance less or equal to `stop` are visited.
    start : int, optional
        Face reporting starts at the given distance level.

    Yields
    ------
    Face
        The next face in breadth-first search.
    int
        Distance to `item`.
    """
    # The try block could be replaced by explicit type checking using
    # isinstance.
    try:
        seeds = [v for v in item]           # item is face or edge
        level = dict.fromkeys(seeds, 0)
        level[item] = 0
    except TypeError:                       # item is a vertex
        seeds = [item]
        level = dict.fromkeys(seeds, 0)
    else:
        if start == 0 and isinstance(item, Face):
            yield item

    queue = deque(seeds)

    while queue:
        v = queue.popleft()

        # Once a stop level vertex is visited in a breadth-first
        # traversal, all faces of lower levels are exhausted.
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


def _faces_lnk(item):
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
    """
    return item._fiter_lnk()


def _faces_frozen(obj):
    """ Face iterator.
    """
    return iter(list(obj._fiter()))
