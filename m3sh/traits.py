# Copyright 2022-2026, m3shware
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

""" Geometric mesh traits.

Convenience functions to compute common and often used geometric mesh
traits like vertex normals and face normals.

References
----------
.. [1] S. Jin, R. Lewis, and D. West: *A comparison of algorithms for
       vertex normal computation*, The Visual Computer 21, 2005.
.. [2] M. Meyer et al.: *Discrete differential-geometry operators for
       triangulated 2-manifolds*. In: HC. Hege, K. Polthier (eds)
       *Visualization and Mathematics III*, 2003.

Example
-------
Face normals of a mesh can be computed as

>>> normals = traits.face_normals(mesh)

Since faces can be used as an index, the normal of a face can be accessed
as ``normals[f]`` or more verbosely as ``normals[f.index]``. Instead of
passing around the pair ``(mesh, normals)``, we can add the normals array
to the mesh as an attribute:

>>> mesh.add_face_data('face_normals', 'normal', normals)

In addition to just adding the array as an attribute this also add a
property to each face of the mesh that makes the corresponding face normal
accessible as ``f.normal``. The normals array can still be accessed
directly as ``mesh.face_normals``. In particular

>>> mesh.face_normals is normals
True
"""

import math
import statistics as stats

import numpy as np

import m3sh.linalg as linalg


def axis_angles(mesh, normals=None):
    r""" Rotation parameters.

    Edge based rotation parameters that align neighboring face planes.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh instance with n faces.
    normals : ndarray, shape (n, 3)
        Unit length face normal vectors. If not specified, face normals
        are computed via the :func:`face_normals` function.

    Returns
    -------
    angles : dict
        Dictionary that maps halfedges to rotation parameters that
        rotate the normal of ``halfedge.face`` about the common edge to the
        normal of ``halfedge.pair.face``.

    Notes
    -----
    Rotation parameters are a unit length axis vector :math:`\mathbf{a}`
    together with the values :math:`\cos(\varphi)` and :math:`\sin(\varphi)`.
    In the setting of this function ``a = unit(halfedge.vector)``.
    """
    if normals is None:
        normals = face_normals(mesh)

    axis_angles = dict()

    for h in mesh._eiter():
        axis = h.vector / linalg.norm(h.vector)

        if h.boundary or h.pair.boundary:
            cos_phi, sin_phi = 1.0, 0.0
        else:
            # This rotation aligns the face plane of h.face with the face
            # plane of the neighboring face h.pair.face.
            cos_phi, sin_phi = linalg.rotation(normals[h.face],
                                               normals[h.pair.face], axis)

        axis_angles[h] = (axis, cos_phi, sin_phi)
        axis_angles[h.pair] = (-axis, cos_phi, sin_phi)

    return axis_angles

# This function should not be here! It is not mesh related at all. There
# is an alternative aabb() function in the bounds module -> t3ch package.
def _bounds(points):
    r""" Bounding box vertices.

    Corner vertices of the axis-aligned bounding box.

    Parameters
    ----------
    points : array_like, shape (n, k)
        Coordinates of n points in k dimensions, one point per row.

    Returns
    -------
    a : ndarray
        Holds the minimum value for each dimension.
    b : ndarray
        Holds the maximum value for each dimension.
    """
    return np.min(points, axis=0), np.max(points, axis=0)


def vertex_normal(vertex):
    """ Vertex normal.

    Compute vertex normal as average of triangle normals. Assumes that
    vertex coordinates live in Euclidean 3-space.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a mesh.

    Returns
    -------
    normal : ndarray, shape (3, )
        Unit length normal vector. Deleted and isolated vertices are
        assigned a vector of :obj:`~numpy.nan` values.

    See Also
    --------
    vertex_normals

    Notes
    -----
    For non-triangular meshes, incident triangles are defined by the
    planes spanned by consecutive edges in a counter-clockwise traversal
    of incident edges.
    """
    # Accessing the point property of a deleted vertex does not trigger
    # an assertion error.
    if vertex.deleted:
        return np.full_like(vertex.point, np.nan)

    if vertex.isolated:
        return np.full_like(vertex.point, np.nan)

    normal = np.zeros_like(vertex.point)

    # The vertex is not isolated, hence this iterator cannot be empty.
    for h in vertex._hiter():
        if h.face is not None:
            normal += linalg.unit_inplace(linalg.cross(h.prev.vector,
                                                       h.vector))

    return linalg.unit_inplace(normal)


def vertex_normals(mesh, broadcast=False):
    """ Vertex normals.

    Compute vertex normals as average of face normals.

    Parameters
    ----------
    mesh : Mesh
        A mesh instance.
    broadcast : bool, optional
        Broadcast or gather face normals.

    Returns
    -------
    normals, ndarray, shape (n, 3)
        Unit length normal vectors for a mesh with n vertices.

    See Also
    --------
    vertex_normal

    Notes
    -----
    Vertex normals are not well defined for deleted or isolated vertices.
    The returned array includes rows of :obj:`~numpy.nan` values in this
    case.
    """
    if broadcast:
        # Outer loop runs over faces. Each face broadcasts is normal vector
        # to each incident vertex. This is typically faster since face
        # normals are only computed once.
        normals = np.zeros_like(mesh.points)
        normals[[v.deleted or v.isolated for v in mesh.vertices]] = np.nan

        # This loop only visits faces not marked as deleted. Deleted or
        # isolated vertices are not visited by the inner loop.
        for f in mesh:
            normal = face_normal(f)

            for v in f:
                normals[v] += normal

        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals
    else:
        # Outer loop runs over vertices. Gather normals of incident faces
        # to compute the vertex normal.
        return np.array([vertex_normal(v) for v in mesh.vertices])


def vertex_angle(vertex):
    r""" Total angle around a vertex.

    Compute :math:`\sum_{i=1}^k \alpha_i` where :math:`\alpha_i` denote
    the angles incident to `vertex`.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.

    Returns
    -------
    float
        Sum of incident angles in radians.

    Notes
    -----
    The value :math:`2\pi - \sum_{i=1}^k \alpha_i` is called angle defect
    or discrete Gaussian curvature. A point-wise estimate of Gaussian
    curvature is often defined as the quotient

    >>> K = (2.0 * pi - vertex_angle(v)) / vertex_area_mixed(v)
    """
    angle = 0.0

    for h in vertex._hiter():
        if h.face is not None:
            angle += linalg.angle(h.vector, h.prev.pair.vector)

    return angle


def vertex_area(vertex):
    """ Vertex area.

    One third of the area of all faces incident to `vertex`. The most simple
    choice of vertex area when doing spatial averages.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.

    Returns
    -------
    area : float
        Area assigned to `vertex`.

    See Also
    --------
    vertex_area_voronoi, vertex_area_mixed
    """
    return sum(face_area(f) for f in vertex._fiter()) / 3.0


def vertex_area_mixed(vertex):
    """ Mixed vertex area.

    Mixed area assigned to `vertex` as defined in [1]_.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.

    Returns
    -------
    area : float
        Area assigned to `vertex`.

    References
    ----------
    .. [1] M. Meyer et al.: *Discrete differential-geometry operators for
           triangulated 2-manifolds*. In: HC. Hege, K. Polthier (eds)
           *Visualization and Mathematics III*, 2003.
    """
    cotan = lambda x: 1.0 / math.tan(x)

    a = vertex.point
    area = 0.0
    max = 0.5 * math.pi

    for h in vertex._hiter():
        if h.boundary:
            continue

        b = h.target.point
        c = h.next.target.point

        face_area = 0.5 * linalg.norm(linalg.cross(b - a, c - b))

        angle_a = linalg.angle(b - a, c - a)
        angle_b = linalg.angle(c - b, a - b)
        angle_c = linalg.angle(a - c, b - c)

        if angle_a < max and angle_b < max and angle_c < max:
            area += 0.125 * (
                linalg.norm_sqrd(b - a) * cotan(angle_c)
              + linalg.norm_sqrd(c - a) * cotan(angle_b))
        elif angle_a > max:
            area += 0.5 * face_area
        else:
            area += 0.25 * face_area

    return area


def vertex_area_voronoi(vertex):
    """ Voronoi vertex area.

    Voronoi area assigned to `vertex`, see [1]_. Only valid if none of
    the incident triangles is obtuse!

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh. The condition on non-obtuse incident
        triangles is not checked. Expect bogus results in such cases.

    Returns
    -------
    area : float
        Area assigned to `vertex`.

    See Also
    --------
    vertex_area_mixed : Handle non-obtuse triangles gracefully.

    References
    ----------
    .. [1] M. Meyer et al.: *Discrete differential-geometry operators for
           triangulated 2-manifolds*. In: HC. Hege, K. Polthier (eds)
           *Visualization and Mathematics III*, 2003.
    """
    cotan = lambda x: 1.0 / math.tan(x)

    a = vertex.point
    area = 0.0

    for h in vertex._hiter():
        if h.boundary:
            continue

        b = h.target.point
        c = h.next.target.point

        angle_b = linalg.angle(c - b, a - b)
        angle_c = linalg.angle(a - c, b - c)

        area += 0.125 * (
            linalg.norm_sqrd(b - a) * cotan(angle_c)
          + linalg.norm_sqrd(c - a) * cotan(angle_b))

    return area


def edge_stats(item):
    """ Edge length statistics.

    Minimal, maximal, average, and median edge length for a mesh or an
    individual face.

    Parameters
    ----------
    item : Face or Mesh
        Mesh instance or face of a mesh.

    Returns
    -------
    min, max, avg, med : float
        Minimal, maximal, average, and median edge length.
    """
    len = [linalg.norm(h.vector) for h in item._eiter()]
    return min(len), max(len), stats.fmean(len), stats.median(len)

    # min, max = np.inf, -np.inf
    # avg, cnt = 0.0, 0

    # for h in item._eiter():
    #     length = linalg.norm(h.vector)

    #     avg += length
    #     cnt += 1

    #     min = length if length < min else min
    #     max = length if length > max else max

    # return min, max, avg / cnt


def _halfedge_normal(halfedge):
    """ Mean curvature vector.
    """
    pass


def _halfedge_angle(self):
    """ Edge angle.

    Angle (in radians) between normals of adjacent faces. The angle
    is positive for convex edges. Applying the corresponding rotation
    (cf. :attr:`_rotation`) aligns adjacent face planes.

    :type: float

    Raises
    ------
    ValueError
        If called for a boundary halfedge.

    Note
    ----
    Our definition of a convex edge depends on the surface orientation.
    Typically one assumes outward normals.
    """
    assert False

    # Do not use this function. The angle is not defined for boundary
    # edges. If the normals of adjacent faces are know the oriented
    # angle can be obtained as
    #
    # linalg.angle(normal[h.face],
    #              normal[h.pair.face],
    #              h.vector)

    assert not self._deleted

    if self._face is None:
        raise ValueError('attribute undefined for boundary halfedge')

    if self._pair._face is None:
        return 0.0

    u = self._face._compute_normal()
    v = self._pair._face._compute_normal()

    clamp = lambda x, l, h: l if x < l else h if x > h else x
    alpha = math.acos(clamp(u.dot(v), -1.0, 1.0))

    # The vector orthogonal to both face normals (oriented according
    # to the right hand rule) determines the sign of alpha.
    if self.vector.dot(linalg.cross(u, v)) < 0.0:
        alpha *= -1.0

    return alpha


def _halfedge_rotation(self):
    r""" Rotation parameters.

    Computes the parameters :math:`\cos(\varphi)` and
    :math:`\sin(\varphi)` of a rotation that aligns the face plane
    of the halfedge's face with the face plane of the opposite face.
    The halfedge serves as oriented rotation axis. The rotation
    aligns the normal of adjacent faces.

    :type: (float, float)

    Raises
    ------
    ValueError
        If called for a boundary halfedge.

    Note
    ----
    Results in the identity along the boundary.
    """
    assert False

    # Do not use this function. It is superseded by axis_angle().

    assert not self._deleted

    if self._face is None:
        raise ValueError('attribute undefined for boundary halfedge')

    if self._pair._face is None:
        return 1.0, 0.0

    u = self._face._compute_normal()
    v = self._pair._face._compute_normal()

    # The vector orthogonal to both face normals. Oriented according
    # to the right hand rule.
    vector = linalg.cross(u, v)

    cos_alpha = u.dot(v)
    sin_alpha = linalg.norm(vector)

    if self.vector.dot(vector) < 0.0:
        sin_alpha *= -1.0

    # assert abs(math.sin(self._angle) - sin_alpha) < 1e-6

    return cos_alpha, sin_alpha


def face_normal(face):
    """ Face normal vector.

    Compute face normal as cross product of edge vectors. This function
    assumes that vertex coordinates live in Euclidean 3-space.

    Parameters
    ----------
    face : Face
        Face of a mesh.

    Returns
    -------
    normal : ndarray, shape (3, )
        Unit length normal vector. For triangular faces this vector
        is consistent with the face orientation. Deleted faces are
        assigned a vector of :obj:`~numpy.nan` values.

    See Also
    --------
    face_normals

    Notes
    -----
    For a non-triangular face the normal is computed by averaging vectors
    obtained as cross products of consecutive edge vectors around the face.
    This approach can be problematic for non-convex faces as some of those
    vectors point to the wrong side.
    """
    # Deleted faces are quietly assigned a normal vector with nan entries.
    # Accessing properties of a deleted face would trigger an assertion.
    if face.deleted:
        return np.full(3, np.nan)

    if len(face) == 3:
        halfedge = face.halfedge
        vector = linalg.cross(halfedge.vector, halfedge.next.vector)
    else:
        vector = np.zeros(3, dtype=float)

        for h in face._hiter():
            # For non-convex faces some normals computed in this way point
            # to the wrong side. For a flat star neighborhood this sum may
            # even average to the zero vector.
            vector += linalg.unit_inplace(linalg.cross(h.vector,
                                                       h.next.vector))

    # Note that division by zero produces a runtime warning. Divison of
    # nan by nan results in nan and does not show a warning.
    return linalg.unit_inplace(vector)


def face_normals(mesh):
    """ Face normals.

    Compute face normals as cross products of edge vectors. This function
    assumes that vertex coordinates live in Euclidean 3-space.

    Parameters
    ----------
    mesh : Mesh
        Mesh with polygonal faces.

    Returns
    -------
    normals : ndarray, shape (m, 3)
        Unit length face normal vectors for a mesh with m faces.

    See Also
    --------
    face_normal

    Notes
    -----
    This function returns a vector for all faces of a mesh, even for those
    marked as deleted. In the latter case a row of all :obj:`~numpy.nan`
    values is assigned.
    """
    # Note that a mesh itself is iterable. In the presence of deleted faces
    # the lists [f for f in mesh] and [f for f in mesh.faces] are different!
    return np.array([face_normal(f) for f in mesh.faces])


def face_area(face):
    """ Area of triangular face.

    Parameters
    ----------
    face : Face
        A triangular face.

    Raises
    ------
    NotImplementedError
        For non-triangular faces.

    Returns
    -------
    area : float
        Face area.

    Notes
    -----
    For faces of higher valence, an ad-hoc fan-like triangulation of the face
    would still give a wrong result for a non-planar and/or non-convex face.
    """
    if len(face) == 3:
        halfedge = face.halfedge
        vector = linalg.cross(halfedge.vector, halfedge.next.vector)
    else:
        raise NotImplementedError('triangular face required')

    return 0.5 * linalg.norm(vector)


def planarity_score(face, denom=None):
    """ Planarity score of a skew quadrilateral.

    Distance of diagonals. Can be normalized to make the measure scale
    independent.

    Parameters
    ----------
    face : array_like, shape (4, 3)
        Vertex coordinate array.
    denom : float, optional
        Normalization factor. By default, the diagonal distance is
        devided by the average length of the quadrilateral's edges.

    Returns
    -------
    float
        Planarity score of quadrilateral. Can result in :obj:`~numpy.inf`
        or :obj:`~numpy.nan` values in case of flipped or degenerate
        quadrilaterals. Always returns :obj:`~numpy.nan` for faces of
        valence greater than four.
    """
    f = np.asarray(face)

    # Return 0.0 for triangular faces and NaN for faces of higher valence.
    if len(f) < 4:
        return 0.0
    elif len(f) > 4:
        return np.nan

    # For bow tie shaped quadrilaterals the vectors v and w can be linear
    # depedent without one of them vanishing.
    v = f[2] - f[0]
    w = f[3] - f[1]

    # Distance of diagnoals can result in inf and nan values in case of
    # flipped or degenerate quadrilaterals.
    n = linalg.cross(v, w)
    d = abs(n.dot(f[1] - f[0]) / linalg.norm(n))

    if denom is None:
        denom = 0.25 * np.sum(np.linalg.norm([f[1]-f[0],
                                              f[2]-f[1],
                                              f[3]-f[2],
                                              f[0]-f[3]], axis=1))

    # Using 1.0 for normalization one gets the unscaled distance of
    # diagonals in whatever unit one is computing.
    return d / denom


def planarity_scores(mesh, denom=None, offset=0.0):
    """ Face planarity scores.

    Parameters
    ----------
    mesh : Mesh
        Mesh instance.
    denom :  float, optional
        Normalization factor. By default, each diagonal distance is
        devided by the average length of the quadrilateral's edges.
    offset : float, optional
        Added to each planarity score.

    Returns
    -------
    list[float]
        Face planarity scores plus an additional offset value.

    See Also
    --------
    planarity_score

    Notes
    -----
    Use the mesh's bounding box diagonal or mean edge length to achieve
    gobal normalization, e.g.

    >>> planarity_scores(mesh, edge_stats(mesh)[-1])
    """
    return [offset + planarity_score(f, denom) for f in mesh]