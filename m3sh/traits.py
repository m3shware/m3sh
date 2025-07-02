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

""" Geometric mesh traits.

Convenience functions to compute common and often used geometric mesh
traits like vertex and face normals, etc.
"""

import math
import numpy as np

import m3sh.linalg as linalg


def _axis_angles(mesh, normals=None):
    """

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh instance.
    normals : ~numpy.ndarray
        Face normal vectors.

    Returns
    -------
    dict
    """
    if normals is None:
        normals = face_normals(mesh)

    axis_angles = dict()

    for h in mesh._eiter():
        axis = h.vector / linalg.norm(h.vector)

        if h.boundary or h.pair.boundary:
            cos_phi, sin_phi = 1.0, 0.0
        else:
            cos_phi, sin_phi = linalg.rotation(normals[h.face],
                                               normals[h.pair.face], axis)

        axis_angles[h] = (axis, cos_phi, sin_phi)
        axis_angles[h.pair] = (-axis, cos_phi, sin_phi)

    return axis_angles


def bounds(points):
    r""" Bounding box vertices.

    Corner vertices of the axis-aligned bounding box.

    Parameters
    ----------
    points : array_like, shape (n, k)
        Coordinates of :math:`n` points in :math:`\mathbb{R}^k`,
        one point per row.

    Returns
    -------
    a : ~numpy.ndarray
        Holds the minimum value for each dimension.
    b : ~numpy.ndarray
        Holds the maximum value for each dimension.
    """
    return np.min(points, axis=0), np.max(points, axis=0)


def vertex_normal(vertex):
    """ Vertex normal.

    Compute vertex normal as average of triangle normals. For
    non-triangular meshes, incident triangles are defined by the
    planes spanned by consecutive edges in a counter-clockwise
    traversal of all incident edges.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a mesh.

    Returns
    -------
    ~numpy.ndarray, shape (3, )
        Unit normal vector.

    Note
    ----
    Vertex normals are not well defined for isolated vertices.
    """
    # Triggers an assertion for deleted vertices. Alternative: quitely
    # assign nan normal vector to deleted vertices.
    if vertex.degree == 0:
        return np.full_like(vertex.point, np.nan)

    normal = np.zeros_like(vertex.point)

    # We checked for vanishing degree, hence this iterator cannot be
    # empty.
    for h in vertex._hiter():
        if h._face is not None:
            normal += linalg.unit(linalg.cross(h.prev.vector,
                                               h.vector))

    return linalg.unit(normal)


def vertex_normals(mesh, broadcast=False):
    """ Vertex normals.

    Compute vertex normals as average of face normals.

    Parameters
    ----------
    mesh : Mesh
        A mesh.
    broadcast : bool, optional
        Broadcast or gather face normals.

    Returns
    -------
    ~numpy.ndarray, shape (n, 3)
        Unit normal vectors for a mesh with n vertices.

    Note
    ----
    Vertex normals are not well defined for isolated vertices. The
    result includes :obj:`np.nan` values in this case.
    """
    if broadcast:
        # Outer loop runs over faces. Each face broadcasts is normal
        # vector to each incident vertex. This is typically faster.
        normals = np.zeros_like(mesh.points)

        for f in mesh.faces:
            normals[f] += face_normal(f)

        # Division by 0 results in nan entries for normal vectors of
        # deleted and isolated vertices.
        normals /= np.linalg.norm(normals, axis=-1)[:, None]
        return normals
    else:
        # Outer loop runs over vertices. Gather normals of incident
        # faces to compute the vertex normal.
        return np.array([vertex_normal(v) for v in mesh.vertices])


def _vertex_normals(mesh):
    """ Vertex normals.

    Parameters
    ----------
    mesh : Mesh
        A mesh.

    Returns
    -------
    ~numpy.ndarray, shape (n, 3)
        Unit normal vectors for a mesh with n vertices.

    Note
    ----
    Vertex normals are not well defined for isolated vertices.
    """
    normals = np.zeros_like(mesh.points)

    for f in mesh.faces:
        n = linalg.unit(linalg.cross(f.halfedge.vector,
                                     f.halfedge.next.vector))

        for v in f._viter():
            normals[v, ...] += n

    normals /= np.linalg.norm(normals, axis=-1)[:, None]
    return normals


def _vertex_angle(vertex):
    clamp = lambda x, l, h: l if x < l else h if x > h else x
    angle = 2.0 * math.pi

    for h in vertex._hiter():
        if h._face is not None:
            u = linalg.unit(h._prev._pair.vector)
            v = linalg.unit(h.vector)

            angle += math.acos(clamp(u.dot(v), -1.0, 1.0))

    return angle


def _vertex_area_simple(vertex):
    """
    """
    return sum(face_area(f) for f in vertex._fiter()) / 3.0


def _vertex_area_mixed(vertex):
    """
    """
    a = vertex.point
    weight = 0.0
    val = 0.5 * math.pi

    for h in vertex._hiter():
        if h.boundary:
            continue

        b = h.target.point
        c = h.next.target.point

        area = 0.5 * linalg.norm(linalg.cross(b-a, c-b))

        angle_a = linalg.angle(b-a, c-a)
        angle_b = linalg.angle(c-b, a-b)
        angle_c = linalg.angle(a-c, b-c)

        if angle_a < val and angle_b < val and angle_c < val:
            weight += 0.125 * (  linalg.sqrd(b-a) * angle_c
                               + linalg.sqrd(c-a) * angle_b)
        elif angle_a > val:
            weight += 0.5 * area
        else:
            weight += 0.25 * area

    return weight


def edge_length(item):
    """ Edge length statistics.

    Minimal, maximal, and average edge length for a mesh or an
    individual face.

    Parameters
    ----------
    item : Face or Mesh
        Mesh or face of a mesh.

    Returns
    -------
    min : float
        Minimal edge length.
    max : float
        Maximal edge length.
    avg : float
        Average edge length.
    """
    min, max = np.inf, -np.inf
    avg, cnt = 0.0, 0

    for h in item._eiter():
        length = linalg.norm(h.vector)

        avg += length
        cnt += 1

        min = length if length < min else min
        max = length if length > max else max

    return min, max, avg / cnt


def _angle_defect(vertex):
    r""" Angle defect.

    For a vertex with :math:`k` incident angles :math:`\alpha_i`,
    the value :math:`2\pi - \sum_{i=1}^k \alpha_i` is called angular
    defect or discrete Gaussian curvature.

    Note
    ----
    Values obtained for boundary vertices (when interpreted as discrete
    Gaussian curvature) are questionable.
    """
    clamp = lambda x, l, h: l if x < l else h if x > h else x
    defect = 2.0 * math.pi

    for h in vertex._hiter():
        if h._face is not None:
            u = linalg.unit(h._prev._pair.vector)
            v = linalg.unit(h.vector)

            defect -= math.acos(clamp(u.dot(v), -1.0, 1.0))

    return defect


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
    """ Face normal.

    Compute face normal as cross product of edge vectors.

    Parameters
    ----------
    face : Face
        Face of a mesh.

    Returns
    -------
    ~numpy.ndarray, shape (3, )
        Unit normal vector.

    Note
    ----
    For a non-triangular face the normal is computed by averaging
    vectors obtained as cross products of consecutive edges around
    the faces.
    """
    # Should deleted faces be quitely assigned a normal with nan
    # entries? Currenlty this case triggers an assertion error.
    if len(face) == 3:
        vector = linalg.cross(face.halfedge.vector,
                              face.halfedge.next.vector)
    else:
        vector = np.zeros(3, dtype=float)

        for h in face._hiter():
            # For non-convex faces some normals computed in this
            # way point to the wrong side.
            vector += linalg.unit_inplace(linalg.cross(h.vector,
                                                       h.next.vector))

    return linalg.unit_inplace(vector)


def face_normals(mesh):
    """ Face normals estimate.

    Estimate face normals by averaging the cross product of consecutive
    edge vectors around each face.

    Parameters
    ----------
    mesh : Mesh
        Mesh with polygonal faces.

    Returns
    -------
    ~numpy.ndarray
        Array of face normal vectors.
    """
    # Should deleted faces be quitely assigned a normal with nan
    # entries?
    return np.array([face_normal(f) for f in mesh.faces])


def face_area(face):
    """ Face area.

    Areas are only computed for triangular faces.

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
    float
        Face area.
    """
    if len(face) == 3:
        vector = linalg.cross(face.halfedge.vector,
                              face.halfedge.next.vector)
    else:
        raise NotImplementedError('triangular face required')

    return 0.5 * linalg.norm(vector)


def planarity_score(face, denom=0.0):
    """ Planarity score of skew quadrilateral.

    Distance of diagonals. Can be normalized to make the measure scale
    independent.

    Parameters
    ----------
    face : array_like, shape (4, 3)
        Vertex coordinate array.
    denom : float, optional
        Zero divides by mean edge length.

    Returns
    -------
    float
        Planarity score. Can result in inf and nan values in case of
        flipped or degenerate quadrilaterals.
    """
    f = np.asarray(face)

    # Return 0.0 for triangular faces and NaN for faces of higher valence.
    if len(f) < 4:
        return 0.0
    elif len(f) > 4:
        return np.nan

    # For bow tie shaped quadrilateral the vectors v and w can be linear
    # depedent without one of them vanishing.
    v = f[2] - f[0]
    w = f[3] - f[1]

    # Distance of diagnoals can result in inf and nan values in case of
    # flipped or degenerate quadrilaterals.
    n = linalg.cross(v, w)
    d = abs(linalg.dot(f[1] - f[0], n) / linalg.norm(n))

    if denom == 0.0:
        denom = 0.25 * np.sum(np.linalg.norm([f[1]-f[0],
                                              f[2]-f[1],
                                              f[3]-f[2],
                                              f[0]-f[3]], axis=1))

    # Using 1.0 for normalization one gets the unscaled distance of
    # diagonals in whatever unit one is computing.
    return d / denom


def planarity_scores(mesh, denom=0.0, offset=0.0):
    """ Face planarity scores.

    Parameters
    ----------
    mesh : Mesh
        Mesh instance.
    denom :  float, optional
        Zero divides by mean edge length.
    offset : float, optional
        Added to each planarity score.

    Returns
    -------
    list[float]
        Face planarity scores plus an additional offset value.
    """
    return [offset + planarity_score(f, denom) for f in mesh]