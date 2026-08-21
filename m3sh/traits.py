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
traits like normals (see, e.g., [1]_) and curvature (see [2]_).

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
import scipy as sp

import m3sh.linalg as linalg


def area(mesh):
    """ Surface area of triangle mesh.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh instance.

    Returns
    -------
    area : float
        Surface area. This value is meaningless if `mesh` has polygonal
        faces with valence greater than three.

    See Also
    --------
    face_area
    """
    area = 0.0

    # Use the face iterator of a mesh. It skips all faces that are marked
    # as deleted.
    for f in mesh._fiter():
        area += linalg.norm(linalg.cross(f.halfedge.vector,
                                         f.halfedge.next.vector))

    return 0.5 * area


def axis_angles(mesh, normals=None):
    r""" Rotation parameters.

    Edge based rotation parameters that align neighboring face planes.
    Along the boundary this rotation is the identity.

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
        rotate the normal of ``halfedge.face`` about the common edge to
        the normal of ``halfedge.pair.face``.

    See Also
    --------
    :func:`~m3sh.linalg.rotation`, :func:`~m3sh.linalg.rotate`

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
            # print(abs(axis.dot(normals[h.face])))
            # print(abs(axis.dot(normals[h.pair.face])))

            # assert abs(axis.dot(normals[h.face])) < 1e-12
            # assert abs(axis.dot(normals[h.pair.face])) < 1e-12

            # This rotation aligns the face plane of h.face with the face
            # plane of the neighboring face h.pair.face.
            cos_phi, sin_phi = linalg.rotation(normals[h.face],
                                               normals[h.pair.face], axis)

        axis_angles[h] = (axis, cos_phi, sin_phi)
        axis_angles[h.pair] = (-axis, cos_phi, sin_phi)

        # Validate the connection with the dihedral angle. Remove this
        # test later!
        angle = dihedral_angle(h)

        if abs(cos_phi - math.cos(angle)) > 1e-9:
            print(f"{cos_phi=}, {math.cos(angle)=}")

        if abs(sin_phi - math.sin(angle)) > 1e-9:
            print(f"{sin_phi=}, {math.sin(angle)=}")

    return axis_angles


def cotan_weight(halfedge, boundary=np.nan, clamp=False):
    """ Cotangent weight.

    Compute the cotangent of the angle opposite of `halfedge`.

    Parameters
    ----------
    halfedge : Halfedge
        A halfedge instance of a triangle mesh.
    boundary : float, optional
        Default value if `halfedge` is on the boundary. A :obj:`~numpy.nan`
        value typically signals missing data. For numerical computations
        the value 0.0 can be useful to skip missing data without raising an
        error.
    clamp : bool, optional
        Pass :obj:`True` to set negative weights to zero.

    Returns
    -------
    float
        Cotangent value. This value is negative if the opposite angle
        is obtuse. For halfedges incident to degenerate triangles (zero
        area) this value can be :obj:`~numpy.inf` or :obj:`~numpy.nan`
        depending on the type of degeneracy. Such cases currently trigger
        an assertion.
    """
    if halfedge.boundary:
        return boundary
    else:
        # Dividing 0/0 results in NaN. Happens when two vertices coincide.
        # Otherwise division by zero results in inf, i.e., when a vertex
        # is contained in an edge but not one of the endpoints.
        cotan = linalg.cotan(halfedge.prev.vector,
                             halfedge.next.pair.vector)

        # For non-boundary edges one would expect to get a finite result.
        # Could be turned into an exception.
        assert math.isfinite(cotan)

        if clamp:
            return max(cotan, 0.0)

        return cotan


def cotan_weights(mesh, boundary=np.nan, clamp=False):
    """ Cotangent weights.

    Compute cotangent weights for all halfedges of `mesh`.

    Parameters
    ----------
    mesh : Mesh
        A triangle mesh instance.
    boundary : float, optional
        Default value if `halfedge` is on the boundary of `mesh`.
    clamp : bool, optional
        Pass :obj:`True` to set negative weights to zero.

    Returns
    -------
    weights : dict[Halfedge, float]
        A dictionary that maps halfedge instances to their cotangent
        weight.

    See Also
    --------
    cotan_weight

    Notes
    -----
    Weights are assigned to halfedges! Add the weights of halfedge pairs
    to obtain edge weights (as in the definition of the cotan-Laplacian).
    """
    # The halfedge dictionary of a mesh never contains deleted halfedges.
    # For halfedges the exceptional case is being on the boundary.
    return {h: cotan_weight(h, boundary, clamp)
            for h in mesh.halfedges.values()}


def diameter(points):
    """ Diameter of a point set.

    Parameters
    ----------
    points : array_like, shape (m, n)
        Point coordinates in n-space.

    Returns
    -------
    float
        Point set diameter.

    See Also
    --------
    scipy.spatial.distance : SciPy module for distance computations
    """
    # As an intermediate result a matrix of size O(m^2) is computed. Is
    # this a good idea for large data sets?
    return sp.spatial.distance.pdist(np.asarray(points)).max()


def dihedral_angle(halfedge, degrees=False):
    """ Dihedral angle.

    Dihedral angle along `halfedge` in radians, i.e., the angle between
    the normals of adjacent faces. The dihedral angle has a sign:
    assuming the outward normal field it is positive for a convex edge
    and negative for a concave edge.

    Parameters
    ----------
    halfedge : Halfedge
        Halfedge of a triangle mesh.
    degrees : bool, optional
        Convert result from radians to degrees.

    Returns
    -------
    angle : float
        Angle in degrees or radians. If `halfedge` or its pair is a
        boundary halfedge the corresponding dihedral angle does not
        exists and :obj:`~numpy.nan` is returned.
    """
    if halfedge.boundary or halfedge.pair.boundary:
        return np.nan

    # Computing cos() and sin() of this angle should be the same as the
    # cos_phi and sin_phi values returned by linalg.rotation().
    return linalg.angle(face_normal(halfedge.face),
                        face_normal(halfedge.pair.face),
                        up=halfedge.vector, degrees=degrees)


def dihedral_angles(mesh, normals=None, degrees=False):
    """ Dihedral angles.

    Dihedral angles for all halfedges of `mesh`.

    Parameters
    ----------
    mesh : Mesh
        A polyhedral mesh instance.
    normals : ndarray, shape (m, 3), optional
        Precomputed faces normals, `m` denotes the number of faces.
    degrees : bool, optional
        Convert angles from radians to degrees.

    Returns
    -------
    angles : dict[Halfedge, float]
        A dictionary that maps halfedges to the corresponding dihedral
        angle.

    See Also
    --------
    dihedral_angle
    """
    normals = face_normals(mesh) if normals is None else normals
    angles = dict()

    for h in mesh._eiter():
        angle = linalg.angle(normals[h.face], normals[h.pair.face],
                             up=h.vector, degrees=degrees)

        angles[h] = angle
        angles[h.pair] = angle

    return angles


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
    normal : ndarray, shape (3,)
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
    if vertex.deleted or vertex.isolated:
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


def mean_curvature_vector(vertex):
    r""" Mean curvature vector.

    .. math::

       \mathbf{H}_i = \sum_{j \sim i} (\cot \alpha_{ij} + \cot \beta_{ij})
       (\mathbf{v}_i - \mathbf{v}_j) / \operatorname{area} \mathbf{v}_i

    is the gradient of surface area (not true because of divsion by
    local area term).

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.

    Returns
    -------
    mcv : ndarray
        The mean curvature vector assigned to `vertex`. For deleted
        vertices this array holds :obj:`~numpy.nan` entries.
    """
    if vertex.deleted or vertex.isolated:
        return np.full_like(vertex.point, np.nan)

    mcv = np.zeros_like(vertex.point)

    for h in vertex._hiter():
        mcv += h.pair.vector * (
            cotan_weight(h, 0.0) + cotan_weight(h.pair, 0.0))

    mcv /= 2.0 * vertex_area(vertex)
    return mcv


def mean_curvature_vectors(mesh, weights=None):
    """ Mean curvature vectors.

    Parameters
    ----------
    mesh : Mesh
        A triangle mesh instance.
    weights : dict[Halfedge, float], optional
        A dictionary with precomputed cotangent weights. When using
        precomputed weights, the weight assigned to boundary
        halfedges has to be zero!

    Returns
    -------
    vectors : ndarray, shape (n, 3)
        Array of mean curvature vectors where n denotes the number of
        vertices of `mesh` (including vertices marked as deleted).

    See Also
    --------
    mean_curvature_vector
    """
    # When precomputing weights, boundary halfedges should be assigned
    # zero weights!
    w = cotan_weights(mesh, 0.0) if weights is None else weights
    a = vertex_areas(mesh, w)

    mcvs = np.full_like(mesh.points, np.nan)

    # This iterator skips deleted vertices. Isolated vertices do not
    # have incident halfedges and mcv results in 0.0 / 0.0 = NaN.
    for v in mesh._viter():
        mcvs[v] = sum(h.pair.vector * (w[h] + w[h.pair]) for h in v._hiter())
        mcvs[v] /= 2.0 * a[v]

    return mcvs


def mean_curvature(vertex, normal=None):
    """ Mean curvature estimate.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.
    normal : ndarray, shape (3,), optional
        Vertex normal, used to determine the sign of mean curvature.

    Returns
    -------
    H : float
        Mean curvature.

    Notes
    -----
    Absolute mean curvature can be obtained as

    >>> 0.5 * norm(mean_curvature_vector(vertex))
    """
    if vertex.deleted or vertex.isolated:
        return np.nan

    mcv = mean_curvature_vector(vertex)
    vec = vertex_normal(vertex) if normal is None else normal

    if mcv.dot(vec) < 0.0:
        return -0.5 * linalg.norm(mcv)

    return 0.5 * linalg.norm(mcv)


def mean_curvatures(mesh, mcvs=None, normals=None):
    """ Mean curvature estimates.
    """
    mcvs = mean_curvature_vectors(mesh) if mcvs is None else mcvs
    vecs = vertex_normals(mesh) if normals is None else normals

    sign = np.array([-1.0, 1.0])
    sign = sign[(np.vecdot(mcvs, vecs) > 0.0).astype(int)]

    return 0.5 * np.linalg.norm(mcvs, axis=-1) * sign


def gauss_curvature(vertex):
    """ Gaussian curvature estimate.

    Point-wise Gaussian curvature estimate.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.

    Returns
    -------
    float
        Gaussian curvature estimate. Isolated vertices and vertices
        marked as deleted are assigned :obj:`~numpy.nan`.

    """
    # Should curvature at boundary vertices be estimated by interpolation
    # of values at adjacent interior vertices?
    if vertex.deleted or vertex.isolated:
        return np.nan

    return (2.0 * np.pi - vertex_angle(vertex)) / vertex_area(vertex)


def gauss_curvatures(mesh, weights=None):
    """ Gaussian curvate estimate.

    Point-wise Gaussian curvature estimate for all vertices of `mesh`.

    Parameters
    ----------
    mesh : Mesh
        A triangle mesh instance.
    weights : dict[Halfedge, float], optional
        A dictionary of precomputed cotangent weights.

    Returns
    -------
    ndarray
        An array of length n, where n denotes the number of vertices
        of `mesh`, holding Gaussian curvature estimates.
    """
    w = cotan_weights(mesh) if weights is None else weights
    a = vertex_areas(mesh, w)

    return np.array([(2.0 * np.pi - vertex_angle(v)) / a[v]
                     for v in mesh.vertices])


def principal_curvature(vertex, normal=None):
    """ Principal curvature estimate.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.
    normal : ndarray, shape (3,), optional
        Vertex normal.
    """
    if vertex.deleted or vertex.isolated:
        return np.nan, np.nan

    H = mean_curvature(vertex, normal)
    K = gauss_curvature(vertex)
    D = math.sqrt(max(H*H - K, 0.0))

    return H + D, H - D


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

    See Also
    --------
    gauss_curvature

    Notes
    -----
    The value :math:`2\pi - \sum_{i=1}^k \alpha_i` is called angle defect
    or discrete Gaussian curvature. A point-wise estimate of Gaussian
    curvature is often defined as the quotient

    >>> K = (2.0 * pi - vertex_angle(v)) / vertex_area(v)
    """
    if vertex.deleted or vertex.isolated:
        return np.nan

    angle = 0.0

    for h in vertex._hiter():
        if h.face is not None:
            angle += linalg.angle(h.vector, h.prev.pair.vector)

    return angle


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


def vertex_area(vertex):
    """ Mixed vertex area.

    The mixed area assigned to `vertex` as defined in [1]_.

    Parameters
    ----------
    vertex : Vertex
        Vertex of a triangle mesh.

    Returns
    -------
    area : float
        Area assigned to `vertex`. Isolated vertices and vertices
        marked as deleted are assigned :obj:`~numpy.nan` as area.

    References
    ----------
    .. [1] M. Meyer et al.: *Discrete differential-geometry operators for
           triangulated 2-manifolds*. In: HC. Hege, K. Polthier (eds)
           *Visualization and Mathematics III*, 2003.
    """
    if vertex.deleted or vertex.isolated:
        return np.nan

    area = 0.0

    for h in vertex._hiter():
        if not h.boundary:
            # The default value for boundary edges does actually not
            # matter: if h is not a boundary halfedge, no halfedge in
            # the same cycle can be a boundary halfedge!
            cotan_c = cotan_weight(h, 0.0)
            cotan_a = cotan_weight(h.next, 0.0)
            cotan_b = cotan_weight(h.prev, 0.0)

            if cotan_a > 0.0 and cotan_b > 0.0 and cotan_c > 0.0:
                # All interior angles of the triangle are smaller than
                # pi/2. Voronoi area is contained inside the triangle.
                area += 0.125 * (
                    linalg.norm_sqrd(h.vector) * cotan_c
                    + linalg.norm_sqrd(h.prev.vector) * cotan_b)
            elif cotan_a <= 0.0:
                # The angle at vertex is obtuse (greater than pi/2).
                area += 0.5 * face_area(h.face)
            else:
                area += 0.25 * face_area(h.face)

    # Remove test and assertion later!
    assert abs(area - vertex_area_mixed(vertex)) < 1e-12
    # print(f"{abs(area - vertex_area_mixed(vertex)):.6e}")

    return area


def vertex_areas(mesh, weights=None):
    """ Mixed vertex areas.

    Parameters
    ----------
    mesh : Mesh
        A triangle mesh instance.
    weights : dict[Halfedge, float], optional
        A dictionary of precomputed cotangent weights.

    Returns
    -------
    area : ndarray
        An array of length n, where n denotes the number of vertices
        of `mesh` (including vertices marked as deleted).

    See Also
    --------
    vertex_area
    """
    w = cotan_weights(mesh, 0.0) if weights is None else weights
    a = np.full(len(mesh.vertices), np.nan)

    for v in mesh._viter():
        a[v] = np.nan if v.isolated else 0.0

        for h in v._hiter():
            if not h.boundary:
                cotan_c = w[h]
                cotan_a = w[h.next]
                cotan_b = w[h.prev]

                if cotan_a > 0.0 and cotan_b > 0.0 and cotan_c > 0.0:
                    a[v] += 0.125 * (
                        linalg.norm_sqrd(h.vector) * cotan_c
                        + linalg.norm_sqrd(h.prev.vector) * cotan_b)
                elif cotan_a <= 0.0:
                    a[v] += 0.5 * face_area(h.face)
                else:
                    a[v] += 0.25 * face_area(h.face)

    return a


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


def face_area(face):
    """ Area of triangular face.

    Compute the area of a triangle. Faces marked as deleted are assigned
    :obj:`~numpy.nan` as area.

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

    See Also
    --------
    area

    Notes
    -----
    For non-triangular faces, an ad-hoc fan-like triangulation would still
    give a wrong result for a non-planar and/or non-convex faces.

    Examples
    --------
    Use a list comprehension to compute the area for all faces of a mesh:

    >>> areas = [face_area(f) for f in mesh.faces]


    """
    if face.deleted:
        return np.nan

    if len(face) == 3:
        halfedge = face.halfedge
        vector = linalg.cross(halfedge.vector, halfedge.next.vector)
    else:
        raise NotImplementedError('triangular face required')

    return 0.5 * linalg.norm(vector)


# def curvature(mesh, normals=None):
#     """ Face based curvature estimation.

#     Returns
#     -------
#     kappa : ndarray, shape (2,)
#         Principal curvature values.
#     vec : ndarray, shape (2, 3)
#         Corresponding principal carvature directions.
#     """

#     def shape_operator(face, normals):
#         # The local positively oriented orthonormal bases of the tangent
#         # space, i.e., the plane of the triangle. The shape operator with
#         # respect to such a bases has a symmetric matrix.
#         x = linalg.unit_inplace(face.halfedge.vector)
#         y = face.halfedge.prev.pair.vector
#         y = linalg.unit_inplace(y - x.dot(y) * x)

#         # The sequence of points and normals has to correspond by index!
#         points = [v.point for v in face]

#         lhs = np.zeros((6, 3))
#         rhs = np.zeros(6)

#         for i in range(3):
#             j = (i + 1) % 3

#             v = points[j] - points[i]
#             v = (x.dot(v), y.dot(v))

#             lhs[2*i] = [v[0], v[1], 0.0]
#             lhs[2*i + 1] = [0.0, v[0], v[1]]

#             n = normals[j] - normals[i]
#             n = (x.dot(n), y.dot(n))

#             rhs[2*i] = n[0]
#             rhs[2*i + 1] = n[1]

#         a, b, c = np.linalg.lstsq(lhs, rhs)[0]
#         w = np.array([[a, b], [b, c]])
#         k, v = np.linalg.eigh(w)

#         return k, v.T @ [x, y]

#     if normals is None:
#         normals = face_normals(mesh)

#     kappa = np.full((len(mesh.faces), 2), np.nan)
#     vec = np.full((len(mesh.faces), 2, 3), np.nan)

#     mean = np.full(len(mesh.faces), np.nan)
#     gauss = np.full(len(mesh.faces), np.nan)

#     for face in mesh.faces:
#         k, v = shape_operator(face, [normals[v] for v in face])

#         kappa[face] = k
#         vec[face] = v

#         mean[face] = 0.5 * (k[0] + k[1])
#         gauss[face] = k[0] * k[1]

#     H = np.full(len(mesh.vertices), np.nan)
#     K = np.full(len(mesh.vertices), np.nan)

#     for v in mesh.vertices:
#         H[v] = sum(mean[f] for f in v._fiter()) / v.degree
#         K[v] = sum(gauss[f] for f in v._fiter()) / v.degree

#     return H


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
    normal : ndarray, shape (3,)
        Unit length normal vector. For triangular faces this vector
        is consistent with the face orientation. Faces marked as deleted
        are assigned a vector of :obj:`~numpy.nan` values.

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
    # The first list omits deleted faces.
    return np.array([face_normal(f) for f in mesh.faces])





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


def plane_equation(face):
    r""" Equation of face plane.

    .. version-added:: 1.1.0

    Computes the coefficient vector :math:`\mathbf{h} = (\mathbf{n}, c)
    \in \mathbb{R}^3 \times \mathbb{R}` of the Hessian normal form of the
    plane spanned by `face`, i.e., :math:`\mathbf{n}^T \mathbf{x} + c = 0`
    such that :math:`\|\mathbf{n}\| = 1`.

    Parameters
    ----------
    face : Face
        Face of a triangle mesh.

    Returns
    -------
    h : ndarray
        Coefficient vector of the Hessian normal form. The computed
        equation is only exact if `face` is planar. Faces marked as
        deleted are assigned a vector of :obj:`~numpy.nan` values.
    """
    if face.deleted:
        return np.full(4, np.nan)

    vec = face_normal(face)
    ofs = vec.dot(face.halfedge.origin.point)

    return vec, ofs
    # eqn = np.array([*vec, -ofs])

    # return eqn


def laplace_matrix(mesh, weights=None, normalize=False):
    """ Laplace matrix.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh instance.
    weights : dict[Halfedge, float], optional
        Edge weights. By default cotangent weights are used.
    normalize : bool, optional
        Normalize result to obtain corresponding umbrella operator.

    Returns
    -------
    L : csr_array
        Laplace matrix.
    """
    print(1)

    # Use halfedge weights 0.5 to get the uniform Laplace matrix or,
    # when normalizing, the uniform umbrella operator.
    w = cotan_weights(mesh, 0.0, True) if weights is None else weights
    n = len(mesh.vertices)
    L = sp.sparse.lil_array((n, n))

    for i, v in enumerate(mesh.vertices):
        weight_sum = 0.0

        for h in v._hiter():
            L[i, int(h.target)] = (weight := w[h] + w[h.pair])
            weight_sum += weight

        L[i, i] = -weight_sum

        if normalize:
            L[i, :] /= weight_sum

    return L.tocsr()


def smooth_(mesh, func, iterations, step=0.5, weights=None):
    """ Smooth piecewise linear function.

    Parameters
    ----------
    mesh : Mesh
        A triangle mesh instance.
    func : ndarray, shape (n, ...)
        A piecewise linear function defined by values at the
        n vertices of `mesh`.
    iterations : int
        Number of
    step : float, optional
        Step size.
    weights : dict[Halfedge, float], optional
        Halfedge weights of Laplace operator. By default cotangent
        weights are used.

    Returns
    -------
    func : ndarray
        The smoothed function (not a copy).
    """
    L = laplace_matrix(mesh, weights, normalize=True)

    for _ in range(iterations):
        func += step * L @ func

    return func


def smooth(mesh, func, iterations, step=0.5, weights=None):
    """ Smooth piecewise linear function.

    Parameters
    ----------
    mesh : Mesh
        A triangle mesh instance.
    func : ndarray, shape (n, ...)
        A piecewise linear function defined by values at the
        n vertices of `mesh`.
    iterations : int
        Number of
    step : float, optional
        Step size.
    weights : dict[Halfedge, float], optional
        Halfedge weights of Laplace operator. By default cotangent
        weights are used.

    Returns
    -------
    func : ndarray
        The smoothed function (not a copy).
    """
    w = cotan_weights(mesh, 0.0, True) if weights is None else weights

    for _ in range(iterations):
        for v in mesh.vertices:
            weight_sum = 0.0

            for h in v._hiter():
                d = (weight := w[h] + w[h.pair]) * (func[h.target] - func[v])
                weight_sum += weight

            func[v] += step * (d / weight_sum)

    return func
