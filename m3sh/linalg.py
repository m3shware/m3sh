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

""" Linear algebra.

Basic vector math, linear algebra, and trigonometric functions. Many
functions are convenience functions that wrap NumPy functionality or
provide basic operations from analytic geometry.
"""

import math
import numpy as np


def affine_map(p, q):
    r""" Affine map from points in general position.

    Compute matrix :math:`A` and vector :math:`\mathbf{b}` such that
    :math:`A \mathbf{p}_i + \mathbf{b} = \mathbf{q}_i`.

    Parameters
    ----------
    p, q : array_like, shape (n+1, n)
        Two sequences of n+1 points in general position.

    Returns
    -------
    A : ndarray, shape (n, n)
        The linear part of the affine map.
    b : ndarray, shape (n,)
        The translational part of the affine map.
    """
    # Shape of p and q has to be (n+1) x n, i.e., n+1 points in n dim.
    # space. Points are stored as rows!
    p = np.append(p, [[1.0]] * len(p), axis=1)
    q = np.asarray(q)
    x = np.linalg.solve(p, q).T

    return x[:, :-1], x[:, -1]


def angle(u, v, up=None, degrees=False):
    """ Angle between vectors.

    Angle between vectors `u` and `v` in radians. To obtain an oriented
    angle the `up` vector has to be specified. If specified, `up` must
    not be contained in the span of `u` and `v`.

    Parameters
    ----------
    u, v : ndarray, shape (3,)
        Non-zero vectors in 3-space.
    up : ndarray, shape (3,), optional
        Non-zero vector in 3-space. Does not have to be of unit length.
    degrees : bool, optional
        Convert result from radians to degrees.

    Returns
    -------
    angle : float
        Angle in degrees or radians.

    Notes
    -----
    If the `up` vector is defined the sign of the angle is determined
    via the right-hand rule, i.e., it is positive if the cross product
    of `u` and `v` (in this order) points in the same direction as `up`.
    """
    # If either u or v is the zero vector, the division results in NaN.
    # In this case clamp() will complain.
    angle = math.acos(clamp(u.dot(v) / (norm(u) * norm(v)), -1.0, 1.0))

    if degrees:
        angle = math.degrees(angle)

    # Angle is a value between 0 and pi (or 0 and 180). We can define a
    # sign by specifying a third vector.
    if up is not None and up.dot(cross(u, v)) < 0.0:
        angle *= -1.0

    return angle


def cotan(u, v):
    r""" Cotangent between vectors.

    Computes :math:`\operatorname{cotan}(\alpha) =
    \mathbf{u}^T \mathbf{v} / \| \mathbf{u} \times \mathbf{v} \|` where
    :math:`\alpha = \angle(\mathbf{u}, \mathbf{v})`.

    Parameters
    ----------
    u, v : ndarray, shape (3,)
        Vectors in 3-space.

    Returns
    -------
    cot : float
        Cotangent of the angle between vectors `u` and `v`. If at least
        one input vector is the zero vector the returned value is
        :obj:`~numpy.nan`. If input vectors are linearly dependent but
        neither the zero vector, the returned value is :obj:`~numpy.inf`.
    """
    # By definition we also have cot = 1.0 / math.tan(angle(u, v)). The
    # obtained values agree up to around 10 decimal places.
    return u.dot(v) / norm(cross(u, v))


def clamp(x, lo, hi):
    """ Clamp value to range.

    Clamp `x` to the closed interval [`lo`, `hi`].

    Parameters
    ----------
    x : float
        Value to clamp. Assumed finite, in particular it may not be
        :obj:`~numpy.nan`.
    lo, hi : float
        Lower and upper bound. Assumes `lo` <= `hi`, negative and positive
        :obj:`~numpy.inf` are allowed.

    Returns
    -------
    float
        Clamped value.

    Notes
    -----
    To prevent data type changes arguments should not mix int and float
    values, i.e., use

    >>> clamp(x, -1.0, 1.0)

    if `x` is a floating point value, not ``clamp(x, -1, 1)``.
    """
    assert not math.isnan(lo) and not math.isnan(hi)
    assert lo <= hi
    assert math.isfinite(x)

    # When clamping happens, the data type changes if not all three
    # arguments are of the same type - might induce hard to track bugs.
    # x = lo if x < lo else x
    # x = hi if x > hi else x

    # return x

    # Alternative implementation. Data types may still change unless
    # equal to begin with. The order of arguments should guarantee that
    # the data type does not change if x is within bounds.
    return max(min(x, hi), lo)


def cramer(A, b):
    r""" Cramer's rule.

    .. version-added:: 1.1.0

    Solve the system :math:`Ax = b` using Cramer's rule. NumPy's general
    purpose method :func:`~numpy.linalg.solve` is faster.

    Parameters
    ----------
    A : ndarray, shape (3, 3)
        Left-hand side matrix.
    b : ndarray, shape (3,)
        Right-hand side vector.

    Returns
    -------
    x : ndarray, shape (3,)
        Solution vector.

    Notes
    -----
    As of now this function can fail without warning, regularity of `A` is
    not checked.
    """
    def det3(a, b, c):
        return (a[0]*b[1]*c[2] + b[0]*c[1]*a[2] + c[0]*a[1]*b[2]
                - a[2]*b[1]*c[0] - b[2]*c[1]*a[0] - c[2]*a[1]*b[0])

    a0, a1, a2 = A.T

    sol = np.empty(3)
    det = det3(a0, a1, a2)

    sol[0] = det3(b, a1, a2) / det
    sol[1] = det3(a0, b, a2) / det
    sol[2] = det3(a0, a1, b) / det

    return sol


def cross(u, v):
    """ Cross product.

    Alternative to NumPy's vectorized :func:`~numpy.cross` function.

    Parameters
    ----------
    u, v : array_like, shape (3,)
        Vectors in 3-space.

    Returns
    -------
    ndarray, shape (3,)
        Cross product of vectors `u` and `v`.

    See Also
    --------
    cross_mat, numpy.cross, numpy.linalg.cross
    """
    # Unpack the arrays. This will also catch any problem with array shape.
    u0, u1, u2 = u
    v0, v1, v2 = v

    return np.array([u1*v2 - u2*v1,
                     u2*v0 - u0*v2,
                     u0*v1 - u1*v0])


def cross_mat(u):
    r""" Cross product matrix.

    Let :math:`A` be the matrix returned by this method, then
    :math:`A \mathbf{x} = \mathbf{u} \times \mathbf{x}`.

    Parameters
    ----------
    u : array_like, shape (3,)
        Vector in 3-space.

    Returns
    -------
    A : ndarray, shape (3, 3)
        Cross product matrix (always skew symmetric).
    """
    # Unpack the array. This will also catch any problem with array shape.
    u0, u1, u2 = u

    return np.array([[0.0, -u2, u1],
                     [u2, 0.0, -u0],
                     [-u1, u0, 0.0]])


def norm(u):
    """ Length of vector.

    Alternative to NumPy's vectorized :func:`~numpy.linalg.norm` function.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Vector with n components.

    Returns
    -------
    float
        Euclidean length of the vector `u`.

    See Also
    --------
    norm_sqrd, numpy.linalg.norm
    """
    return math.sqrt(u.dot(u))


def norm_sqrd(u):
    """ Squared length of vector.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Vector with n components.

    Returns
    -------
    float
        Squared Euclidean length of the vector `u`, i.e., the
        inner product of `u` with itself.

    See Also
    --------
    numpy.inner, numpy.vecdot
    """
    return u.dot(u)


def unit(u):
    """ Vector normalization.

    Convenience function to normalize a vector.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Vector with n components.

    Returns
    -------
    ndarray, shape (n,)
        Normalized copy of input vector. The input vector `u` remains
        unchanged.

    See Also
    --------
    unit_inplace

    Notes
    -----
    No error checking (division by zero, etc) is performed. Trying to
    normalize the zero vector results in a vector of :obj:`~numpy.nan`
    values.
    """
    return u / norm(u)


def unit_inplace(u):
    """ In-place vector normalization.

    Convenience function to normalize a vector.

    Parameters
    ----------
    u : ndarray, shape (n,)
        Vector with n components.

    Returns
    -------
    ndarray, shape (n,)
        The normalized input vector (not a normalized copy).

    Warnings
    --------
    This function modifies the input argument!

    Notes
    -----
    No error checking (division by zero, etc) is performed. Trying to
    normalize the zero vector results in a vector of :obj:`~numpy.nan`
    values.
    """
    u /= norm(u)
    return u


def rank(A):
    """ Matrix rank.

    Alternative implementation of NumPy's :func:`~numpy.linalg.matrix_rank`.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Matrix without any shape restriction.

    See Also
    --------
    :func:`~scipy.linalg.null_space`
    """
    s = np.linalg.svd(A)[1]
    m, n = A.shape
    tol = s.max() * max(m, n) * np.finfo(s.dtype).eps

    return sum(s > tol)


def rotate(x, axis, phi, sin_phi=None):
    r""" Rotate vector about axis.

    A rotation is defined by an oriented axis vector of unit length and
    an oriented angle. The corresponding rotation can be performed via
    Rodrigues' rotation formula:

    .. math::

       \mathbf{x}_{\text{rot}} = \mathbf{x} \cos(\varphi) +
            (1-\cos(\varphi)) \mathbf{a} \mathbf{a}^T \mathbf{x} +
            (\mathbf{a} \times \mathbf{x}) \sin(\varphi).

    This corresponds to a positive (counter-clockwise) rotation for
    :math:`\varphi > 0` when looking on the normal plane of the axis
    vector pointing towards the viewer (right-hand rule).

    Parameters
    ----------
    x : ndarray, shape (3, )
        Vector to be rotated.
    axis : ndarray, shape (3, )
        Unit length axis vector.
    phi : float
        Rotation angle in radians or the value :math:`\cos(\varphi)`,
        see notes.
    sin_phi : float, optional
        The value :math:`\sin(\varphi)`.

    Returns
    -------
    x_rot : ndarray
        The rotated vector.

    See Also
    --------
    rotation

    Notes
    -----
    Computation of :math:`\sin(\varphi)` can be avoid by providing this
    value as the optional argument `sin_phi`. In this case `phi` is
    assumed to hold :math:`\cos(\varphi)` and no trigonometric functions
    are evaluated.
    """
    if sin_phi is not None:
        cphi = phi
        sphi = sin_phi
    else:
        cphi = math.cos(phi)
        sphi = math.sin(phi)

    return (x * cphi
            + axis * axis.dot(x) * (1.0 - cphi)
            + cross(axis, x) * sphi)


def rotation(u, v, axis):
    r""" Rotation parameters.

    Computes the values :math:`\cos(\varphi)` and :math:`\sin(\varphi)`
    of the rotation about `axis` that aligns the vectors `u` and `v` in
    the sense that `u` and `v` become parallel and point in the same
    direction. This assumes that `u` and `v` are orthogonal to `axis`.
    If this is not the case the vectors are first projected onto
    the plane perpendicular to `axis`.

    Parameters
    ----------
    u, v : ndarray, shape (3,)
        Non-zero vectors in 3-space.
    axis : ndarray, shape (3,)
        Unit length vector in 3-space not contained in the span
        of `u` and `v`.

    Returns
    -------
    cos_phi : float
        Cosine of rotation angle.
    sin_phi : float
        Sine of rotation angle.

    See Also
    --------
    rotate

    Notes
    -----
    Assume that vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` are
    orthogonal to :math:`\mathbf{a}`. If this is not the case set
    :math:`\mathbf{x} = \mathbf{x} - \mathbf{a}\mathbf{a}^T \mathbf{x}`
    for :math:`\mathbf{x} \in \{ \mathbf{u}, \mathbf{v} \}`. There is
    :math:`\mu > 0` such that

    >>> v = mu * rotate(u, a, *rotation(u, v, a))
    """
    # Projection of u and v into the plane orthogonal to axis. The axis
    # vector may not be in the span of u and v.
    u = u - axis * axis.dot(u)
    v = v - axis * axis.dot(v)

    # Normalization necessary before computing inner products.
    u /= norm(u)
    v /= norm(v)

    # Since u and v are orthogonal to axis, this vector is parallel to
    # axis. Inner product with the axis determines the sign of the angle.
    normal = cross(u, v)

    cos_alpha = clamp(u.dot(v), -1.0, 1.0)
    sin_alpha = norm(normal)

    if normal.dot(axis) < 0.0:
        sin_alpha *= -1.0

    return cos_alpha, sin_alpha


def rotation_from_quaternion(a):
    """ Rotation matrix.

    Convert unit quaternion to corresponding rotation matrix, see e.g. [1]_.

    Parameters
    ----------
    a : ndarray, shape (4,)
        Unit quaternion.

    Returns
    -------
    A : ndarray, shape (3, 3)
        Rotation matrix.

    References
    ----------
    .. [1] B. Horn: *Closed-form solution of absolute orientation using
           unit quaternions*, J. Opt. Soc. Am. A 4, 629-642, 1987.
    """
    # If a[0] = 1.0 or a[0] = -1.0 we get the (3, 3) identity matrix. We
    # could catch this case and explicitly return np.eye(3).

    a00 = a[0] * a[0]
    a01 = a[0] * a[1]
    a02 = a[0] * a[2]
    a03 = a[0] * a[3]

    a11 = a[1] * a[1]
    a12 = a[1] * a[2]
    a13 = a[1] * a[3]

    a22 = a[2] * a[2]
    a23 = a[2] * a[3]

    a33 = a[3] * a[3]

    return np.array(
        [[a00 + a11 - a22 - a33, 2.0 * (a12 - a03), 2.0 * (a13 + a02)],
         [2.0 * (a12 + a03), a00 - a11 + a22 - a33, 2.0 * (a23 - a01)],
         [2.0 * (a13 - a02), 2.0 * (a23 + a01), a00 - a11 - a22 + a33]])
