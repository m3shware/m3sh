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

Basic vector math, linear algebra, and analytic geometry (in particular
different representations of rotations). Many function are convenience
functions that wrap NumPy functionality.
"""

import math
import numpy as np


def angle(u, w, up=None, degrees=False):
    r""" Angle between vectors.

    Angle between vectors `u` and `w` in radians. To obtain an oriented
    angle the `up` vector has to be specified. If specified, `up` must not
    be contained in the span of `u` and `w`.

    Parameters
    ----------
    u, w : ndarray, shape (3,)
        Vectors in 3-space.
    up : ndarray, shape (3,), optional
        Vector in 3-space.
    degrees : bool, optional
        Convert result from radians to degrees.

    Returns
    -------
    angle : float
        Angle in degrees or radians.

    Notes
    -----
    If the `up` vector is defined the sign is determined via the right-hand
    rule, i.e., it is positive if the cross product of `u` and `w` (in this
    order) points in the same direction as `up`.
    """
    # Yields a value between 0 and pi. We can define a sign by specifying
    # a third vector.
    angle = math.acos(clamp(u.dot(w) / (norm(u) * norm(w)), -1.0, 1.0))

    if degrees:
        angle = math.degrees(angle)

    # For linear dependent vectors v and w we either get 0.0 or pi. There
    # is no point in introducing -pi!
    if up is not None and up.dot(cross(u, w)) < 0.0:
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
    float
        Cotangent of the angle between vectors `u` and `v`.

    Notes
    -----
    Division by zero produces :obj:`~numpy.inf` or :obj:`~numpy.nan`
    (depending on the value of the numerator).
    """
    return u.dot(v) / norm(cross(u, v))


def clamp(x, lo, hi):
    """ Clamp value to range.

    Clamp `x` to the closed interval [`lo`, `hi`].

    Parameters
    ----------
    x : float
        Value to clamp.
    lo, hi : float
        Lower and upper bound.

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
    assert lo <= hi

    # When clamping happens, the data type changes if not all three
    # arguments are of the same type - might induce hard to track bugs.
    # x = lo if x < lo else x
    # x = hi if x > hi else x

    # return x

    # Alternative implementation. Data types may still change unless
    # equal to begin with. The order of arguments should guarantee that
    # the data type does not change if x is within bounds.
    return max(min(x, hi), lo)


def cross(u, v):
    r""" Cross product.

    Alternative to NumPy's vectorized :func:`~numpy.cross` function.

    Parameters
    ----------
    u, v : array_like, shape (3, )
        Vectors in 3-space.

    Returns
    -------
    ndarray, shape (3, )
        Cross product of vectors `u` and `v`.

    See Also
    --------
    cross_mat
    """
    # Unpack the arrays. This will also catch any problem with array shape.
    u0, u1, u2 = u
    v0, v1, v2 = v

    return np.array([u1*v2 - u2*v1,
                     u2*v0 - u0*v2,
                     u0*v1 - u1*v0])


def cross_mat(u):
    r""" Cross product matrix.

    Let A be the matrix returned by this method, then
    ``A @ x == cross(u, x)``.

    Parameters
    ----------
    u : array_like, shape (3, )
        Vector in 3-space.

    Returns
    -------
    ndarray, shape (3, 3)
        Cross product matrix (always skew symmetric).
    """
    # Unpack the array. This will also catch any problem with array shape.
    u0, u1, u2 = u

    return np.array([[0.0, -u2, u1],
                     [u2, 0.0, -u0],
                     [-u1, u0, 0.0]])


def norm(u):
    r""" Length of vector.

    Alternative to NumPy's vectorized :func:`~numpy.linalg.norm` function.

    Parameters
    ----------
    u : ndarray, shape (n, )
        Vector with n components.

    Returns
    -------
    float
        Euclidean length of the vector `u`.

    See Also
    --------
    norm_sqrd
    """
    return math.sqrt(u.dot(u))


def norm_sqrd(u):
    r""" Squared length of vector.

    Parameters
    ----------
    u : ndarray, shape (n, )
        Vector with n components.

    Returns
    -------
    float
        Squared Euclidean length of the vector `u`.
    """
    return u.dot(u)


def unit(u):
    r""" Vector normalization.

    Convenience function to normalize a vector.

    Parameters
    ----------
    u : ndarray, shape (n, )
        Vector with n components.

    Returns
    -------
    ndarray, shape (n, )
        Normalized copy of input vector. The input vector `u` remains
        unchanged.

    See Also
    --------
    unit_inplace

    Notes
    -----
    No error checking (division by zero, etc) is performed.
    """
    return u / norm(u)


def unit_inplace(u):
    r""" In-place vector normalization.

    Convenience function to normalize a vector. Modifies the input
    argument!

    Parameters
    ----------
    u : ndarray, shape (n, )
        Vector with n components.

    Returns
    -------
    ndarray, shape (n, )
        The normalized input vector (not a normalized copy).

    Notes
    -----
    No error checking (division by zero, etc) is performed.
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


def rotate(x, a, phi, sinphi=None):
    r""" Rotate vector about axis.

    Rotation is performed via Rodrigues' rotation formula,

    .. math::

       \mathbf{x}_{\text{rot}} = \mathbf{x} \cos(\varphi) +
            (1-\cos(\varphi)) \mathbf{a} \mathbf{a}^T \mathbf{x} +
            (\mathbf{a} \times \mathbf{x}) \sin(\varphi)

    which results in a positive (counter-clockwise) rotation for
    :math:`\varphi > 0` when looking on the normal plane of the axis
    vector :math:`a` pointing towards the viewer (right-hand rule).

    Parameters
    ----------
    x : ndarray, shape (3, )
        Vector to be rotated.
    a : ndarray, shape (3, )
        Unit length axis vector.
    phi : float
        Rotation angle in radians or the value :math:`\cos(\varphi)`,
        see notes.
    sinphi : float, optional
        The value :math:`\sin(\varphi)`.

    Returns
    -------
    ndarray
        The rotated vector.

    See Also
    --------
    rotation

    Notes
    -----
    Computation of :math:`\sin(\varphi)` can be avoid by providing this
    value as the optional argument `sinphi`. In this case `phi` is assumed
    to hold :math:`\cos(\varphi)` and no trigonometric functions are
    evaluated.
    """
    if sinphi is not None:
        cphi = phi
        sphi = sinphi
    else:
        cphi = math.cos(phi)
        sphi = math.sin(phi)

    return x * cphi + a * a.dot(x) * (1.0 - cphi) + cross(a, x) * sphi


def rotation(u, w, a):
    r""" Rotation parameters.

    Computes the values :math:`\cos(\varphi)` and :math:`\sin(\varphi)`
    of the rotation about the axis :math:`\mathbf{a}` that aligns the
    vectors :math:`\mathbf{u}^{\bot}` and :math:`\mathbf{w}^{\bot}`,
    see the notes for more details.

    Parameters
    ----------
    u, w : ndarray, shape (3, )
        Vectors in 3-space.
    a : ndarray, shape (3, )
        Unit length vector in 3-space.

    Returns
    -------
    cosphi : float
        Cosine of rotation angle.
    sinphi : float
        Sine of rotation angle.

    See Also
    --------
    rotate

    Notes
    -----
    Let :math:`\mathbf{x}^{\bot} = \mathbf{x} - \mathbf{a}\mathbf{a}^T
    \mathbf{x}`. In particular the vectors :math:`\mathbf{u}^{\bot}` and
    :math:`\mathbf{w}^{\bot}` are perpendicular to :math:`\mathbf{a}`. If
    :math:`\mathbf{u}` and :math:`\mathbf{w}` are two vectors of equal
    length and perpendicular to :math:`\mathbf{a}` then

    >>> w = rotate(u, a, *rotation(u, w, a))
    """
    # Projection of v and w into the plane orthogonal to the axis a. The
    # axis vector may not be in the span of v and w.
    u = u - a * a.dot(u)
    w = w - a * a.dot(w)

    # Normalization necessary before computing inner products.
    u /= norm(u)
    w /= norm(w)

    # A vector parallel to the axis. Inner product with the axis determines
    # the sign of the angle.
    n = cross(u, w)

    cos_alpha = clamp(u.dot(w), -1.0, 1.0)
    sin_alpha = norm(n)

    if n.dot(a) < 0.0:
        sin_alpha *= -1.0

    return cos_alpha, sin_alpha


def rotation_from_quaternion(a):
    """ Rotation matrix.

    Convert unit quaternion to corresponding rotation matrix, see e.g. [1]_.

    Parameters
    ----------
    a : ndarray, shape (4, )
        Unit quaternion, i.e., ``norm(a) = 1``.

    Returns
    -------
    A : ndarray, shape (3, 3)
        Rotation matrix.

    References
    ----------
    .. [1] B. Horn: "Closed-form solution of absolute orientation using
           unit quaternions", J. Opt. Soc. Am. A 4, 629-642, 1987.
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