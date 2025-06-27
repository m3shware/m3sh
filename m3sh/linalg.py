# Copyright 2024-25, m3shware
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

""" Basic vector math.
"""

import math
import numpy as np


def angle(v, w, a=None, deg=False):
    r""" Angle between vectors.

    Angle between vectors :math:`\mathbf{v}` and :math:`\mathbf{w}` in
    radians. To obtain an oriented angle an axis vector :math:`\mathbf{a}`
    has to be specified. Typically the axis vector is a vector parallel
    to the cross product vector :math:`\mathbf{v} \times \mathbf{w}` with
    the desired orientation.

    Parameters
    ----------
    v, w : ~numpy.ndarray, shape (3, )
        Vector in 3-space.
    a : ~numpy.ndarray, shape (3, ), optional
        Axis vector in 3-space.
    deg : bool, optional
        Convert result from radians to degrees.

    Returns
    -------
    float
        Angle in degrees or radians.

    Note
    ----
    If an axis vector is defined the sign is determined via the right-hand
    rule. None of the vectors may be the zero vector. The axis vector may
    not be contained in the span of :math:`\mathbf{v}` and :math:`\mathbf{w}`.
    """
    # Yields a value between 0 and pi. We can define a sign by specifying
    # an axis vector, see below.
    angle = math.acos(clamp(v.dot(w) / (norm(v) * norm(w)), -1., 1.))

    if deg:
        angle = math.degrees(angle)

    if a is not None and a.dot(cross(v, w)) < 0.0:
        angle *= -1.0

    return angle


def clamp(x, lo, hi):
    """ Clamp value to range.

    Clamp `x` to the closed interval [`lo`, `hi`].

    Parameters
    ----------
    x : float
        Value to clamp.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    float
        Clamped value.

    Note
    ----
    To prevent data type changes arguments should not mix :class:`int` and
    :class:`float` values.
    """
    # assert type(x) is type(lo) and type(x) is type(hi)
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
        Vector in :math:`\mathbb{R}^3`.

    Returns
    -------
    ~numpy.ndarray, shape (3, )
        Cross product of vectors :math:`\mathbf{u}` and :math:`\mathbf{v}`.
    """
    # Unpack the arrays. This will also catch any problem with array shape.
    u0, u1, u2 = u
    v0, v1, v2 = v

    return np.array([u1*v2 - u2*v1,
                     u2*v0 - u0*v2,
                     u0*v1 - u1*v0])

    # return np.array([u[1]*v[2] - u[2]*v[1],
    #                  u[2]*v[0] - u[0]*v[2],
    #                  u[0]*v[1] - u[1]*v[0]])


def cross_mat(u):
    r""" Cross product matrix.

    Let :math:`A` be the matrix returned by this method, then
    :math:`A \mathbf{x} = \mathbf{u} \times \mathbf{x}`.

    Parameters
    ----------
    u : array_like, shape (3, )
        Vector in :math:`\mathbb{R}^3`.

    Returns
    -------
    ~numpy.ndarray, shape (3, 3)
        Cross product matrix.

    Note
    ----
    The matrix :math:`A` is skew symmetric.
    """
    # Unpack the array. This will also catch any problem with array shape.
    u0, u1, u2 = u

    return np.array([[0.0, -u2, u1],
                     [u2, 0.0, -u0],
                     [-u1, u0, 0.0]])


def _dot(u, v):
    r""" Dot product.

    Inner product of vectors. Alternative to :func:`numpy.dot` for
    3-dimensional vectors.

    Parameters
    ----------
    u : array_like, shape (3, )
        Vector in :math:`\mathbb{R}^3`.
    v : array_like, shape (3, )
        Vector in :math:`\mathbb{R}^3`.

    Returns
    -------
    float
        Inner product of vectors `u` and `v`.

    Note
    ----
    The fastest way to do dot products seems to be using
    :meth:`numpy.ndarray.dot` instead of the function :func:`numpy.dot`.
    """
    # There seems to be no real advantage of using this over ndarray.dot().
    # The vedcot and linalg.vecdot functions are slower than ndarray.dot().
    # All those considerations are only valid when doing non-vectorized
    # operations!
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def norm(u):
    r""" Length of vector.

    Alternative to NumPy's vectorized :func:`~numpy.linalg.norm` function.

    Parameters
    ----------
    u : array_like, shape (n, )
        Vector in :math:`\mathbb{R}^n`.

    Returns
    -------
    float
        Euclidean length of the vector :math:`\mathbf{u}`.
    """
    # return math.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
    return math.sqrt(u.dot(u))


def unit_inplace(u):
    r""" In-place vector normalization.

    Convenience function to normalize a vector. Modifies the input
    argument!

    Parameters
    ----------
    u : ~numpy.ndarray, shape (3, )
        Vector in :math:`\mathbb{R}^3`.

    Returns
    -------
    ~numpy.ndarray, shape (3, )
        The normalized input vector (not a normalized copy).

    Note
    ----
    No error checking (division by zero, input vector shape) is performed.
    """
    u /= norm(u)
    return u


def unit(u):
    r""" Vector normalization.

    Convenience function to normalize a vector.

    Parameters
    ----------
    u : ~numpy.ndarray, shape (3, )
        Vector in :math:`\mathbb{R}^3`.

    Returns
    -------
    ~numpy.ndarray, shape (3, )
        Normalized copy of input vector.

    Note
    ----
    No error checking (division by zero, input vector shape) is performed.
    """
    return u / norm(u)


def rotate(x, a, phi, sinphi=None):
    r""" Rotate vector about axis.

    Rotation is performed via Rodrigues' rotation formula,

    .. math::

       \mathbf{x}_{\text{rot}} = \mathbf{x} \cos(\varphi) +
            (1-\cos(\varphi)) \mathbf{a} \mathbf{a}^T \mathbf{x} +
            (\mathbf{a} \times \mathbf{x}) \sin(\varphi)

    which results in a positive (counter-clockwise) rotation for
    :math:`\varphi > 0` when looking on the normal plane of the axis
    vector pointing towards the viewer (right-hand rule).

    Parameters
    ----------
    x : ~numpy.ndarray, shape (3, )
        Vector to be rotated.
    a : ~numpy.ndarray, shape (3, )
        Normalized axis vector.
    phi : float
        Rotation angle in radians or :math:`\cos(\varphi)`.
    sinphi : float, optional
        :math:`\sin(\varphi)`.

    Returns
    -------
    ~numpy.ndarray
        The rotated vector.

    Note
    ----
    Computation of :math:`\sin(\varphi)` can be avoid by providing this
    value as the optional argument `sinphi`. In this case `phi` is taken
    to be :math:`\cos(\varphi)` and no trigonometric functions are evaluated.
    """
    if sinphi is not None:
        cphi = phi
        sphi = sinphi
    else:
        cphi = math.cos(phi)
        sphi = math.sin(phi)

    return x * cphi + a * a.dot(x) * (1.0 - cphi) + cross(a, x) * sphi


def rotation(v, w, a):
    r""" Rotation parameters.

    Computes the values :math:`\cos(\varphi)` and :math:`\sin(\varphi)`
    of the rotation about the axis :math:`\mathbf{a}` that aligns two
    vectors :math:`\mathbf{v}` and :math:`\mathbf{w}` of equal norm, i.e.,
    using  Rodrigues' rotation formula we get

    .. math::

       \mathbf{w} = \mathbf{v} \cos(\varphi) +
            (1-\cos(\varphi)) \mathbf{a} \mathbf{a}^T \mathbf{v} +
            (\mathbf{a} \times \mathbf{v}) \sin(\varphi).

    Parameters
    ----------
    v, w : ~numpy.ndarray, shape (3, )
        Vector in 3-space.
    a : ~numpy.ndarray, shape (3, )
        Unit vector in 3-space.

    Returns
    -------
    cosphi : float
        Cosine of rotation angle.
    sinphi : float
        Sine of rotation angle.

    Note
    ----
    The length preconditions on :math:`\mathbf{v}` and :math:`\mathbf{w}` are
    not checked. If :math:`\| \mathbf{v} \| \neq \| \mathbf{w} \|` the formula

    .. math::

       \lambda \mathbf{w} = \mathbf{v} \cos(\varphi) +
            (1-\cos(\varphi)) \mathbf{a} \mathbf{a}^T \mathbf{v} +
            (\mathbf{a} \times \mathbf{v}) \sin(\varphi)

    holds with :math:`\lambda = \| \mathbf{v} \| / \| \mathbf{w} \|`.
    """
    # Projection of v and w into the plane orthogonal to the axis a. The
    # axis vector may not be in the span of v and w.
    v = v - a * a.dot(v)
    w = w - a * a.dot(w)

    # Normalization necessary before computing inner products.
    v /= norm(v)
    w /= norm(w)

    # A vector parallel to the axis. Inner product with the axis determines
    # the sign of the angle.
    n = cross(v, w)

    cos_alpha = clamp(v.dot(w), -1.0, 1.0)
    sin_alpha = norm(n)

    if n.dot(a) < 0.0:
        sin_alpha *= -1.0

    return cos_alpha, sin_alpha
