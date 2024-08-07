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

""" Basic vector math.

For basic computations, specialized non-vectorized functions offer better
performance than some NumPy functions. This seems to be the case for
:func:`numpy.cross`, at least in some older releases.
"""

import math
import numpy as np


def cross(u, v):
    """ Cross product.

    Alternative to NumPy's vectorized function :func:`numpy.cross` of the
    same name.

    Parameters
    ----------
    u : array_like, shape (3, )
        Vector in :math:`\mathbb{R}^3`.
    v : array_like, shape (3, )
        Vector in :math:`\mathbb{R}^3`.

    Returns
    -------
    ~numpy.ndarray, shape (3, )
        Cross product of vectors `u` and `v`.

    Note
    ----
    NumPy v2.0 introduced :func:`numpy.linalg.cross` which may offer
    better performance.
    """
    return np.array([u[1]*v[2] - u[2]*v[1],
                     u[2]*v[0] - u[0]*v[2],
                     u[0]*v[1] - u[1]*v[0]])


def dot(u, v):
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
        Inner product :math:`\mathbf{u}^T \mathbf{v}`.

    Note
    ----
    The fastest way to do dot products seems to be using
    :meth:`numpy.ndarray.dot` instead of the function :func:`numpy.dot`.
    """
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def norm(u):
    """ Length of vector.

    Alternative to NumPy's :func:`numpy.linalg.norm`, specialized for
    the case of vectors in 3-space.

    Parameters
    ----------
    u : array_like, shape (3, )
        Vector in :math:`\mathbb{R}^3`.

    Returns
    -------
    float
        Euclidean length of the vector `u`.

    Note
    ----
    The length of the input vector is not checked. Passing vectors with
    more than three entries will produce a **wrong** result without any
    warning.
    """
    return math.sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])


def unit(u):
    """ In-place vector normalization.

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
    No error checking (division by zero, input vector shape) is
    performed.
    """
    u /= norm(u)
    return u


def rotate(x, a, phi, sin_phi=None):
    r""" Rotate vector around axis.

    Rotation is performed via Rodrigues' rotation formula,

    .. math::

       \mathbf{x}' = \mathbf{x} \cos(\varphi) +
            (1-\cos(\varphi)) \mathbf{a} \mathbf{a}^T \mathbf{x} +
            (\mathbf{a} \times \mathbf{x}) \sin(\varphi)

    which results in a positive (counter-clockwise) rotation for
    :math:`\varphi > 0` when looking on the normal plane of the axis
    vector pointing towards the viewer (right hand rule).

    Parameters
    ----------
    x : ~numpy.ndarray, shape (3, )
        Vector to be rotated.
    a : ~numpy.ndarray, shape (3, )
        Normalized axis vector.
    phi : float
        Rotation angle in radians or :math:`\cos(\varphi)`.
    sin_phi : float, optional
        :math:`\sin(\varphi)`.

    Returns
    -------
    ~numpy.ndarray
        The rotated vector.

    Note
    ----
    Evaluation of trigonometric functions can be avoid by providing
    the optional argument `sin_phi`. In this case `phi` is interpreted
    as :math:`\cos(\varphi)`.
    """
    if sin_phi is not None:
        cphi = phi
        sphi = sin_phi
    else:
        cphi = math.cos(phi)
        sphi = math.sin(phi)

    return x * cphi + a * a.dot(x) * (1.0 - cphi) + cross(a, x) * sphi