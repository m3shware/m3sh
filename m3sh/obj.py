# Copyright 2024-2026, m3shware developers
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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

""" Wavefront object file input/output.

Low-level functions to read and write .obj files. Only a subset of the
object file format standard is supported. Complete specifications can be
found `here <https://paulbourke.net/dataformats/obj/>`_.
"""

from io import StringIO

import numpy as np
# import scipy as sp


class CSType:
    """ Wrapper class for **cstype** statements.

    Represents a subset of the freeform curve and surface statements found in
    the Alias/Wavefront OBJ specification, see
    `here <https://en.wikipedia.org/wiki/Wavefront_.obj_file>`_ or
    `Appendix B <http://fegemo.github.io/cefet-cg/attachments/obj-spec.pdf>`_
    of the `Advanced Visualizer Manual` for details. This class serves as a
    link between Wavefront OBJ files and application code. Supports import and
    export of curves and surfaces in Bézier and B-spline representation.

    Parameters
    ----------
    parm : tuple[list[float]]
        Knot vector(s).
    inds : list[ndarray]
        Control points indices (absolute, 0-based).
    degree : tuple[int]
        Degree.
    type : string
        Either
    rational : bool
        Indicates that the last entry of control point coordinates
        serves as a weight.

    Raises
    ------
    ValueError
        If invalid values for ``cstype`` are given.

    Note
    ----
    Control points of rational curves and surface are of the form
    :math:`(x, y, z, w)` where :math:`(x, y, z)` are Euclidian coordinates
    and :math:`w` acts as a weight. The Wavefront OBJ specification
    does not prohibit the use of negative or zero weights.
    """

    # The preferred way of initializing :py:class:`CSType` objects is by
    # using the :py:meth:`read` or the :py:meth:`bezier` and :py:meth:`bspline`
    # member functions. The script

    # .. literalinclude:: ../../examples/cstype_io.py
    #    :lines: 8-

    # results in the following output:

    # .. code-block:: none

    #    v 0.0 0.0 0.0
    #    v 1.0 0.0 1.0
    #    v 1.0 1.0 2.0
    #    v 0.0 1.0 3.0
    #    v 0.0 0.0 4.0
    #    cstype bspline
    #    deg 3
    #    curv 0.0 2.0 -5 -4 -3 -2 -1
    #    parm u 0.0 0.0 0.0 0.0 1.0 2.0 2.0 2.0 2.0
    #    end

    def __init__(self, parm, inds, degree, type='bspline', rational=False):
        if type in {'bezier', 'bspline'}:
            self._type = type
        else:
            raise ValueError(f'invalid basis type {type}')

        # For curves parm and deg are supposed to be 1-element tuples (or
        # lists). For surfaces parm and deg are supposed to be 2-element
        # tuples (or lists).
        assert len(parm) == len(degree)
        assert 0 < len(degree) < 3

        match type:
            case 'bezier':
                if len(degree) == 1:
                    m, n = len(parm[0]) - 1, len(inds) - 1

                    if n != degree[0] * m:
                        raise ValueError('invalid')
                elif len(degree) == 2:
                    m = (len(parm[0]) - 1, len(parm[1]) - 1)
                    n = (m[0] * degree[0], m[1] * degree[1])

                    if (n[0] + 1) * (n[1] + 1) != len(inds):
                        raise ValueError('invalid')
            case 'bspline':
                if len(degree) == 1:
                    if len(parm[0]) - degree[0] - 1 != len(inds):
                        raise ValueError('invalid curve definition')
                elif len(degree) == 2:
                    n = (len(parm[0]) - degree[0] - 1,
                         len(parm[1]) - degree[1] - 1)

                    if n[0] * n[1] != len(inds):
                        raise ValueError('invalid surface definition')

        self._rational = bool(rational)
        self._parm = parm
        self._inds = inds
        self._degree = degree

    def __repr__(self):
        return (f"{type(self).__name__}(parm={self._parm!r}, " +
                f"inds={self._inds!r}, degree={self._degree}, " +
                f"type={self.type!r}, rational={self._rational!r})")

    def __str__(self):
        with StringIO() as stream:
            str = self._to_stream(stream).getvalue()

        return str

    @staticmethod
    def bezier(b, *args):
        r""" Initialize Bézier curve/surface.

        Initializes a Bézier curve/surface from its control point array.

        Parameters
        ----------
        b : ~numpy.ndarray, shape (n+1, d)
            Control points of a Bézier curve/surface. For a surface
            shape should be (m+1, n+1, d), see below for more details.
        *args
            Variable length argument list. Pass ``'func'`` to disambiguate the
            case of curves and functional surfaces. Pass ``'rational'`` or
            ``'rat'`` to indicate that the last of the d coordinate entries of
            each control point is a weight.

        Returns
        -------
        CSType
            Corresponding Bézier curve/surface representation.


        ----

        **Curves.** Let :math:`n \geq 0`. Control points :math:`\mathbf{b}_0,
        \dots, \mathbf{b}_{n} \in \mathbb{R}^d` define a Bézier curve

        .. math::

            \mathbf{b}(u) = \sum_{i=0}^{n} \mathbf{b}_i B_i^n(u)

        of degree :math:`n` over the interval :math:`[0, 1]`. The control
        points can be passed by packing them into an array ``b`` of shape
        (n+1, d).


        **Surfaces.** Let :math:`m \geq 0` and :math:`n \geq 0`. A tensor
        product Bézier surface of degree :math:`(m, n)` is defined as

        .. math::

            \mathbf{b}(u,v) = \sum_{i=0}^{m} \sum_{j=0}^{n} \mathbf{b}_{ij}
            B_i^m(u) B_j^n(v).

        Control points should be passed as an array ``b`` of shape
        (m+1, n+1, d).

        Note
        ----
        If we are given weights as last component of control points, a
        corresponding rational curve is defined according to

        .. math::

            \mathbf{b}(u) = \\frac{\sum_{i=0}^{n} w_i\mathbf{a}_i B_i^n(u)}
                            {\sum_{i=0}^{n} w_i B_i^n(u)}, \qquad
                            \mathbf{b}_i = (\mathbf{a}_i, w_i)

        and analogously for surfaces.
        """
        cs = CSType('bezier', *args)

        if b.ndim == 1:
            cs._deg = (len(b)-1,)
            cs._vid = [i for i in range(len(b))]
            cs._cps = np.asarray(b)
            cs._rangeu = (0.0, 1.0)
            cs._parmu = (0.0, 1.0)
        elif b.ndim == 3 or (b.ndim == 2 and 'func' in args):
            cs._deg = (b.shape[0]-1, b.shape[1]-1)
            cs._vid = [i for i in range(b.shape[0]*b.shape[1])]
            cs._cps = np.asarray(b)
            cs._rangeu = (0.0, 1.0)
            cs._rangev = (0.0, 1.0)
            cs._parmu = (0.0, 1.0)
            cs._parmv = (0.0, 1.0)
        elif b.ndim == 2:
            cs._deg = (len(b)-1,)
            cs._vid = [i for i in range(len(b))]
            cs._cps = np.asarray(b)
            cs._rangeu = (0.0, 1.0)
            cs._parmu = (0.0, 1.0)
        else:
            msg = ("bezier(): cannot handle control point array of " +
                   "shape " + str(b.shape))
            raise ValueError(msg)

        return cs

    @staticmethod
    def bspline(t, c, k, *args):
        r""" Initialize B-spline curve/surface.

        Initializes a B-spline curve/surface using a (t, c, k) triple
        as in :py:class:`scipy.interpolate.BSpline`.

        Parameters
        ----------
        t : ~numpy.ndarray, shape (2k+n+2,)
            Knot vector of a curve. For a surface a pair of knot vectors
            with shapes (2k+m+2,) and (2l+n+2,) have to be specified.
        c : ~numpy.ndarray, shape (k+n+1, d)
            Control point array. For a surface an array of shape
            (k+m+1, l+n+1, d) is required.
        k : int
            The degree of the curve. For a surface a pair (k, l) of integers
            is required.
        *args
            Variable length argument list. Pass ``'rational'`` to indicate
            that the last of the d coordinate entries of each control point
            is a weight.

        Raises
        ------
        ValueError
            If the values and size given for degree, knots, and control
            points are inconsistent.

        Returns
        -------
        CSType
            Corresponding B-spline curve/surface representation.


        ----

        **Curves.** Let :math:`n \geq 0`. Control points :math:`\mathbf{c}_0,
        \dots, \mathbf{c}_{k+n} \in \mathbb{R}^d` and knots :math:`t_0, \dots,
        t_{2k+n+1} \in \mathbb{R}` such that :math:`t_i \leq t_{i+1}` define
        a B-spline curve

        .. math::

            \mathbf{c}(t) = \sum_{i=0}^{n+k} \mathbf{c}_i N_i^k(t)

        of degree :math:`k` over the interval :math:`[t_k, t_{k+n+1})` if
        :math:`t_i < t_{i+k+1}` for :math:`i \in \{0, \dots, n+k\}`. Those
        values can be passed as arguments by packing the control points into
        an array ``c`` of shape (k+n+2, d) and knots as an array ``t`` of
        shape (2k+n+2,) or as ``list[float]``.

        .. literalinclude:: ../../examples/cstype_io.py
           :lines: 8-21


        The curve :math:`\mathbf{c}` is polynomial over each non empty interval
        :math:`[t_j, t_{j+1})` for :math:`j = k, \dots, k+n`. The integer
        :math:`n` determines the number of polynomial segments of
        :math:`\mathbf{c}`.


        **Surfaces.** Let :math:`m \geq 0` and :math:`n \geq 0`. A tensor
        product B-spline surface of degree
        :math:`(k, l)` over the knot sequences :math:`u_0, \dots, u_{2k+n+1}`
        and :math:`v_0, \dots, v_{2l+m+1}` is defined as

        .. math::

            \mathbf{s}(u,v) = \sum_{i=0}^{m+k} \sum_{j=0}^{n+l} \mathbf{c}_{ij}
            N_i^k(u) N_j^l(v).

        The control points can be stored in an array of shape (k+m+1, l+n+1, d).
        Knot vectors can be passed as a pair of 1-dimensional arrays or a
        pair of ``list[float]``.
        """
        # Create empty CSType object. Attributes are filled using tck.
        cs = CSType('bspline', *args)

        # A curve is defined if k is a single integer. If the condition
        # len(t) = len(c) + k + 1 fails an Exception is triggered.
        if isinstance(k, int):
            if k < 0 or len(t) != len(c) + k + 1:
                msg = ('bspline(): inconsistent degree and shapes of ' +
                       'knot vector and control point array!')
                raise ValueError(msg)

            cs._deg = (k,)
            cs._vid = [i for i in range(len(c))]
            cs._cps = np.asarray(c)
            cs._rangeu = (t[k], t[-k-1])
            cs._parmu = tuple(t)

        # Surface definition. Degree and shape of knots and control points
        # need to be consistent in both dimensions.
        elif len(k) == 2:
            if (k[0] < 0 or len(t[0]) != c.shape[0] + k[0] + 1
                    or k[1] < 0 or len(t[1]) != c.shape[1] + k[1] + 1):
                msg = ('bspline(): inconsistent degree and shapes of ' +
                       'knot vectors and control point array!')
                raise ValueError(msg)

            cs._deg = (k[0], k[1])
            cs._vid = [i for i in range(np.shape(c)[0]*np.shape(c)[1])]
            cs._cps = np.asarray(c)
            cs._rangeu = (t[0][k[0]], t[0][-k[0]-1])
            cs._rangev = (t[1][k[1]], t[1][-k[1]-1])
            cs._parmu = tuple([u for u in t[0]])
            cs._parmv = tuple([v for v in t[1]])

        # Things don't match up...
        else:
            msg = ("bspline(): degree 'k' needs to be an integer " +
                   "value or a pair of integers!")
            raise ValueError(msg)

        # Return the generated curve/surface representation.
        return cs

    @property
    def type(self):
        return self._type

    @property
    def rational(self):
        return self._rational

    @property
    def parameters(self):
        return self._parm

    @property
    def degree(self):
        return self._degree

    @property
    def domain(self):
        parm = self._parm

        match self.type:
            case 'bezier':
                if len(parm) == 2:
                    return ((parm[0][0], parm[0][-1]),
                            (parm[1][0], parm[1][-1]))
                else:
                    return ((parm[0][0], parm[0][-1]), )
            case 'bspline':
                deg = self._degree

                if len(parm) == 2:
                    return ((parm[0][deg[0]], parm[0][-deg[0]-1]),
                            (parm[1][deg[1]], parm[1][-deg[1]-1]))
                else:
                    return ((parm[0][deg[0]], parm[0][-deg[0]-1]), )

    def cparray(self, points):
        """ Control point array.

        Parameters
        ----------
        points : array_like
            Control point coordinates.
        """
        # Assumes that points is the list of all vertex coordinates read
        # from the OBJ file that contained the cstype definition.
        pts = [points[i] for i in self._inds]
        deg = self._degree

        if len(deg) == 2:
            # Control point indices in u-direction: 0, 1, ..., n[0]-1
            # and 0, 1, ..., n[1]-1 in v-direction, hence
            match self.type:
                case 'bezier':
                    n = (deg[0] + 1, deg[1] + 1)
                case 'bspline':
                    n = (len(self._parm[0]) - deg[0] - 1,
                         len(self._parm[1]) - deg[1] - 1)

            # Order of control polygon vertices for deg (3, 2) surface:
            # u-direction is stored as columns, and v-direction as rows.
            # b00 b01 b02   0 4 8
            # b10 b11 b12 = 1 5 9
            # b20 b21 b22   2 6 10
            # b30 b31 b32   3 7 11

            c = np.reshape(pts, (n[1], n[0], -1)).transpose(1, 0, 2)
            f = np.reshape(pts, (n[0], n[1], -1), order='F')

            assert (f == c).all()

            return f
        elif len(deg) == 1:
            return np.array(pts)

        raise ValueError(f'cstype data corruption')

    def _to_stream(self, stream, points=None):
        """ Export to Wavefront OBJ file.

        Parameters
        ----------
        stream : file-like
            Writes the objects OBJ representation to the given file.
        """
        if points is not None:
            pass

        # if self._cps.ndim == 1:
        #     ncps = self._cps.shape[0]
        #     for c in self._cps:
        #         out.write('v ' + str(c) + '\n')
        # elif self._cps.ndim == 2 and len(self._deg) == 2:
        #     ncps = self._cps.shape[0]*self._cps.shape[1]
        #     for j in range(self._cps.shape[1]):
        #         for i in range(self._cps.shape[0]):
        #             out.write('v ')
        #             out.write(str(self._cps[i,j]))
        #             out.write('\n')
        # elif self._cps.ndim == 2:
        #     ncps = self._cps.shape[0]
        #     for c in self._cps:
        #         out.write('v ')
        #         for x in c:
        #             out.write(str(x) + ' ')
        #         out.write('\n')
        # elif self._cps.ndim == 3:
        #     ncps = self._cps.shape[0]*self._cps.shape[1]
        #     for j in range(self._cps.shape[1]):
        #         for i in range(self._cps.shape[0]):
        #             out.write('v ')
        #             for x in self._cps[i,j]:
        #                 out.write(str(x) + ' ')
        #             out.write('\n')
        # else:
        #     msg = "CSType.write(): cannot interpret control point array!"
        #     raise RuntimeError(msg)

        stream.write(f"cstype {'rat' if self.rational else ''}{self.type}")
        stream.write('\n')

        match len(self.degree):
            case 1:
                stream.write(f'deg {self.degree[0]}\n')
                stream.write(f'curv {self.domain[0][0]} {self.domain[0][1]} ')
            case 2:
                stream.write(f'deg {self.degree[0]} {self.degree[1]}\n')
                stream.write(f'surf ')
                stream.write(f'{self.domain[0][0]} {self.domain[0][1]} ')
                stream.write(f'{self.domain[1][0]} {self.domain[1][1]} ')

        for i in range(-len(self._inds), 0):
            stream.write(f'{i} ')
        stream.write('\n')

        for i, parm in enumerate(self.parameters):
            stream.write(f"parm {'u' if i == 0 else 'v'}")
            for t in parm:
                stream.write(f' {t}')
            stream.write('\n')
        stream.write('end')

        return stream

    # def _reshape(self, array):
    #     deg = self._degree

    #     if deg[0] is not None and deg[1] is not None:
    #         # Control point indices in u-direction: 0, 1, ..., n[0]-1
    #         # and 0, 1, ..., n[1]-1 in v-direction, hence
    #         n = (len(self._parm[0]) - deg[0] - 1,
    #              len(self._parm[1]) - deg[1] - 1)

    #         if n[0] * n[1] != len(array):
    #             raise ValueError()

    #         # Order of control polygon vertices for deg (3, 2) surface:
    #         # u-direction is stored as columns, and v-direction as rows.
    #         # b00 b01 b02   0 4 8
    #         # b10 b11 b12 = 1 5 9
    #         # b20 b21 b22   2 6 10
    #         # b30 b31 b32   3 7 11
    #         # self._curv = np.reshape(self._curv, (n[0], n[1]), order='F')
    #         # self._cpid = np.reshape(self._cpid, (n[0], n[1]), order='F')
    #         return np.reshape(array, (n[0], n[1]), order='F')
    #     elif deg[0] is not None:
    #         # Check for consistency. No actual reshaping necessary.
    #         # if len(self._parm[0]) - deg[0] != len(self._cpid) + 1:
    #         #     raise ValueError()
    #         pass
    #     else:
    #         raise ValueError()

    # def _generate_cparray(self, points, vid):
    #     """ Generate control point array.

    #     Reshapes a sequence of control points as read from an OBJ file
    #     into an array. Inverse to :meth:`_flatten_cparray`.

    #     Note
    #     ----
    #     This function should only be called once to reshape the control
    #     point list read from an OBJ file into an array. Do not call directly
    #     from application code!
    #     """
    #     if points is None:
    #         raise ValueError()

    #     k = self._degu, self._degv

    #     if k[0] is not None and k[1] is not None:
    #         # The surface case, control point array C has 3 axis. The last
    #         # axis, i.e., c[i, j, :] holds point coordinates.
    #         if self.type == 'bspline':
    #             # Control point indices in u-diretion: 0, 1, ..., n[0]
    #             # and 0, 1, ..., n[1] in v-direction
    #             n = (len(self._parmu)-k[0]-2, len(self._parmv)-k[1]-2)
    #         elif self.type == 'bezier':
    #             # For a Bezier curve/surface control points are numbered
    #             # 0, 1, ..., k[0] in u-direction, etc.
    #             n = (k[0], k[1])

    #         # Order of control polygon vertices for deg (3, 2) surface:
    #         # b00 b01 b02   0 4 8
    #         # b10 b11 b12 = 1 5 9
    #         # b20 b21 b22   2 6 10
    #         # b30 b31 b32   3 7 11
    #         c = np.array([points[i] for i in self._vid])
    #         c = np.reshape(c, (n[1]+1, n[0]+1, c.shape[-1]), copy=False)

    #         self._cps = c.transpose(1, 0, 2)
    #     elif k[0] is not None:
    #         # The curve case, control point array c has 2 axis. The last
    #         # axis, i.e., c[i, :] holds point coordinates.
    #         if self.type == 'bspline':
    #             if len(self._parmu) - k[0] != len(self._vid) + 1:
    #                 raise ValueError()
    #         elif self.type == 'bezier':
    #             if k[0] != len(self._vid) - 1:
    #                 raise ValueError()

    #         self._cps = np.array([points[i] for i in self._vid])
    #     else:
    #         raise ValueError()

    #     del self._vid

    # def _flatten_cparray(self):
    #     """ Flatten the control point array of surface.

    #     Has no effect on the shape of the control point array when called
    #     for a curve. This is the inverse of :py:meth:`_generate_cparray`.

    #     Returns
    #     -------
    #     V : ~numpy.ndarray
    #         Linear sequence of control points. Corresponds to OBJ specification
    #         of flat arrays when used as surface control points.
    #     """
    #     if len(self._deg) == 2:
    #         if self._cps.ndim == 2:
    #             V = np.transpose(self._cps)
    #             V = np.ravel(V)
    #         else:
    #             V = np.transpose(self._cps, (1,0,2))
    #             V = np.ravel(V)
    #             V = np.reshape(V, (-1, self._cps.shape[-1]))
    #     else:
    #         V = np.array(self._cps)

    #     return V


class OBJError(Exception):
    pass


def read(filename, *tags, quiet=True):
    """ Read from .obj file.

    Assumes an .obj-like file structure, i.e., a text file where each
    line starts with a tag followed by numerical values. Lines whose tag
    is contained in `tags` are read. The values corresponding to a tag
    are returned as a list of lists: one list per tag that holds a list
    of numerical values per line read.

    Parameters
    ----------
    filename : str
        Name of object file.
    *tags
        Variable number of arguments of type :class:`str`.
    quiet : bool, optional
        Pass :obj:`False` to print comments and summary.

    Raises
    ------
    ValueError
        If invalid arguments are passed via `tags`.
    OBJError
        If file parsing fails.

    Returns
    -------
    data : list or tuple(list, ...)
        If multiple tags are specified the return value ``data[i]``
        corresponds to ``tags[i]``.


    To read vertex coordinates and vertex normals from file do

    >>> v, vn = read('input-file.obj', 'v', 'vn')

    Data blocks are returned as objects of type :class:`list`.
    """

    def parse_vertex(string):
        """ Parse vertex definition.

        Returned values can be negative (relative offsets). If positive,
        indices are 1-based.

        Parameters
        ----------
        string : str
            A v/vt/vn string representing a vertex definition as
            encountered when reading 'f' statements.

        Raises
        ------
        ValueError
            If conversion from `string` to integer fails.
        OBJError
            If `string` is malformed.

        Returns
        -------
        v : int or None
            Vertex index.
        vt : int or None
            Vertex texture index.
        vn : int or None
            Vertex normal index.
        """
        # Default return values. If v could not be assigned there is a
        # problem with the input.
        v, vt, vn = None, None, None

        if '//' in string:
            bits = string.split('//')

            # A v//vn statement is split by // into exactly two parts.
            # The definition is invalid in all other cases.
            if len(bits) == 2 and all(bits):
                v, vn = (int(bit) for bit in bits)
            else:
                raise OBJError(f"invalid {string!r}")
        elif '/' in string:
            bits = string.split('/')

            # A v/vt or v/vt/vn statement depending on how many parts
            # it gets split into by the / separator.
            if len(bits) == 2 and all(bits):
                v, vt = (int(bit) for bit in bits)
            elif len(bits) == 3 and all(bits):
                v, vt, vn = (int(bit) for bit in bits)
            else:
                raise OBJError(f"invalid {string!r}")
        else:
            # Base case, only v given. This will raise ValueError if
            # string cannot be converted to an integer value.
            v = int(string)

        return v, vt, vn

    def parse_freeform(tokens):
        """ Parse freeform statement.

        Read until error or a closing 'end' statement is encountered.
        As a side effect this method advances the file pointer.

        Parameters
        ----------
        tokens : list[str]
            Result of splitting the current line into tokens. Has to
            start with either 'curv', 'curv2', or 'surf'.

        Returns
        -------
        parm : tuple
            Knot vectors. For curves this is a one element tuple.
        inds : list[int]
            Control point indices.
        """
        nonlocal filepos

        # Default values for parmu and parmv. For curves parmu should
        # be set before encountering 'end'. For surfaces both should be
        # set to a squence of floats before reading 'end'.
        parmu, parmv = None, None

        if tokens[0] == 'curv':
            range = ((float(tokens[1]), float(tokens[2])), )

            # Indices refer to the list of vertices defined via lines
            # starting with v. No texture vertices or normals may be
            # specified for curves.
            try:
                inds = [parse_vertex(token)[0] for token in tokens[3:]]
            except OBJError as err:
                raise OBJError(
                    f"{filepos}: malformed {line.rstrip()!r} - {err}"
                ) from None
        elif tokens[0] == 'curv2':
            # Indices refer to vp, i.e., vertices in the parameter
            # domain, not the vertices defined via v. No texture or
            # normal may be specified.
            pass
        elif tokens[0] == 'surf':
            range = ((float(tokens[1]), float(tokens[2])),
                     (float(tokens[3]), float(tokens[4])))

            # The most general type of vertex definitions in the form
            # v/vt/vn can be used (as for face statements). Currently
            # everything but the v part of v/vt/vn is ignored.
            try:
                inds = [parse_vertex(token)[0] for token in tokens[5:]]
            except OBJError as err:
                raise OBJError(
                    f"{filepos}: malformed {line.rstrip()!r} - {err}"
                ) from None

        # Read freeform body statements until a terminating 'end' is found.
        # Note that file and line are defined in the enclosing scope!
        while line := file.readline():
            filepos += 1

            if tokens := line.split():
                if tokens[0] == 'parm':
                    if len(tokens) > 2:
                        # Knot vectors in u and v direction of a freeform
                        # curve or surface.
                        if tokens[1] == 'u':
                            parmu = [float(knot) for knot in tokens[2:]]
                        elif tokens[1] == 'v':
                            parmv = [float(knot) for knot in tokens[2:]]
                        else:
                            raise OBJError(
                                f"{filepos} : malformed {line.rstrip()!r} - "
                                + f"expecting 'u' or 'v', not {tokens[1]!r}")
                    else:
                        raise OBJError(
                            f"{filepos} : malformed {line.rstrip()!r} - "
                            + "not enough values to unpack")
                elif tokens[0] == 'trim':
                    pass
                elif tokens[0] == 'hole':
                    pass
                elif tokens[0] == 'end':
                    # End of freeform body statements. Always return a tuple
                    # of parameters values (aka knot vectors). For curves this
                    # has shape (parmu, ), for surfaces (parmu, parmv).
                    parm = tuple(x for x in (parmu, parmv) if x is not None)
                    return parm, inds

    def parse_scalar(string):
        """ Convert string to numeric value.

        Parameters
        ----------
        string : str
            A string representing a numeric value.

        Raises
        ------
        ValueError
            If `string` could not be converted.

        Returns
        -------
        scalar : int or float
            Numeric value.
        """
        try:
            value = int(string)
        except ValueError:
            value = float(string)

        return value

    # All input arguments have to be of type str (they are matched against
    # tokens read from an OBJ file and used as keys in a dictionary).
    # if any(not isinstance(tag, str) for tag in tags):
    #     raise ValueError("tags have to be of type 'str'")

    # Ouput data blocks for all arguments stored in a dictionary, keyed
    # on the corresponding tag. All data blocks are represented as lists.
    data = {tag: [] for tag in tags}

    # The number of encountered item definitions. Needed to resolve
    # negative (relative) item indices.
    vcnt, vtcnt, vncnt, vpcnt = 0, 0, 0, 0

    # Trimming loops and holes are defined via the curv2 statement. They
    # are referenced like vertices.
    curv2cnt = 0

    # Line number, current position in the file. Used to print informative
    # error messages and warnings.
    filepos = 0

    with open(filename, 'r') as file:
        while line := file.readline():
            filepos += 1

            # Extract runs of non-whitespace characters, the split()
            # method will also strip all whitespace.
            if tokens := line.split():
                # Increment of the number of mesh items encountered up to
                # this point. This is for counting!
                match tokens[0]:
                    case 'v':
                        vcnt += 1
                    case 'vt':
                        vtcnt += 1
                    case 'vn':
                        vncnt += 1
                    case 'vp':
                        vpcnt += 1
                    case 'curv2':
                        curv2cnt += 1

                # Groups, merging groups, smoothing groups and objects
                # defined in the file.
                if tokens[0] in {'g', 'mg', 's', 'o'}:
                    pass

                # Print any comments encountered during file parsing if
                # requested by the caller.
                if tokens[0].startswith('#') and not quiet:
                    print(line.rstrip())

                if tokens[0] == 'cstype':
                    if len(tokens) == 2:
                        type = tokens[1]
                        rational = False
                    elif len(tokens) == 3 and tokens[1] == 'rat':
                        type = tokens[2]
                        rational = True
                    else:
                        raise OBJError(
                            f"{filepos}: malformed {line.rstrip()!r}")

                if tokens[0] == 'deg':
                    if len(tokens) == 2:
                        deg = (int(tokens[1]), )
                    elif len(tokens) == 3:
                        deg = (int(tokens[1]), int(tokens[2]))
                    else:
                        raise OBJError(
                            f"{filepos}: malformed {line.rstrip()!r}")

                # Parse data if a recognized tag is given in args. Skip
                # everything else.
                if tokens[0] in tags:
                    if tokens[0] == 'p':
                        # Multiple point elements may be defined by listing
                        # their indices on a single line.
                        pass
                    elif tokens[0] == 'l':
                        # Line element definition. In addition to a vertex
                        # index, a texture vertex index may be specified.
                        pass
                    elif tokens[0] == 'f':
                        # Face element definition. In addition to a vertex
                        # index, texture vertex index and/or vertex normal
                        # index may be specified.
                        try:
                            f = [parse_vertex(token) for token in tokens[1:]]
                        except OBJError as err:
                            raise OBJError(
                                f"{filepos}: malformed {line.rstrip()!r} - {err}"
                            ) from None
                        except ValueError as err:
                            raise OBJError(
                                f"{filepos}: invalid {line.rstrip()!r} - {err}"
                            ) from None

                        for i, (v, vt, vn) in enumerate(f):
                            # Replace all relative and absolute indices with
                            # 0-based absolute indices.
                            v += vcnt if v < 0 else -1

                            if vt is not None:
                                vt += vtcnt if vt < 0 else -1

                            if vn is not None:
                                vn += vncnt if vn < 0 else -1

                            f[i] = (v, vt, vn)

                        data[tokens[0]].append(f)
                    elif tokens[0] in {'curv', 'surf'}:
                        # Freeform curve or surface definition follows.
                        # This requires parsing multiple lines until and
                        # end statement is found.
                        parm, inds = parse_freeform(tokens)
                        cs = CSType(parm, inds, deg, type, rational)

                        # Convert 1-based and relative vertex indices to
                        # 0-based absolute vertex indices.
                        cs._inds = [i + vcnt if i < 0 else i - 1
                                    for i in cs._inds]

                        data[tokens[0]].append(cs)
                    else:
                        # Generic data, i.e., a line starting with a tag
                        # and an arbitrary number of numeric values that
                        # are converted to ints or floats. More complicated
                        # data would require a custom parsing method.
                        try:
                            val = [parse_scalar(token)
                                   for token in tokens[1:]]
                        except ValueError as err:
                            raise OBJError(
                                f"{filepos}: invalid {line.rstrip()!r} - {err}"
                            ) from None

                        data[tokens[0]].append(val)

    if not quiet:
        print(f'parsing {'\33[1m'}{filename}{'\33[0m'} results in')

        print(f'\t\u251c\u2500 {vcnt} vertex definitions')
        print(f'\t\u251c\u2500 {vncnt} vertex normal definitions')
        print(f'\t\u2514\u2500 {vtcnt} vertex texture definitions')

    # Return None instead of () and data instead of (data, ) tuples if no
    # or only one tag is specified.
    match len(tags):
        case 0:
            return
        case 1:
            return data[tags[0]]

    # This works as intended because dictionary values are iterated over
    # in insertion order (guaranteed since version 3.7 of Python).
    return tuple(data.values())


def write(filename, **data):
    """ Write as .obj file.

    Face definitions: each entry of a face (a vertex definition) can be a
    single integer or a 3-tuple of integers. Tuple entries are interpreted
    as v/vt/vn triples, missing entries have to be specified with a
    :obj:`None` value.

    Parameters
    ----------
    filename : str
        Name of output file.
    **data
        Keyword arguments.


    Data to be stored in the file is passed via keyword arguments:

    >>> mesh.write('output-file.obj', tag=value, ...)

    This assumes that ``value`` can be interpreted as a 2-dimensional
    array. The contents of each row are written to a line that starts
    with the given tag (keyword).
    """

    def format(vertex):
        """ Pretty print vertex.

        Parameters
        ----------
        vertex : int or tuple
            Integer or 3-tuple of integers.

        Returns
        -------
        str
            String representation according to .obj standard.
        """
        try:
            v = vertex[0] + 1
        except TypeError:
            return f' {int(vertex) + 1}'
        else:
            vt = '' if vertex[1] is None else vertex[1] + 1
            vn = '' if vertex[2] is None else vertex[2] + 1

            if vertex[2] is not None:
                return f' {v}/{vt}/{vn}'
            else:
                if vertex[1] is not None:
                    return f' {v}/{vt}'
                else:
                    return f' {v}'

    with open(filename, 'w') as file:
        for key, value in data.items():
            if key == 'f':
                for face in value:
                    file.write('f')

                    for vertex in face:
                        file.write(format(vertex))

                    file.write('\n')
            else:
                # If a data block is iterable, each obtained value
                # defines the content(s) of a single line.
                try:
                    lines = iter(value)
                except TypeError:
                    file.write(f'{key} {value}\n')
                else:
                    # If a line is iterable it produces as sequence
                    # of values written to a single line.
                    for line in lines:
                        try:
                            items = iter(line)
                        except TypeError:
                            file.write(f'{key} {line}\n')
                        else:
                            file.write(key)

                            for item in items:
                                file.write(f' {item}')

                            file.write('\n')
