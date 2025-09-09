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

""" Halfedge data structure.

An orientable 2-manifold mesh (with or without boundary) is described by
three containers:

    - a list of :class:`Vertex` objects,
    - a list of :class:`Face` objects,
    - and a dictionary of :class:`Halfedge` objects.

These containers and the relations between their items are managed by
the :class:`Mesh` class.

Note
----
To ease debugging, this module relies on assertions which can slow down
script execution. You can disable assertions by running in optimized mode
via the "-O" command line argument.
"""

from pathlib import Path
from datetime import datetime
from time import time   #use perf_counter instead
from copy import copy, deepcopy

import numpy as np

import m3sh.obj as obj
import m3sh.flags as flags


class Mesh:
    """ Mesh kernel.

    The combinatorics of a mesh can be built by reading from a file or
    by converting a sequence of vertex coordinates and a sequence of face
    definitions to its halfedge representation.

    Parameters
    ----------
    points : array_like, optional
        Vertex coordinates. Converted to an equivalent
        :obj:`~numpy.ndarray` object if necessary.
    faces : array_like, optional
        Face definitions, 0-based vertex indexing.
    name : str, optional
        Name tag.

    Raises
    ------
    NonManifoldError
        When trying to initialize a mesh from non-manifold data.

    Note
    ----
    The `NumPy glossary <https://numpy.org/doc/stable/glossary.html>`_
    states the following about :term:`array_like` data representations:

        *Any scalar or sequence that can be interpreted as*
        :class:`numpy.ndarray`. *In addition to ndarrays and scalars
        this category includes lists (possibly nested and with different
        element types) and tuples. Any argument accepted by*
        :func:`numpy.array` *is array_like.*
    """

    def __init__(self, points=None, faces=None, *, name=None):
        """ Initialize from vertex and face lists.
        """
        CWHITERED = '\33[41m'               # white on red background
        CEND = '\33[0m'

        # The points argument could also be an integer resulting in an
        # abstract connectivity-only halfedge mesh...
        if points is None and faces is not None:
            msg = "face definitions require 'points' argument != None"
            raise ValueError(msg)

        if points is not None:
            self._points = np.asarray(points)
            self._verts = [Vertex(i, parent=self)
                           for i in range(len(points))]

            if self._points.base is not None:
                print(f"{CWHITERED}array data buffer not owned by " +
                      f"'points' array{CEND}")
        else:
            self._points = None
            self._verts = []

        # Used to detect and handle non-manifold vertices during mesh
        # construction.
        self._vhout = {v: set() for v in self._verts}

        # User defined vertex, halfedge, and face data. Each container
        # holds tuples (name, attr, default) that describe the mesh
        # attribute name, the mesh item property name and the default
        # value when adding a new mesh item of the respective type.
        self._vattr = []
        self._hattr = []
        self._fattr = []

        # A dictionary that maps pairs of vertices to halfedges. Useful
        # and efficient for checking if vertices are adjacent.
        self._halfs = dict()

        # Add all faces to the mesh. A vertex that does not get a valid
        # outgoing edge assigned during this process is disconnected from
        # all faces and not adjacent to any other vertex.
        self._faces = []

        if faces is not None:
            for face in faces:
                self.add_face(face)

        # Typically one does not expect isolated vertices in a mesh.
        if any(v.isolated for v in self._verts):
            print(f'{CWHITERED}there are isolated vertices{CEND}')

        # Vertex neighborhood iterators will not work properly in the
        # presence of non-manifold vertices.
        for v in self._verts:
            if not v._manifold:
                raise NonManifoldError(f'vertex #{v._idx} is non-manifold')

        # The corresponding property setter will strip any directory
        # prefix and type suffix from the name.
        self.name = name

    # def __repr__(self):
    #     return (f'Mesh({repr(self._points)}, \n' +
    #             f'{[[int(v) for v in f] for f in self]})')

    # def __str__(self):
    #     return (str(self._points)
    #             + '\n[' + '\n '.join([str(f) for f in self]) + ']')

    def __iter__(self):
        """ Face iterator.

        The returned iterator visits all faces of a mesh that are **not**
        marked as deleted in order of ascending face indices.

        Yields
        ------
        Face
            Next face in insertion order traversal.


        The loop that visits all faces of a mesh that contribute to its
        combinatorics

        .. code-block:: python
           :linenos:

           for f in mesh:
               # do something with the face
               ...

        is equivalent to explicitly checking the :attr:`deleted` attribute
        of a face:

        .. code-block:: python
           :linenos:

           for f in mesh.faces:
               if not f.deleted:
                   # do something with the face
                   ...
        """
        return (f for f in self._faces if not f._deleted)

    def __copy__(self):
        """ Shallow mesh copy.

        Duplicate the mesh combinatorics and vertex coordinates. Equivalent
        to :meth:`copy` method.

        Returns
        -------
        Mesh
            Copy of the mesh.
        """
        return self.copy()

    def __deepcopy__(self, *args):
        """ Reserved for future use.

        Use :meth:`copy` to copy a halfedge mesh.
        """
        raise NotImplementedError('use .copy() instead')

    def __bool__(self):
        return True

    def __getitem__(self, index):
        # Treat a mesh like a tuple consisting of a list of goemetric
        # vertices and face definitions.
        if index == 0:
            return self._points
        elif index == 1:
            return self._faces

        raise IndexError(f'index must be 0 or 1, got {index}')

    @property
    def points(self):
        """ Vertex coordinate array.

        Direct read and write access to vertex coordinates. Changing the
        size of the coordinate array is likely to break the halfedge data
        structure.

        :type: ~numpy.ndarray

        Note
        ----
        The vertex coordinate array contains coordinate entries of deleted
        vertices. Calling :meth:`clean` removes those entries.
        """
        return self._points

    @points.setter
    def points(self, value):
        self._points = np.asarray(value)

    @property
    def vertices(self):
        """ Vertex list.

        Read access to the vertex list. This list should not be modified
        directly.

        :type: list[Vertex]

        Note
        ----
        The vertex list may contain deleted vertices. Call
        :meth:`~Mesh.clean` to remove deleted vertices from the vertex
        container.
        """
        return self._verts

    @property
    def faces(self):
        """ Face list.

        Read access to the face list. This list should not be modified
        directly.

        :type: list[Face]

        This is **not** the list passed as argument `faces` during mesh
        construction but it can be generated easly with a list
        comprehension:

        >>> faces = [[int(v) for v in f] for f in mesh]

        Note
        ----
        The face list may contain deleted faces. A call to
        :meth:`~Mesh.clean` will remove such entries.
        """
        return self._faces

    @property
    def halfedges(self):
        """ Halfedge dictionary.

        Dictionary that maps pairs of :class:`Vertex` objects to
        :class:`Halfedge` instances. Hence, to visit :class:`Halfedge`
        instances one can use

        .. code-block:: python

            for h in mesh.halfedges.values():
                v = h.origin
                w = h.target
                ...

        to directly visit halfedges as :class:`Vertex` instance pairs use

        .. code-block:: python

            for v, w in mesh.halfedges.keys():
                ...

        To check whether two vertices of a mesh are adjacent (connected
        by an edge) we can do one of the following. The resulting value
        depends on the chosen query method:

        .. code-block:: python

           (v, w) in mesh.halfedges                 # True or False
           h = mesh.halfedges[v, w]                 # may raise KeyError
           h = mesh.halfedges.get((v, w))           # can result in h = None
        """
        return self._halfs

    @property
    def size(self):
        """ Mesh size.

        Mesh size **not** accounting for deleted vertices and faces. The
        attribute value :math:`(v, e, f)` holds the number of vertices,
        the number of edges, and the number of faces.

        :type: (int, int, int)

        Note
        ----
        In the presence of **deleted** items the value ``len(mesh.vertices)``
        (resp. ``len(mesh.faces)``) and the corresponding value of
        ``mesh.size`` are different.
        """
        assert len(self._halfs) % 2 == 0

        return (sum(1 for _ in self._viter()),
                len(self._halfs) // 2,
                sum(1 for _ in self._fiter()))

    @property
    def name(self):
        """ Name property.

        Name augmented with current time stamp.

        :type: str

        Note
        ----
        The returned string does not include a type suffix!
        """
        date = str(datetime.now())[:10]
        time = str(datetime.now())[11:16]

        return f'{date}_{time}_{self._name}'

    @name.setter
    def name(self, value):
        self._name = value if value is None else Path(value).stem

    @classmethod
    def read(cls, filename, *args, merge=False, quiet=True):
        """ Read mesh from file.

        Read mesh combinatorics (face definitions) and vertex coordinates
        from an OBJ file. Additional data is read on request.

        Parameters
        ----------
        filename : str
            Name of an OBJ file.
        *args
            Variable number of arguments of type :class:`str`.
        quiet : bool, optional
            Suppress console output.

        Returns
        -------
        mesh : Mesh
            Mesh object.
        data : ndarray or tuple(ndarray, ...)
            Data blocks as requested via `args`. If a data block could
            not be read, a :obj:`None` value is returned.


        Vertex normals or texture coordinates stored in a file can be
        read via

        >>> mesh, vecs, uvs = Mesh.read(filename, 'vn', 'vt')

        Note
        ----
        Additional return values (vertex normals, texture vertices, and
        custom data) are returned in the same order as they are presented
        in the argument list `args`.
        """
        CBOLD = '\33[1m'                    # bold text, white on black
        CEND = '\33[0m'

        if not quiet:
            start = time()
            print(f'reading {CBOLD}{Path(filename).name}{CEND}', end=' ...')

        if 'v' in args:
            raise ValueError("'v' cannot be used as argument")

        if 'f' in args:
            raise ValueError("'f' cannot be used as argument")

        # The *data expression will assign a list of all return values not
        # assigned to a name to data.
        verts, faces, *data = obj.read(filename, 'v', 'f', *args)

        # Convert list of data blocks to a dictionary. Since insertion
        # order traversal is guaranteed, *data before conversion is equal
        # to *data.values() after conversion.
        data = {arg: block for arg, block in zip(args, data)}

        if not quiet:
            print(f' done ({time()-start:.3f} sec, {merge=})')

            for arg in args:
                print(f"\t\u251c\u2500 data block '{arg}' " +
                      f"of size {np.shape(data[arg])}")

            print(f'\t\u251c\u2500 {len(verts)} vertices')
            print(f'\t\u2514\u2500 {len(faces)} faces')

        if merge:
            start = time()
            faces = obj.merge(verts, faces)

            if not quiet:
                print(f'merged vertices of {CBOLD}{Path(filename).name}' +
                      f'{CEND} by distance ({time()-start:.3f} sec)')

        if 'vn' in args:
            # Check if each vertex is assigned the normal with identical
            # index value.
            for face in faces:
                for v, _, vn in face:
                    if vn is not None and v != vn:
                        break
                else:
                    continue

                # Break outer loop. Only reached if break was executed
                # in the inner loop.
                break
            else:
                # All vertices are assigned the normal vector with
                # identical index. This is fine for mesh generation.
                mesh = cls(verts, [[v[0] for v in f] for f in faces])
                mesh.name = filename

                return mesh, *data.values()

            if not quiet:
                print(f'rebuilding normal data block of ', end='')
                print(f'{CBOLD}{Path(filename).name}{CEND}')
                print(f'\t\u251c\u2500 {len(data["vn"])} normals read')

            # Fix the normals data block such that there are as many
            # normals as vertices and both blocks correponds by index.
            # If normal indices are not specified explicitly in face
            # definitions, none of this is executed!
            idx = [set() for _ in verts]

            for face in faces:
                for v, _, vn in face:
                    idx[v].add(vn)

            # The set idx[v] now holds all indices of normals assigned
            # to vertex v. If there there are multiple normals assigned
            # (allowed by OBJ specifications) they will be averaged.
            block = data['vn']
            normals = np.empty(verts.shape)
            num_avg = 0

            for v, _ in enumerate(verts):
                cluster = idx[v]

                if len(cluster) > 0:
                    normals[v, :] = sum(block[j, :] for j in cluster)

                    if len(cluster) > 1:
                        normals[v, :] /= np.linalg.norm(normals[v, :])
                        num_avg += 1
                else:
                    normals[v, :] = np.nan

            data['vn'] = normals

            if not quiet:
                print(f'\t\u2514\u2500 {num_avg} averages performed')

        if args:
            mesh = cls(verts, [[v[0] for v in f] for f in faces])
            mesh.name = filename

            return mesh, *data.values()

        return cls(verts, [[v[0] for v in f] for f in faces], name=filename)

    def write(self, filename, quiet=True, **data):
        """ Write mesh to file.

        Data arrays, like vertex normals and texture coordinates, can be
        saved by passing them as keyword arguments.

        Parameters
        ----------
        filename : str
            Name of an OBJ file.
        quiet : bool, optional
            Suppress console output.
        **data
            Arbitrary keyword arguments.


        User defined data blocks can be written with

        >>> mesh.write('output-file.obj', line_tag=data)

        This assumes that ``data`` can be interpreted as a 2-dimensional
        array. The contents of each row are written to a line that starts
        with the given tag. If vertex normals are available, they can
        be stored via

        >>> mesh.write('outfile-file.obj', vn=normals)

        Note
        ----
        The standard OBJ tags 'v' and 'f' may not be used as keywords
        since they are implicitly used when writing mesh data to an OBJ
        file.
        """
        CBOLD = '\33[1m'                    # bold text, white on black
        CEND = '\33[0m'

        if 'v' in data.keys():
            raise ValueError("'v' may not be used as data tag")

        if 'f' in data.keys():
            raise ValueError("'f' may not be used as data tag")

        vt_given = False
        vn_given = False

        if 'vt' in data.keys():
            vt = data['vt']
            vt_given = True

            if len(vt) != len(self._points):
                msg = (f'number of texture vertices ({len(vt)}) != ' +
                       f'number of vertices ({len(self._points)})')
                raise ValueError(msg)

        if 'vn' in data.keys():
            vn = data['vn']
            vn_given = True

            if len(vn) != len(self._points):
                msg = (f'number of vertex normals ({len(vn)}) != ' +
                       f'number of vertices ({len(self._points)})')
                raise ValueError(msg)

        tidx = lambda v : int(v) if vt_given else None
        nidx = lambda v : int(v) if vn_given else None

        faces = (((int(v), tidx(v), nidx(v)) for v in f) for f in self)

        if not quiet:
            start = time()
            print(f'writing {CBOLD}{Path(filename).name}{CEND}', end=' ...')

        obj.write(filename, v=self._points, f=faces, **data)

        if not quiet:
            print(f' done ({time()-start:.3} sec)')

    def add_vertex(self, point, *args, **kwargs):
        """ Create and add new vertex.

        The first point added to a mesh determines the dimensionality
        of all mesh vertices.

        Parameters
        ----------
        point : array_like or float
            Vertex coordinates.
        *args
            Variable number of scalars.
        **kwargs
            Attribute name and value pairs.

        Raises
        ------
        ValueError
            If `point` has the wrong shape.

        Returns
        -------
        Vertex
            The newly created :class:`Vertex` instance.

        Note
        ----
        When adding a new vertex all vertex data blocks are extended by
        a correponding value:

            - either by using the `default` value specified when
              the data block was added,

            - or by using a value provided as keyword argument.

        Values specified as keyword arguments that don't fit this pattern
        are added as :class:`Vertex` instance attributes and only availabe
        as attributes of this particular instance.
        """
        # Append to (or create) the array of all vertex coordinates.
        point = [point, *args] if len(args) else point
        self._points = obj._array_append(self._points, point)

        # New vertex object that goes to the end of the list of all
        # vertices.
        v = Vertex(len(self._verts), parent=self)

        self._verts.append(v)
        self._vhout[v] = set()
        self._add_attr_values(v, *self._vattr, **kwargs)

        # Attributes that are not managed via a corresponding mesh data
        # block become ordinary vertex instance attribute.
        for key, value in kwargs.items():
            # This calls the setter method of the property with the
            # name 'key' if it exists.
            # if key not in (attr for _, attr, _ in self._vattr):
            setattr(v, key, value)

        return v

    def add_vertex_data(self, name, attr, data, default=None):
        """ Add vertex data.

        The `data` object has to allow access to vertex data using
        index notation. The data block as a whole can be accessed via
        ``self.name`` and the value ``data[v]`` as ``v.attr``.

        Parameters
        ----------
        name : str
            Name of the data block.
        attr : str
            Name of vertex attribute.
        data : list or dict or ~numpy.ndarray
            Data object.
        default : object, optional
            Immutable default vertex attribute value.

        Raises
        ------
        ValueError
            If a data block of the same name already exists.


        To add a vector field (one vector per vertex) to a mesh we can
        do the following:

        .. code-block:: python
           :linenos:

            # Allocate data block of appropriate size and type.
            vecs = np.array_like(mesh.points)

            # Compute values for each vertex of the mesh.
            for v in mesh.vertices:
                vecs[v] = ...

            # Add the data block to the mesh. The vecs array can now be
            # accessed as mesh.vecs (we could have used any other name).
            mesh.add_vertex_data('vecs', 'vec', vecs)
            print(mesh.vecs is vecs)

            # Rows of the data block can now be accessed locally just
            # like the rows of the vertex coordinate array.
            for v in mesh.vertices:
                print(v.vec == vecs[v])


        Note
        ----
        A mutable `default` value has the same drawbacks as mutable
        default function arguments.
        """

        def get(self):
            # self refers to a vertex instance
            return getattr(self._mesh, private_name)[self]

        def get_data(self):
            # self refers to a mesh instance
            return getattr(self, private_name)

        def set(self, value):
            getattr(self._mesh, private_name)[self] = value

        def set_data(self, value):
            setattr(self, private_name, value)

        def del_data(self):
            for i, (data_name, _, _) in enumerate(self._vattr):
                if data_name == private_name:
                    del self._vattr[i]

            delattr(self, private_name)

        # The hidden name for direct access of the attribute data block.
        # Make sure not to unintentionally overwrite existing data.
        private_name = '_' + name

        if hasattr(self, private_name):
            raise ValueError(f"data block '{name}' already exists")

        # if hasattr(Vertex, attr):
        #     raise ValueError(f"vertex attribute '{attr}' already in use")

        setattr(self, private_name, data)

        # Store name and type of attribute. Memory management functions
        # need this information.
        self._vattr.append((private_name, attr, default))

        # Attribute access via properties. One global property bound to
        # the mesh and local properties bound to vertices.
        setattr(self.__class__, name, property(get_data, set_data, del_data))
        setattr(Vertex, attr, property(get, set))

    def add_face(self, face, *args, **kwargs):
        """ Create and add new face.

        Vertex identifiers used in the definition of a face have to
        refer to existing vertices of the mesh.

        Parameters
        ----------
        face : list[int] or list[Vertex]
            Combinatorial face definition.
        *args
            Variable number of :class:`Vertex` or :class:`int` arguments.
        **kwargs
            Attribute name and value pairs.

        Raises
        ------
        NonManifoldError
            If topological problems occur.
        IndexError
            If the given vertex indices are out of bounds.
        ValueError
            If the given arguments do not define a valid face.

        Returns
        -------
        Face
            The newly created :class:`Face` instance.

        Note
        ----
        See :meth:`~Mesh.add_vertex` for a detailed discussion of `kwargs`.
        """
        # Number of vertices of the face, same as the number of edges
        # that bound the face.
        face = [face, *args] if len(args) else face
        n = len(face)

        # Check for degeneracies: All vertices have to be topologically
        # different. If this test is passed there still need to be at
        # least three vertices. Duplicate coordinates are not a problem.
        if len(set(face)) != n:
            raise ValueError('face contains duplicate vertices')

        if n < 3:
            raise ValueError('face has less than three vertices')

        # Check consistency pre-conditions on vertex attributes. Failing
        # indicates an invalid halfedge data structure.
        for k in range(n):
            v = self._verts[face[k]]

            assert not v._deleted or not self._vhout[v]
            assert v._halfedge is not None or not self._vhout[v]

        f = Face(len(self._faces))          # new face object
        edge_loop = []                      # halfedge loop around face

        # Dry run.
        for k in range(n):
            v = self._verts[face[k]]
            w = self._verts[face[(k + 1) % n]]
            h = self._halfs.get((v, w), None)

            if h is not None and h._face is not None:
                msg = f'edge ({v._idx}, {w._idx}) is non-manifold'
                raise NonManifoldError(msg)

        # Add/process edges of the given face. The _add_halfedge method
        # may raise a NonManifoldError. If vertex indices are out of
        # bounds an IndexError is raised in the loop.
        for k in range(n):
            v = self._verts[face[k]]
            w = self._verts[face[(k + 1) % n]]
            h = self._add_halfedge(v, w)

            edge_loop.append(h)

        # Proceed with linking mesh items together. The _pair attribute
        # of h was set by _add_halfedge if the pair was already mapped.
        for h in edge_loop:
            h._face = f
            h._origin._halfedge = h

            # A halfedge should always have a pair. Create a boundary
            # edge if the pair does not exist yet.
            if h._pair is None:
                self._add_halfedge(h._target, h._origin)

        self._faces.append(f)               # add face
        f._halfedge = edge_loop[0]          # set incident halfedge

        # Take care of next and prev halfedge pointer around the inner
        # edge loop of the face.
        for i in range(n):
            j = (i + 1) % n

            edge_loop[i]._next = edge_loop[j]
            edge_loop[j]._prev = edge_loop[i]

        # The interior edge loop of a face is already complete. We need to
        # take care of the outer loop. To do this we rely on the correctly
        # set pair pointers and prev/next pointers of the inner loop.
        for h in edge_loop:
            ph = None                       # previous halfedge, if any
            hh = h                          # initialize iteration

            # Rotate the edge h = (v, w) as long as possible clockwise
            # around its origin v until we reach the boundary. There is
            # nothing to do if we circulate back to h itself.
            while True:
                if hh._pair._face is None:
                    ph = hh._pair
                    break

                hh = hh._pair._next

                if hh is h:
                    break

            hh = h                          # reset iteration variable

            # If ph was set, halfedge h is now rotated counterclockwise
            # about its origin v until we reach the boundary.
            while ph is not None:
                hh = hh._prev._pair

                if hh is h:
                    # This branch should never be taken. There's no
                    # recovery from this problem.
                    msg = f'vertex #{h._origin._idx} is non-manifold'
                    raise NonManifoldError(msg)

                if hh._face is None:
                    # We reached the boundary again. Link the halfedges.
                    hh._prev = ph
                    ph._next = hh

                    break

        self._add_attr_values(f, *self._fattr, **kwargs)

        # Attributes that are not managed via a corresponding mesh data
        # block become ordinary face instance attributes.
        for key, value in kwargs.items():
            # This calls the setter method of the property with the
            # name 'key' if it exists, i.e., for managed attributes we
            # set the same value as done in the previous loop (could be
            # avoided with an if-condition).
            # if key not in (attr for _, attr, _ in self._fattr):
            setattr(f, key, value)

        return f

    def add_face_data(self, name, attr, data, default=None):
        """ Add face data.

        The `data` object has to allow access to face data using
        index notation. The data block as a whole can be accessed via
        ``self.name`` and the value ``data[f]`` as ``f.attr``.

        Parameters
        ----------
        name : str
            Name of the data block.
        attr : str
            Name of face attribute.
        data : list or dict or ~numpy.ndarray
            Data object.
        default : object, optional
            Immutable default face attribute value.

        Raises
        ------
        ValueError
            If a data block of the same name already exists.

        Note
        ----
        See :meth:`~Mesh.add_vertex_data` for an example.
        """

        def get(self):
            # self refers to a face instance
            return getattr(self._halfedge._origin._mesh, private_name)[self]

        def get_data(self):
            # self refers to a mesh instance
            return getattr(self, private_name)

        def set(self, value):
            getattr(self._halfedge._origin._mesh, private_name)[self] = value

        def set_data(self, value):
            setattr(self, private_name, value)

        def del_data(self):
            for i, (data_name, _, _) in enumerate(self._fattr):
                if data_name == private_name:
                    del self._fattr[i]

            delattr(self, private_name)

        # The hidden name for direct access of the attribute data block.
        # Make sure to not unintentionally overwrite existing data.
        private_name = '_' + name

        if hasattr(self, private_name):
            raise ValueError(f"data block '{name}' already exists")

        # if hasattr(Face, attr):
        #     raise ValueError(f"face attribute '{attr}' already in use")

        setattr(self, private_name, data)

        # Store name and type of attribute. Memory management functions
        # need this information.
        self._fattr.append((private_name, attr, default))

        # Attribute access via properties. One global property bound to
        # the mesh and local properties bound to faces.
        setattr(self.__class__, name, property(get_data, set_data, del_data))
        setattr(Face, attr, property(get, set))

    def add_halfedge_data(self, name, attr, data, default=None):
        """ Add halfedge data.

        The `data` object has to allow access to halfedge data using
        index notation. The data block as a whole can be accessed via
        ``self.name`` and the value ``data[h]`` as ``h.attr``.

        Parameters
        ----------
        name : str
            Name of the data block.
        attr : str
            Name of halfedge attribute.
        data : dict
            Data object.
        default : object, optional
            Immutable default halfedge attribute value.

        Raises
        ------
        ValueError
            If a data block of same name already exists or the
            data block is not a dictionary instance.

        Note
        ----
        See :meth:`~Mesh.add_vertex_data` for an example.
        """
        if not isinstance(data, dict):
            raise ValueError("data block has to be of type 'dict'")

        def get(self):
            # self refers to a halfedge instance
            return getattr(self._origin._mesh, private_name)[self]

        def get_data(self):
            # self refers to a mesh instance
            return getattr(self, private_name)

        def set(self, value):
            getattr(self._origin._mesh, private_name)[self] = value

        def set_data(self, value):
            setattr(self, private_name, value)

        def del_data(self):
            for i, (data_name, _, _) in enumerate(self._hattr):
                if data_name == private_name:
                    del self._hattr[i]

            delattr(self, private_name)

        # The hidden name for direct access of the attribute data block.
        # Make sure not to unintentionally overwrite existing data.
        private_name = '_' + name

        if hasattr(self, private_name):
            raise ValueError(f"data block '{name}' already exists")

        # if hasattr(Halfedge, attr):
        #     raise ValueError(f"halfedge attribute '{attr}' already in use")

        setattr(self, private_name, data)

        # Store name and type of attribute. Memory management functions
        # need this information.
        self._hattr.append((private_name, attr, default))

        # Attribute access via properties. One global property bound to
        # the mesh and local properties bound to halfedges.
        setattr(self.__class__, name, property(get_data, set_data, del_data))
        setattr(Halfedge, attr, property(get, set))

    def clear(self):
        """ Clear all mesh items.

        The memory occupied by the coordinate array is garbage collected
        once no further references or views of it remain.

        Note
        ----
        Data blocks are cleared by calling the data block's own
        :func:`~object.clear` method.
        """
        self._points = None

        # Lists and dictionary attributes are cleared in place instead of
        # resetting them to a new empty container.
        self._verts.clear()
        self._halfs.clear()
        self._faces.clear()
        self._vhout.clear()

        for attr in (self._vattr, self._hattr, self._fattr):
            for name, _, _ in attr:
                data = getattr(self, name)

                try:
                    data.clear()
                except AttributeError:
                    obj._array_clear(data)

    def clean(self):
        """ Garbage collection.

        Removes all deleted mesh items from the respective containers.
        Previously obtained vertex and face indices may become invalid.

        Note
        ----
        Use sparingly.
        """
        assert len(self._points) == len(self._verts)

        def shrink(data, items, idx):
            if isinstance(data, list):
                data[:] = (data[item] for item in items if not item._deleted)
            elif isinstance(data, dict):
                for key in data.keys():
                    if key._deleted:
                        del data[key]
            elif isinstance(data, np.ndarray):
                # The first assignment has no effect on data for an empty
                # list of indices, resize will then set the length of the
                # first axis of data to zero.
                data[:len(idx), ...] = data[idx, ...]
                data.resize(shape, refcheck=False)

        # Invalidate all attributes of vertices to be removed from the mesh.
        # This should prevent accidental access by triggering assertions and
        # raising exceptions by references outside the mesh instance.
        for v in self._verts:
            if v._deleted:
                del self._vhout[v]
                v._invalidate()

        # Find the indices of vertices *not* marked for deletion. Calculate
        # the shape of the compressed coordinate array.
        vidx = [i for i, v in enumerate(self._verts) if not v._deleted]
        fidx = [i for i, f in enumerate(self._faces) if not f._deleted]

        # Move the coordinates of live vertices to the front of the point
        # array, preserving their relative order. Then compress the array
        # by in-place resizing.
        shape = list(self._points.shape)
        shape[0] = len(vidx)

        if len(vidx):
            self._points[:len(vidx), ...] = self._points[vidx, ...]
            self._points.resize(shape, refcheck=False)
        else:
            self._points = None

        # Attribute data blocks have to be rearranged before changing the
        # corresponding mesh item container!
        for attr, items, idx in zip((self._vattr, self._hattr, self._fattr),
                                    (self._verts, self._halfs, self._faces),
                                    (vidx, None, fidx)):
            for name, _, _ in attr:
                # The nested shrink method relies on the shape variable
                # set earlier.
                shrink(getattr(self, name), items, idx)

        # Update vertex container to skip all unused vertices. Update all
        # vertex indices afterwards.
        self._verts[:] = (v for v in self._verts if not v._deleted)

        for i, v in enumerate(self._verts):
            v._idx = i

        # Remove deleted halfedges from the halfedge container. Invalidate
        # all their attributes.
        for key, h in self._halfs.items():
            if h._deleted:
                # This should never be reached. By design there should
                # not be any halfedges marked as deleted in the dictionary.
                raise RuntimeError('halfedge structure seems corrupt')

                h = self._halfs.pop(key)
                h._invalidate()

        # Invalidate all attributes of faces to be removed from the mesh.
        for f in self._faces:
            if f._deleted:
                f._invalidate()

        # Update face container to skip all unwanted faces. Afterwards
        # update all face indices.
        self._faces[:] = (f for f in self._faces if not f._deleted)

        for i, f in enumerate(self._faces):
            f._idx = i

    def clone(self, mesh):
        """ In-place mesh copy.

        Implements assignment operator like behavior. Performs the same
        operation as :meth:`copy` but assigns the result to the mesh
        instance `self`.

        Parameters
        ----------
        mesh : Mesh
            Source mesh.
        """

        def dict_copy(data, map):
            # The map argument provides a mapping of keys. Mapped values
            # are just assigned and not copied.
            data_copy = dict()

            for key, value in data.items():
                # This could be used to make copies of ndarrays stored
                # as dictionary values -- but makes copy() inconsistent.
                data_copy[map[key]] = value

            return data_copy

        vmap, hmap, fmap = self._clone_connectivity_from(mesh)

        # Since _points is of type ndarray, this is a deep copy, i.e. no
        # parts of the vertex coordinate data buffers are shared.
        self._points = mesh._points.copy()

        # Clear data blocks in the target mesh. Some of them may have already
        # been removed from the source mesh.
        for attr_type in (self._vattr, self._hattr, self._fattr):
            for private_name, _, _ in attr_type:
                delattr(self, private_name)

        # Shallow copies of lists that hold tuples (immutable, no point
        # making a deep copy).
        self._vattr = mesh._vattr.copy()
        self._hattr = mesh._hattr.copy()
        self._fattr = mesh._fattr.copy()

        # Copy all data blocks. Shallow copies, except for ndarray, custom
        # method to handle dictionary copies.
        for attr, map in zip((mesh._vattr, mesh._hattr, mesh._fattr),
                             (vmap, hmap, fmap)):
            for private_name, _, _ in attr:
                data = getattr(mesh, private_name)

                if isinstance(data, dict):
                    setattr(self, private_name, dict_copy(data, map))
                else:
                    setattr(self, private_name, data.copy())

        return self

    def copy(self):
        """ Return mesh copy.

        Duplicate combinatorics, vertex coordinates, and data blocks of a
        mesh. Data blocks are copied using the data object's
        :meth:`~object.copy` method. For data blocks of type :class:`list`
        and :class:`dict` this results in shallow copies. Data of type
        :class:`~numpy.ndarray` won't share data buffers with data blocks
        of the copy.

        Returns
        -------
        Mesh
            Copy of the mesh.

        Note
        ----
        User defined vertex, face, and halfedge instance attributes are
        copied using :func:`copy.copy`.
        """
        return self.__class__().clone(self)

    def delete_vertex(self, vertex, fill=None):
        """ Delete vertex.

        Delete vertex and optionally fill the formed hole.

        Parameters
        ----------
        vertex : Vertex or int
            Vertex identifier.
        fill : str, optional
            Pass 'tri' to triangulate the hole or 'ngon'
            to turn it into a polygonal face.

        Raises
        ------
        NonManifoldError
            If vertex deletion was not possible.

        Returns
        -------
        Face or list[Face]
            The faces used to patch the hole. :obj:`None` if hole
            filling was not requested.
        """
        v = self._verts[vertex]
        assert not v._deleted

        if v.isolated:
            v._deleted = True
        elif not v._manifold:
            if fill is not None:
                raise NonManifoldError('cannot fill hole formed by ' +
                                       'deleting a non-manifold vertex')

            # Collect all faces incident with vertex v. Use _vhout[v] to
            # to treat non-manifold vertices correctly.
            faces = [h.face for h in self._vhout[v] if h.face is not None]

            for f in faces:
                self.delete_face(f, del_isolated_verts=True)
        else:
            boundary = v.boundary

            while not v._deleted:
                h = v._halfedge
                left = h.face
                right = h.pair.face

                while True:
                    next = h.next
                    self.delete_halfedge(h, del_isolated_verts=True)

                    if next.face is left and next.pair.face is right:
                        h = next
                    else:
                        break

            if fill is None:
                if not boundary:
                    self.delete_face(right, del_isolated_verts=True)
            else:
                if not boundary:
                    if fill == 'tri':
                        raise NotImplementedError('fill method missing')
                else:
                    raise NotImplementedError('fill method missing')

    def delete_face(self, face, del_isolated_verts=True):
        """ Delete face.

        Vertices of `face` rendered isolated by the combinatorial face
        removal can be kept as isolated vertices or be marked as deleted.

        Parameters
        ----------
        face : Face or int
            Face identifier.
        del_isolated_verts : bool, optional
            Mark isolated vertices for deletion.

        Note
        ----
        The deleted face is not removed from the mesh's face container
        immediately. It is marked as deleted and removed from the face
        container when calling :meth:`clean`.
        """
        f = self._faces[face]
        assert not f._deleted

        # Get halfedges bordering the face. Nothing to do for an empty
        # face.
        edge_loop = [h for h in f._hiter()]

        # A boundary face gets merged with the adjacent boundary component.
        # Halfedges that connect the face with the boundary are deleted.
        for h in edge_loop:
            if not h._pair._face:
                self.delete_halfedge(h, del_isolated_verts)
            else:
                h._face = None

        # Finally mark the face as deleted.
        f._deleted = True

    def delete_halfedge(self, halfedge, del_isolated_verts=True):
        """ Merge adjacent faces.

        Merges the incident faces of an edge. Merging across a boundary
        edge will delete the incident non-boundary face. Vertices of
        `halfedge` rendered isolated by the change in mesh combinatorics
        can be kept as isolated vertices or be marked as deleted.

        Parameters
        ----------
        halfedge : Halfedge
            The halfedge that spans the edge to be deleted.
        del_isolated_verts : bool, optional
            Mark isolated vertices for deletion.

        Raises
        ------
        NonManifoldError
            If the operation leads to invalid combinatorics.

        Returns
        -------
        Face
            The merged face or :obj:`None` when merging across the
            boundary. For interior `halfedge`, this is equal to the
            face to its right.

        Note
        ----
        This operation may create *dangling (half)edges*, i.e., halfedges
        where ``h.pair`` equals ``h.next``.
        """
        assert not halfedge._deleted

        # If halfedge is a boundary halfedge we replace it with its pair
        # if this is not at the boundary. In case of dangling edges both
        # halfedges can be boundary halfedges.
        if halfedge._face is None and halfedge._pair._face is not None:
            halfedge = halfedge._pair

        # The face to the left of halfedge gets merged with the face to
        # the right. Neither halfedge nor its pair may be used as the
        # halfedge attribute of this face.
        if halfedge._pair._face is not None:
            # This path is only taken if neither halfedge nor its pair
            # are boundary halfedges.
            h = halfedge._pair._face._halfedge

            # Walk around the right hand side face and try to find an
            # alternative value for its halfedge attribute.
            while h is halfedge or h is halfedge._pair:
                h = h._next

                if h is halfedge._pair._face._halfedge:
                    break

            # If no alternative was found there is no way to continue
            # without corrupting the halfedge data structure.
            if h is halfedge or h is halfedge._pair:
                raise NonManifoldError('halfedge deletion failed')

            halfedge._pair._face._halfedge = h

        # Make sure that the deleted halfedge is not stored as the
        # outgoing halfedge of its origin.
        v = halfedge._origin

        for h in self._vhout[v]:
            if h is not halfedge:
                v._halfedge = h
                break

        # If no alternative outgoing halfedge could be found, vertex v
        # is a dangling vertex and becomes unused after the halfedge is
        # removed.
        if v._halfedge is halfedge:
            if del_isolated_verts:
                v._deleted = True
            else:
                v._halfedge = None

        # Perform the same steps for halfedge.pair since this will also
        # get removed.
        w = halfedge._pair._origin

        for h in self._vhout[w]:
            if h is not halfedge._pair:
                w._halfedge = h
                break

        if w._halfedge is halfedge._pair:
            if del_isolated_verts:
                w._deleted = True
            else:
                w._halfedge = None

        # The face that survives the merge operation. Can be None if a
        # boundary face gets deleted by the merge.
        face = halfedge._pair._face

        # The two faces indicent with halfedge get merged. If they are
        # equal the deleted halfedge is a dangling edge.
        if halfedge._pair._face is not halfedge._face:
            halfedge._face._deleted = True
            h = halfedge._next

            # All halfedges of the face to the left get assgined to
            # the face on the right hand side of the halfedge. We do
            # not change attributes of halfedge itself.
            while True:
                h._face = face
                h = h._next

                if h is halfedge:
                    break

        # Remove the pair of halfedges from face defining halfedge loops.
        # Then pop from the halfedge dictionary.
        halfedge._prev._next = halfedge._pair._next
        halfedge._pair._next._prev = halfedge._prev

        halfedge._next._prev = halfedge._pair._prev
        halfedge._pair._prev._next = halfedge._next

        self._pop_halfedge(halfedge)
        self._pop_halfedge(halfedge._pair)

        assert not v._deleted or not self._vhout[v]
        assert not w._deleted or not self._vhout[w]

        assert v._halfedge is not None or not self._vhout[v]
        assert w._halfedge is not None or not self._vhout[w]

        return face

    def collapse_halfedge(self, halfedge, point=None, del_target=True, *,
                          check=True):
        """ Perform edge collapse.

        Collapse `halfedge` into its :attr:`~Halfedge.origin` vertex. By
        default, the :attr:`~Halfedge.target` vertex of `halfedge` is
        marked as deleted but it can also be kept as an isolated vertex.

        Parameters
        ----------
        halfedge : Halfedge
            Halfedge to be contracted.
        point : array_like, optional
            Coordinates of collapse location.
        del_target : bool, optional
            Mark target vertex as deleted.
        check : bool, optional
            Pass :obj:`False` to skip collapsibility test.

        Raises
        ------
        ValueError
            If `halfedge` is a boundary halfedge.

        Returns
        -------
        Vertex
            Reference to the :attr:`~Halfedge.origin` of `halfedge` or
            :obj:`None` in case of failure (the latter behavior requires
            ``check=True``).

        Note
        ----
        It is assumed that the applicability of an edge collapse has been
        checked for explicitly via :attr:`~Halfedge.collapsible` when
        skipping the test.
        """

        def prepare_he_loop(h, set_origin_halfedge=True):
            # The length of a face defining loop of halfedges. For a
            # boundary halfedge this is the length of the boundary.
            loop_len = h._compute_loop_len()

            # If the face to the left is a triangle it will disappear
            # during the halfedge collapse operation.
            if loop_len == 3:
                # There has to be a face, otherwise the edge cannot be
                # collapsed because of topological problems.
                h._face._deleted = True

                # The vertex opposite the halfedge. Ensure its outgoing
                # halfedge is valid after the collapse.
                v = h._next._target
                v._halfedge = h._next._pair

                # Remove halfedges of the interior edge loop from the
                # halfedge container.
                self._pop_halfedge(h._prev)
                self._pop_halfedge(h._next)

            # Lazy property management. This could go inside an else
            # block (no effect for triangular faces).
            if h._face is not None:
                h._face._valence = None

            # Remove halfedge from halfedge dictionary. Sets its status
            # to deleted.
            self._pop_halfedge(h)

            # Ensure the outgoing halfedges of origin stays valid after
            # the halfedge is collapsed.
            if set_origin_halfedge:
                h._origin._halfedge = h._prev._pair

            return loop_len

        def glue_he_faces(h):
            # Glue previous and next halfedge of an edge of a triangle.
            h._prev._pair._pair = h._next._pair
            h._next._pair._pair = h._prev._pair

        # Early exit if halfedge cannot be collapsed. Return None value to
        # signal failure.
        if halfedge._deleted or (check and not halfedge.collapsible):
            return None

        if halfedge._face is None:
            raise ValueError('boundary halfedge cannot be collapsed - ' +
                             'use its pair')

        lt_len = prepare_he_loop(halfedge)
        rt_len = prepare_he_loop(halfedge._pair, set_origin_halfedge=False)

        origin = halfedge._origin
        target = halfedge._target

        # All halfedges starting at target now start at origin. The target
        # vertex itself will be unused after the collapse is complete.
        for h in target._hiter():
            # Do not modify combinatorial attributes of halfedges that are
            # no longer used by the new mesh. This makes it easy to undo a
            # halfedge collapse operation later.
            if h is not halfedge._pair:
                if lt_len > 3 or h is not halfedge._next:
                    self._set_origin(h, origin)

                if rt_len > 3 or h._pair is not halfedge._pair._prev:
                    self._set_target(h._pair, origin)

        # Now either glue two halfedges together to delete a neighboring
        # triangle or skip a halfedge if case of larger face valence.
        if lt_len == 3:
            glue_he_faces(halfedge)
        else:
            halfedge._prev._next = halfedge._next
            halfedge._next._prev = halfedge._prev

        if rt_len == 3:
            glue_he_faces(halfedge._pair)
        else:
            halfedge._pair._prev._next = halfedge._pair._next
            halfedge._pair._next._prev = halfedge._pair._prev

        # The target vertex is now disconnected from the mesh. Keep it
        # as isolated vertex or mark it for deletion.
        if del_target:
            # Mark target vertex as deleted. Note that the vertex keeps
            # its outgoing halfedge.
            halfedge._target._deleted = True
        else:
            halfedge._target._halfedge = None

        # For consistency of the hds state, the set of outgoing halfedges
        # has to be empty, no matter if we mark it as deleted or not.
        assert not self._vhout[halfedge._target]

        # Move vertex v to its new location. Can raise ValueError if
        # point could not be broadcast to the correct point shape.
        if point is not None:
            halfedge._origin.point = point

        return halfedge._origin

    def flip_halfedge(self, halfedge, *, check=True):
        """ Flip halfedge.

        Only halfedges incident to triangular faces can be flipped.

        Parameters
        ----------
        halfedge : Halfedge
            The halfedge to be flipped.
        check : bool, optional
            Pass :obj:`False` to skip flippability test.

        Returns
        -------
        Halfedge
            The halfedge resulting from the edge flip or :obj:`None` in
            case of failure (the latter behavior requires ``check=True``).

        Note
        ----
        It is assumed that the applicability of an edge flip has been checked
        explicitly via :attr:`~Halfedge.flippable` when skipping the test.
        Unchecked edge flipping results in undefined behavior.
        """
        if halfedge._deleted or (check and not halfedge.flippable):
            return None

        v = halfedge._next._target
        w = halfedge._pair._next._target

        # This is an extra test layer. The flippable method of a halfedge
        # should have flagged such an edge as non-flippable.
        if (v, w) in self._halfs:
            return None

        origin = halfedge._origin
        target = halfedge._target

        a, b = halfedge._next, halfedge._prev
        c, d = halfedge._pair._next, halfedge._pair._prev

        # Unconditional re-assignment of halfedge attributes. This is not
        # always necessary and could be turned in a conditional assignment.
        origin._halfedge = c
        target._halfedge = a

        self._set_vertices(halfedge, w, v)
        self._set_vertices(halfedge._pair, v, w)

        halfedge._next, halfedge._prev = b, c
        halfedge._pair._next, halfedge._pair._prev = d, a

        a._next, a._prev = halfedge._pair, d
        d._next, d._prev = a, halfedge._pair

        b._next, b._prev = c, halfedge
        c._next, c._prev = halfedge, b

        a._face = halfedge._pair._face
        c._face = halfedge._face

        halfedge._face._halfedge = halfedge
        halfedge._pair._face._halfedge = halfedge._pair

        return halfedge

    def insert_halfedge(self, face, origin, target):
        """ Insert face diagonal.

        Splits `face` into two polygonal parts.

        Parameters
        ----------
        face : Face
            Polygonal face of a mesh.
        origin : Vertex
            Origin vertex, incident with `face`.
        target : Vertex
            Target vertex, incident with `face` and
            different from `origin`.

        Raises
        ------
        ValueError
            In case of invalid input.
        NonManifoldError
            If something goes wrong.

        Returns
        -------
        Halfedge
            The newly inserted halfedge. The face to its left is equal
            to `face`.

        Note
        ----
        This method cannot be used to split off parts of a mesh's
        boundary. Use :meth:`~Mesh.add_face` for that.
        """
        if origin not in face:
            msg = f'origin vertex {origin} is no vertex of {face}'
            raise ValueError(msg)

        if target not in face:
            msg = f'target vertex {target} is no vertex of {face}'
            raise ValueError(msg)

        if (origin, target) in self._halfs:
            msg = f'halfedge ({origin}, {target}) already exists'
            raise NonManifoldError(msg)

        if origin is target:
            msg = 'halfedge vertices have to be different'
            raise NonManifoldError(msg)

        # Create and add two new halfedges to the halfedge container.
        # This will establish the halfedges pair pointers.
        h = self._add_halfedge(origin, target)
        hbar = self._add_halfedge(target, origin)

        # Find the next and previous halfedges of h and hbar among the
        # existing halfeges of face.
        h_next = next(h for h in face._hiter() if h.origin is target)
        hbar_next = next(h for h in face._hiter() if h.origin is origin)

        hbar_prev = h_next._prev
        h_prev = hbar_next._prev

        # Take care of the face to the left of h. This face keeps its
        # identifier. A new face is create to the right of it.
        h._next = h_next
        h_next._prev = h
        h._prev = h_prev
        h_prev._next = h
        h._face = face

        face._halfedge = h
        face._valence = None

        # Take care of the face to the right of h. This is a new face that
        # takes over some of the existing halfedges of the original face.
        fbar = Face(len(self._faces))
        fbar._halfedge = hbar

        self._faces.append(fbar)
        self._add_attr_values(fbar, *self._fattr)

        hbar._next = hbar_next
        hbar_next._prev = hbar
        hbar._prev = hbar_prev
        hbar_prev._next = hbar

        # The halfedge loop around the new face has been established. Set
        # the face attribute of each halfedge in this loop to fbar.
        while True:
            hbar._face = fbar
            hbar = hbar._next

            if fbar._halfedge is hbar:
                break

        return h

    def split_halfedge(self, halfedge, point=None, triangulate=True):
        """ Split halfedge.

        Subdivides an edge by inserting a new vertex. The resulting
        polygonal faces to the left and right of the edge are
        triangulated on request.

        Parameters
        ----------
        halfedge : Halfedge
            Halfedge to split.
        point : array_like, optional
            Coordinates of the inserted vertex. By default the halfedges'
            :attr:`~Halfedge.midpoint` is used.
        triangulate : bool, optional
            Triangle fan like triangulation of neighboring faces.

        Returns
        -------
        Vertex
            The newly inserted vertex.
        """
        assert not halfedge._deleted

        # Take the edge midpoint as vertex location if none is given.
        if point is None:
            point = halfedge.midpoint

        # Get the original endpoints of the halfedge. Add the new point
        # as a vertex of the mesh.
        u = halfedge._origin
        v = self.add_vertex(point)
        w = halfedge._target

        h = self._add_halfedge(v, w)
        h._face = halfedge._face
        h._pair = halfedge._pair
        h._prev = halfedge
        h._next = halfedge._next
        h._next._prev = h

        if h._face is not None:
            h._face._valence = None

        v._halfedge = h

        hh = self._add_halfedge(v, u)
        hh._face = halfedge._pair._face
        hh._pair = halfedge
        hh._prev = halfedge._pair
        hh._next = halfedge._pair._next
        hh._next._prev = hh

        if hh._face is not None:
            hh._face._valence = None

        # ... this change has to be reflected in the halfedge dictionary
        # by reinserting them with a new key.
        del self._halfs[u, w]
        del self._halfs[w, u]

        self._halfs[u, v] = halfedge
        self._halfs[w, v] = halfedge._pair

        # Take care of the combinatorial attributes of halfedge and its
        # pair. Since one of their endpoints changed ...
        halfedge._pair._pair = h
        halfedge._pair._target = v
        halfedge._pair._next = hh

        halfedge._pair = hh
        halfedge._target = v
        halfedge._next = h

        # The split has resulted in polygonal faces to left and right of
        # halfedge. On request, these faces are triangulated.
        if triangulate:
            self._fan_triangulation(h)
            self._fan_triangulation(hh)

        return v

    def _add_attr_values(self, key, *args, **kwargs):
        """ Add attribute values.

        Extend data block by a value provided as keyword argument
        or the by the default value of the data block.

        Parameters
        ----------
        key : Vertex or Halfedge or Face
            The mesh item.
        *args : list
            List of data block descriptors.
        **kwargs : dict
            Dictionary of attribute names and values.
        """
        for name, attr, default in args:
            data = getattr(self, name)
            value = kwargs.get(attr, default)

            if isinstance(data, list):
                data.append(value)
            elif isinstance(data, dict):
                data[key] = value
            elif isinstance(data, np.ndarray):
                obj._array_append(data, value)

    def _add_halfedge(self, v, w, **kwargs):
        """ Create and add new halfedge.

        Generate new halfedge and take care of its :attr:`~Halfedge.origin`,
        :attr:`~Halfedge.target`, and :attr:`~Halfedge.pair` attributes. The
        :attr:`~Halfedge.next`, :attr:`~Halfedge.prev`, and
        :attr:`~Halfedge.face` attributes retain their default :obj:`None`
        values.

        Parameters
        ----------
        v : Vertex
            Origin vertex of the halfedge.
        w : Vertex
            Target vertex of the halfedge
        **kwargs
            Attribute name and value pairs.

        Raises
        ------
        NonManifoldError
            If there are topological issues adding the halfedge.
        TypeError
            If managed attributes could not be set.

        Returns
        -------
        Halfedge
            Halfedge pointing from `v` to `w`.

        Note
        ----
        If the opposite halfedge is already mapped, its :attr:`~Halfedge.pair`
        attribute is set accordingly. This is an internal helper function for
        face creation.
        """
        # Need instances of the Vertex class, not integers. Vertices have
        # to belong to the mesh instance.
        assert isinstance(v, Vertex) and v._mesh is self
        assert isinstance(w, Vertex) and w._mesh is self

        # This edge is topologically degenerate if the origin and target
        # vertex coincide. Not to be confused with geometrically degenerate
        # if the vertex locations coincide.
        if v is w:
            msg = f'topologically degenerate edge ({v._idx}, {w._idx})'
            raise NonManifoldError(msg)

        # In case we are re-using previously deleted vertices. The set of
        # outgoing halfedges has to be empty.
        assert not v._deleted or not self._vhout[v]
        assert not w._deleted or not self._vhout[w]

        try:
            # If the edge (v, w) is already mapped we have a topological
            # problem unless it is mapped as a boundary halfedge.
            h = self._halfs[v, w]

            if h._face is not None:
                msg = f'edge ({v._idx}, {w._idx}) is non-manifold'
                raise NonManifoldError(msg)
        except KeyError:
            # The edge is not mapped yet. Create it and set its pair
            # pointer if the pair has been mapped.
            h = Halfedge(v, w)
            h._pair = self._halfs.get((w, v), None)

            # If the pair is mapped, also set its pair pointer to the
            # newly created halfedge.
            if h._pair is not None:
                h._pair._pair = h

            # Insert h into the dictionary of all halfedges and add it
            # to the set of outgoing halfedges of its origin vertex.
            self._halfs[v, w] = h
            self._vhout[v].add(h)
            self._add_attr_values(h, *self._hattr, **kwargs)

            # Attributes that are not managed via a corresponding mesh
            # data block become ordinary halfedge instance attributes.
            for key, value in kwargs.items():
                # This calls the setter method of a property with the name
                # 'key' if it exists (guard with if-condition to prevent
                # setting the value of managed attributes again).
                # if key not in (attr for _, attr, _ in self._hattr):
                setattr(h, key, value)

        v._deleted = False
        w._deleted = False

        return h

    def _pop_halfedge(self, h):
        """ Remove halfedge from halfedge container.

        Called as a helper function whenever a halfedge is deleted during
        topological mesh modification. By design, a valid halfedge structure
        **never** contains deleted halfedges in its halfedge dictionary.

        Parameters
        ----------
        h : Halfedge
            The halfedge to be removed.

        Raises
        ------
        KeyError
            If the halfedge could not be removed. This indicates a corrupted
            halfedge data structure or erroneous code, e.g. trying to remove
            a halfedge twice.

        Note
        ----
        Combinatorial halfedge attributes are neither invalidated nor changed
        in any way.
        """
        assert not h._deleted

        # In case there are external references to this halfedge we set its
        # deleted flag. In this way user code can check if a halfedge is
        # still valid without searching through the halfedge container.
        h._deleted = True

        v = h._origin
        w = h._target

        # Remove the halfedge from the dictionary that holds all halfedges.
        # Also remove it from the set of outgoing halfedges of its origin.
        # Both operations can a KeyError if the halfedge is not present.
        del self._halfs[v, w]
        self._vhout[v].remove(h)

    def _push_halfedge(self, h):
        """ Add halfedge to halfedge container.

        (Re)insert existing halfedge into the dictionary of all halfedges.

        Parameters
        ----------
        h : Halfedge
            Halfedge to be re-inserted.

        Note
        ----
        This method should **only** be applied to halfedges that have
        previously been removed by :meth:`_pop_halfedge`.
        """
        assert h._deleted

        # Assumes that h has been removed by _pop_halfedge. Undo all the
        # changes that were made in said function.
        v = h._origin
        w = h._target

        assert (v, w) not in self._halfs
        assert h not in self._vhout[v]

        h._deleted = False

        self._halfs[v, w] = h
        self._vhout[v].add(h)

    def _set_origin(self, h, v):
        """ Set halfedge origin vertex.

        In-place connectivity modification.

        Parameters
        ----------
        h : Halfedge
            Valid mesh halfedge.
        v : Vertex
            Origin vertex, different from ``h.target``.
        """
        assert not h._deleted

        u = h._origin
        w = h._target

        assert v is not w
        assert h not in self._vhout[v]
        assert (v, w) not in self._halfs

        self._vhout[u].remove(h)
        self._vhout[v].add(h)

        del self._halfs[u, w]
        self._halfs[v, w] = h

        h._origin = v

    def _set_target(self, h, v):
        """ Set halfedge target vertex.

        In-place connectivity modification.

        Parameters
        ----------
        h : Halfedge
            Valid mesh halfedge.
        v : Vertex
            Target vertex, different from ``h.origin``.
        """
        assert not h._deleted

        u = h._origin
        w = h._target

        assert v is not u
        assert (u, v) not in self._halfs

        del self._halfs[u, w]
        self._halfs[u, v] = h

        h._target = v

    def _set_vertices(self, h, v, w):
        """ Set halfedge vertices.

        In-place connectivity modification. No halfedge attributes except
        :attr:`~Halfedge.origin` and :attr:`~Halfedge.target` are changed.

        Parameters
        ----------
        h : Halfedge
            Halfedge of a mesh, not marked as deleted.
        v : Vertex
            Halfedge origin vertex.
        w : Vertex
            Halfedge target vertex, different from `v`.
        """
        assert v is not w

        self._pop_halfedge(h)

        h._origin = v
        h._target = w

        self._push_halfedge(h)

    def _fan_triangulation(self, halfedge):
        """ Triangulate face with triangle fan.

        The faces of the triangle fan can be traversed with a simple
        loop:

        .. code-block:: python

            stop =
            h = halfedge

            while h is not stop:
                f = h.face
                h = h.prev.pair


        Parameters
        ----------
        halfedge : Halfedge
            Origin of `halfedge` defines the apex of the fan.

        Returns
        -------
        Halfedge
            Fan delimiting halfedge, equal to ``halfedge.prev.pair``
            before triangulation.
        """
        if halfedge._face is None:
            return

        v = halfedge._origin
        f = halfedge._face
        h = halfedge._next

        # Need to copy that value before it gets changed by the
        # edge insertion method.
        stop_here = halfedge._prev._prev

        while h is not stop_here:
            h = self.insert_halfedge(f, v, h._target)
            h = h._next

    def _clone_connectivity_from(self, other):
        """ Clone mesh connnectivity.

        Generates shallow copies of user defined vertex and face
        attributes, i.e., all attributes not managed by the halfedge
        structure.

        Parameters
        ----------
        other : Mesh
            Template mesh connectivity.

        Returns
        -------
        vmap : dict
            Maps vertices of `other` to `self`.
        hmap : dict
            Maps halfedges of `other` to `self`.
        fmap : dict
            Maps faces of `other` to `self`.
        """
        # The following way of copying a mesh is not intended, use the
        # copy() method instead.
        if self is other:
            msg = 'self cannot serve as template mesh connectivity'
            raise ValueError(msg)

        # Replace vertex and face containers with shallow copies of
        # vertices and faces of the template mesh.
        self._verts[:] = [copy(v) for v in other._verts]
        self._faces[:] = [copy(f) for f in other._faces]

        # Clear halfedge related containers. This wipes the mesh
        # combinatorics completely.
        self._halfs.clear()
        self._vhout.clear()

        # Maps that assign each item of the template mesh the
        # corresponding item of the cloned mesh.
        vmap = {v: w for v, w in zip(other._verts, self._verts)}
        fmap = {f: g for f, g in zip(other._faces, self._faces)}
        hmap = dict()

        # Clone halfedges. Need cloned vertices to generate halfedges.
        # Sanity check: halfedges should never be marked as deleted.
        for (v, w), h in other._halfs.items():
            assert not h._deleted

            v_new = vmap[v]
            w_new = vmap[w]
            h_new = Halfedge(v_new, w_new)
            hmap[h] = h_new
            self._halfs[v_new, w_new] = h_new

        # Set halfedge attribute of cloned vertices. Since this loops
        # over all cloned vertices we can also set their mesh attribute.
        for v in other._verts:
            v_new = vmap[v]
            v_new._mesh = self
            v_new._halfedge = (hmap[v._halfedge]
                               if v._halfedge is not None else None)
            self._vhout[v_new] = set()

            for h in other._vhout[v]:
                self._vhout[v_new].add(hmap[h])

        # Copy the faces. Only one pass needed, requires valid halfedge
        # map.
        for f in other._faces:
            f_new = fmap[f]
            f_new._halfedge = (hmap[f._halfedge]
                               if f._halfedge is not None else None)

        # Set remaining attributes of cloned halfedges with the help of
        # the face and halfedge maps.
        for h in other._halfs.values():
            h_new = hmap[h]
            h_new._next = hmap[h._next]
            h_new._prev = hmap[h._prev]
            h_new._pair = hmap[h._pair]
            h_new._face = fmap[h._face] if h._face is not None else None

        return vmap, hmap, fmap

    def _check(self, contains_test=False):
        """ Perform sanity checks.
        """
        for v, halfs in self._vhout.items():
            if not halfs:
                assert v._deleted or v._halfedge is None

            for h in halfs:
                assert h in self._halfs.values()
                assert h._origin is v

        for (v, w), h in self._halfs.items():
            assert h._origin is v
            assert h._target is w
            assert h._pair in self._halfs.values()
            assert h in self._vhout[v]

            h._check(contains_test)

        for v in self._verts:
            assert self._verts[v._idx] is v

            if v._deleted:
                assert not self._vhout[v]
            else:
                if v._halfedge is None:
                    assert not self._vhout[v]
                else:
                    assert v._halfedge in self._vhout[v]
                    assert v._halfedge in self._halfs.values()

            v._check(contains_test)

        for f in self._faces:
            assert self._faces[f._idx] is f

            if not f._deleted:
                assert f._halfedge in self._halfs.values()

            f._check(contains_test)

        for private_name, _, _ in self._vattr:
            assert len(getattr(self, private_name)) == len(self._verts)

        for private_name, _, _ in self._hattr:
            assert len(getattr(self, private_name)) == len(self._halfs)

        for private_name, _, _ in self._fattr:
            assert len(getattr(self, private_name)) == len(self._faces)

    def _viter(self):
        """ Generator expression skipping deleted vertices.
        """
        return (v for v in self._verts if not v._deleted)

    def _viter_frozen(self):
        """ Generator expression skipping deleted vertices.
        """
        return iter(list(self._viter()))

    def _fiter(self):
        """ Generator expression skipping deleted faces.
        """
        return (f for f in self._faces if not f._deleted)

    def _fiter_frozen(self):
        """ Generator expression skipping deleted faces.
        """
        return iter(list(self._fiter()))

    def _hiter(self):
        """ Generator expression.
        """
        return iter(self._halfs.values())

    def _hiter_frozen(self):
        """ Generator expression.
        """
        return iter(list(self._halfs.values()))

    def _eiter(self):
        """ Generator expression.
        """
        return (h for h in self._halfs.values()
                if h._origin._idx < h._target._idx)

    def _eiter_frozen(self):
        """ Generator expression.
        """
        return iter(list(self._eiter()))


class Vertex:
    """ Vertex base class.

    Vertices are considered as abstract topological entities. Vertex
    coordinates are assigned when a vertex becomes part of a mesh. Its
    coordinates can then be accessed via the :attr:`point` property .

    Parameters
    ----------
    index : int
        Vertex index.
    parent : Mesh, optional
        The parent mesh object.

    Note
    ----
    In addition to :attr:`index`, implementations of the special functions
    :meth:`~object.__int__` and :meth:`~object.__index__` are provided.
    The latter makes it possible to use vertex instances as list indices.
    """

    def __init__(self, index, parent=None):
        # Initialize combinatorial/topological attributes.
        self._idx = index
        self._mesh = parent
        self._halfedge = None

        # Initialize internal state attributes.
        self._deleted = False
        self._flags = flags.VertexFlag(0)

    def __repr__(self):
        return f'Vertex({self._idx})'

    def __str__(self):
        try:
            point = self.point
        except AttributeError:
            point = '[None]'

        if self._flags:
            return f'v {self._idx} {point} {self._flags}'

        return f'v {self._idx} {point}'

    def __index__(self):
        """ Vertex index.

        Vertices can be used directly as list and array indices, i.e.,
        one can write ``some_list[v]`` instead of the slightly longer
        ``some_list[v.index]`` expression.

        Returns
        -------
        int
            Vertex index.
        """
        return self._idx

    def __int__(self):
        """ Vertex index.

        The expression ``int(v)`` is equivalent to ``v.index``.

        Returns
        -------
        int
            Vertex index.
        """
        return self._idx

    def __bool__(self):
        return True

    def __array__(self, dtype=None, copy=None):
        """ Experimental NumPy support.

        There has been a change in the signature of __array__ from
        NumPy version 1.26 to 2.0 (the signature used here).

        Parameters
        ----------
        dtype : data-type, optional
            The desired data type for the array.
        copy : bool, optional
            If :obj:`True` then the array data is copied. If :obj:`None`,
            a copy will only be made if necessary. For :obj:`False` it
            raises a :class:`ValueError` if a copy cannot be avoided.

        Raises
        ------
        ValueError
            If `copy` evaluates to :obj:`False` and a copy cannot be
            avoided.

        Returns
        -------
        ~numpy.ndarray
            Array of vertex coordinates.

        Note
        ----
        If no copy is requested (or implied by data type conversion) the
        returned value is a view of the mesh's vertex coordinate array.
        """
        return np.array(self._mesh._points[self._idx, ...],
                        dtype=dtype, copy=copy)

    @property
    def index(self):
        """ Vertex index.

        Position of the vertex in the list :attr:`~Mesh.vertices` of all
        mesh vertices. Same as ``int(self)``.

        :type: int
        """
        return self._idx

    @property
    def point(self):
        """ Vertex coordinates.

        Read and write access to vertex coordinates. View of the
        vertex coordinate array. Requires a valid parent mesh.

        :type: ~numpy.ndarray

        Note
        ----
        This attribute can be set from any :term:`array_like` vertex
        coordinate representation. Coordinates are assigned to the
        corresponding row of the parent mesh's coordinate array. In
        particular, NumPy's :term:`broadcast` rules apply.
        """
        return self._mesh._points[self._idx, ...]

    @point.setter
    def point(self, value):
        self._mesh._points[self._idx, ...] = value

    @property
    def flags(self):
        """ Vertex flags.

        Read and write access to vertex flags.

        :type: VertexFlag
        """
        return self._flags

    @flags.setter
    def flags(self, value):
        self._flags = value

    @property
    def halfedge(self):
        """ Outward pointing halfedge.

        A halfedge that starts at the vertex or :obj:`None` for isolated
        vertices.

        :type: Halfedge
        """
        assert not self._deleted
        return self._halfedge

    @property
    def degree(self):
        """ Vertex degree.

        The number of adjacent vertices, equivalent to the number of
        incident edges -- also called the valence of a vertex.

        :type: int
        """
        assert not self._deleted
        assert self._manifold
        assert len(self._mesh._vhout[self]) == self._compute_degree()
        return len(self._mesh._vhout[self])

    @property
    def deleted(self):
        """ Internal state.

        Topological mesh modification (e.g., edge conctractions) render
        vertices as deleted when they do no longer contribute to a mesh's
        combinatorics.

        :type: bool

        Note
        ----
        Calling :meth:`~Mesh.clean` removes all deleted vertices from the
        vertex container of a mesh.
        """
        return self._deleted

    @property
    def boundary(self):
        """ Topological state.

        A vertex is defined to be a boundary vertex if it is incident to
        a boundary halfedge.

        :type: bool
        """
        assert not self._deleted
        return any(h.boundary for h in self._hiter())

    @property
    def isolated(self):
        """ Topological state.

        A vertex is isolated if its :attr:`~Vertex.halfedge` attribute
        holds a :obj:`None` value. Topologically, isolated vertices are
        not linked to any other mesh items and form their own connected
        component.

        :type: bool

        Note
        ----
        Isolated vertices are treated as **manifold** vertices. Their
        presence does not influence the halfedge data structure
        functionality negatively.
        """
        assert not self._deleted

        h = self._halfedge
        vhout = self._mesh._vhout[self]

        # If h is set it has to be a member of vhout. In particular this
        # imples that vhout is not empty if h is set.
        assert h is None or h in vhout

        # If h is not set the set vhout has to be empty.
        assert h is not None or not vhout

        return self._halfedge is None

    @property
    def _manifold(self):
        """ Topological state.

        Isolated vertices are considered manifold vertices as they
        don't break the halfedge structure.

        :type: bool
        """
        assert not self._deleted

        # We cannot rely on self.halfedge since this allows us only to
        # visit one fan of triangles attached to the vertex. The point
        # is to find out if there is more than one such triangle fan.
        halfs = self._mesh._vhout[self]

        if not halfs:                       # isolated vertex is fine
            return True

        # There should be either one closed triangle fan or one open
        # triangle fan attached. There can never be more than two None
        # faces in such a triangle fan (in case of a boundary vertex).
        faces = [h.face for h in halfs] + [h.pair.face for h in halfs]
        count = faces.count(None)

        if count == 0 or count == 2:
            # Interior vertex: faces has to form a single closed fan of
            # triangles. Boundary vertex: single open fan of triangles.
            fmap = {h.face: h.pair.face for h in halfs}
            loop = [faces[0]] if count == 0 else [None]

            while fmap:
                # Want to see the KeyError, this should never happen
                # and indicates unexpected behavior.
                loop.append(fmap.pop(loop[-1]))

                if loop[0] is loop[-1]:
                    break

            # Everything is fine if we reach this point and the loop has
            # closed while fmap was exhausted.
            if loop[0] is loop[-1] and not fmap:
                return True

        return False

    def _compute_degree(self):
        """ Vertex degree.

        Compute vertex degree in the canonical way. Only applicable
        to manifold vertices.

        Returns
        -------
        int
            Vertex degree.

        Note
        ----
        This is an internal function only used for debugging. To access
        the vertex degree always use :attr:`degree`.
        """
        # Dangling vertices, i.e., endpoints of dangling edges are the
        # only vertices with degree 1. A vertex should only be in this
        # transitional state during mesh construction (or modification).
        return sum(1 for _ in self._viter())

    def _check(self, contains_test):
        """ Perform sanity checks.

        Parameters
        ----------
        contains_test : bool
            Enable time consuming ``__contains__()`` tests.
        """
        if not self._deleted:
            if self._halfedge is not None:
                assert not self._halfedge._deleted
                assert self._halfedge._origin is self
                assert self.degree == self._compute_degree()

                if self._halfedge._face is not None:
                    assert not self._halfedge._face._deleted

                    if contains_test:
                        assert self in self._halfedge._face

    def _invalidate(self):
        """ Reset all combinatorial attributes.

        Invalidate should be called as a last step in garbage collection
        when removed from the corresponding container object. This should
        make it easier to find bugs originating from using references to
        deleted objects.
        """
        assert self._deleted

        self._idx = None
        self._mesh = None
        self._halfedge = None

    def _viter(self):
        """ Adjacent vertex iterator.
        """
        assert not self._deleted
        h = self._halfedge

        if h is None:
            return

        while True:
            yield h._target
            h = h._prev._pair

            if h is self._halfedge:
                return

    def _fiter(self):
        """ Incident face iterator.
        """
        assert not self._deleted
        h = self._halfedge

        if h is None:
            return

        while True:
            if h._face is not None:
                yield h._face

            h = h._prev._pair

            if h is self._halfedge:
                return

    def _hiter(self):
        """ Incident halfedge iterator.
        """
        assert not self._deleted
        h = self._halfedge

        if h is None:
            return

        while True:
            yield h
            h = h._prev._pair

            if h is self._halfedge:
                return


class Halfedge:
    """ Halfedge base class.

    Halfedges store references to their vertices, the successor, predecessor,
    and twin halfedge as well as the incident face -- the face to its left.
    A closed loop of halfedges defines a face and its orientation. Successor
    and predecessor refer to the next and previous halfedge in such a loop.

    Parameters
    ----------
    origin : Vertex
        Origin vertex of the halfedge.
    target : Vertex
        Target vertex of the halfedge.

    Note
    ----
    As halfedges are stored in a dictionary and not in a list, they do not
    have a canonical index value but a key that is formed by the pair of
    origin and target vertex.
    """

    def __init__(self, origin, target):
        self._origin = origin
        self._target = target

        self._next = None
        self._prev = None
        self._pair = None
        self._face = None

        self._deleted = False
        self._flags = flags.HalfedgeFlag(0)

    def __repr__(self):
        return f'Halfedge({repr(self._origin)}, {repr(self._target)})'

    def __str__(self):
        if self._flags:
            return (f'h ({self._origin._idx}, {self._target._idx})' +
                    f' {self._flags}')

        return f'h ({self._origin._idx}, {self._target._idx})'

    def __bool__(self):
        return True

    # def __array__(self, dtype=None, copy=None):
    #     """ Experimental NumPy support.

    #     There has been a change in the signature of __array__ from
    #     NumPy version 1.26 to 2.0 (the signature used here).

    #     Parameters
    #     ----------
    #     dtype : data-type, optional
    #         The desired data type for the array.
    #     copy : bool, optional
    #         If :obj:`True` then the array data is copied. If :obj:`None`,
    #         a copy will only be made if necessary. For :obj:`False` it
    #         raises a :class:`ValueError` if a copy cannot be avoided.

    #     Raises
    #     ------
    #     ValueError
    #         If `copy` evaluates to :obj:`False` and a copy cannot be
    #         avoided.

    #     Returns
    #     -------
    #     ~numpy.ndarray
    #         Coordinates of halfedge direction vector.
    #     """
    #     return self.vector

    def __contains__(self, vertex):
        """ Incidence test.

        Parameters
        ----------
        vertex : Vertex
            Vertex to be tested.

        Returns
        -------
        bool
            :obj:`True` if `vertex` is one of the halfedge's vertices.
        """
        # The functionality would still be there even if this method was
        # removed because __iter__ is defined.
        assert not self._deleted
        return (vertex is self._origin) or (vertex is self._target)

    def __iter__(self):
        """ Vertex iterator.

        Produces the origin and target vertex of a halfedge.

        Yields
        ------
        Vertex
            Next vertex.
        """
        yield self.origin
        yield self.target

    def __getitem__(self, index):
        """ Vertex access.

        Vertex access by relative index.

        Parameters
        ----------
        index : int
            Relative vertex index, either 0 or 1.

        Raises
        ------
        IndexError
            If `index` is out of bounds.

        Returns
        -------
        Vertex
            Origin or target vertex of a halfedge.
        """
        assert not self._deleted

        if index == 0:
            return self._origin
        elif index == 1:
            return self._target

        raise IndexError(f'index {index} out of range(0, 2)')

    @property
    def origin(self):
        """ Halfedge origin vertex.

        :type: Vertex
        """
        assert not self._deleted
        return self._origin

    @property
    def target(self):
        """ Halfedge target vertex.

        :type: Vertex
        """
        assert not self._deleted
        return self._target

    @property
    def vector(self):
        """ Halfedge direction vector.

        The vector ``self.target.point - self.origin.point``.

        :type: ~numpy.ndarray
        """
        assert not self._deleted
        return self._target.point - self._origin.point

    @property
    def midpoint(self):
        """ Halfedge midpoint.

        The point ``0.5 * (self.origin.point + self.target.point)``.

        :type: ~numpy.ndarray
        """
        assert not self._deleted
        return 0.5 * (self._origin.point + self._target.point)

    @property
    def next(self):
        """ Successor halfedge.

        Next halfedge in a face defining halfedge loop.

        :type: Halfedge
        """
        assert not self._deleted
        return self._next

    @property
    def prev(self):
        """ Predecessor halfedge.

        Previous halfedge in a face defining halfedge loop.

        :type: Halfedge
        """
        assert not self._deleted
        return self._prev

    @property
    def pair(self):
        """ Opposite halfedge.

        Halfedge pointing in the opposite direction.

        :type: Halfedge
        """
        assert not self._deleted
        return self._pair

    @property
    def face(self):
        """ Incident face.

        The face to left of the halfedge or :py:obj:`None` in case of
        a boundary halfedge.

        :type: Face
        """
        assert not self._deleted
        return self._face

    @property
    def flags(self):
        """ Halfedge flags.

        Read and write access to halfedge flags.

        :type: HalfedgeFlag
        """
        return self._flags

    @flags.setter
    def flags(self, value):
        self._flags = value

    @property
    def deleted(self):
        """ Internal state.

        Topological mesh modification (e.g., edge conctractions) may
        render halfedges as deleted when they do no longer contribute
        to a mesh's combinatorics.

        :type: bool

        Note
        ----
        The halfedge dictionary :attr:`~Mesh.halfedges` of a mesh will
        never contain deleted halfedges.
        """
        return self._deleted

    @property
    def boundary(self):
        """ Topological state.

        A halfedge is called a boundary halfedge if its :attr:`face`
        attribute evaluates to :obj:`None`.

        :type: bool
        """
        assert not self._deleted
        return self._face is None

    @property
    def collapsible(self):
        """ Topological state.

        A edge joining non-boundary vertices of a **triangle mesh** is
        collapsible if the neighborhoods of :attr:`origin` and :attr:`target`
        vertex intersect in the two vertices opposite the query edge. The
        test tries to handle meshes with higher valence faces but may not
        always give a correct answer.

        :type: bool

        Note
        ----
        For technical reasons, **boundary halfedges** are always classified
        as non-collapsible.
        """

        def one_sided_check(h):
            assert h._face is not None
            assert h._pair._face is None

            if h._pair._compute_loop_len() == 3:
                # The adjacent boundary loop has only three faces.
                # Collapsing the halfedge would change the topology.
                return False

            if len(h._face) == 3:
                if v_neigh.intersection(w_neigh) == {h._next._target}:
                    return True

                return False

            return True

        # Test makes no sense for deleted halfedges, they are not part
        # of valid mesh combinatorics.
        if self._deleted or self._face is None:
            return False

        v = self._origin
        w = self._target

        # This should never be a problem for pure triangle meshes but
        # can happen for general polygonal meshes.
        v_faces = {x for x in v._fiter()} - {self._face, self._pair._face}
        w_faces = {x for x in w._fiter()} - {self._face, self._pair._face}

        if v_faces.intersection(w_faces):
            return False

        # Collect neighbors in a set to compute one-ring intersection.
        v_neigh = {x for x in v._viter() if x is not w}
        w_neigh = {x for x in w._viter() if x is not v}

        if self._face is None:
            return one_sided_check(self._pair)

        if self._pair._face is None:
            return one_sided_check(self)

        # Interior edge that connects boundaries is not collapsible.
        # Result would be a non-manifold mesh.
        if v.boundary and w.boundary:
            return False

        if len(self._face) == 3:
            p = self._next._target

            if len(self._pair._face) == 3:
                q = self._pair._next._target

                if p is q:
                    return False

                if v_neigh == w_neigh:
                    return False

                # Triangular faces to the left and right. One rings of
                # endpoints have to intersect in the vertices opposite
                # the query edge.
                if v_neigh.intersection(w_neigh) == {p, q}:
                    return True
            else:
                # Triangular face to the left and n-gon to the right.
                if v_neigh.intersection(w_neigh) == {p}:
                    return True
        else:
            # Triangular face to the right and n-gon to the left.
            if len(self._pair._face) == 3:
                q = self._pair._next._target

                if v_neigh.intersection(w_neigh) == {q}:
                    return True

            # There are n-gons to the left and to the right of the
            # query edge.
            return True

        return False

    @property
    def flippable(self):
        """ Topological state.

        A non-boundary edge of a triangle mesh can be flipped if the
        vertices opposite the edge are not adjacent.

        :type: bool
        """
        if self._deleted or self._face is None or self._pair._face is None:
            return False

        # An interior edge can only be flipped if the incident faces are
        # triangles.
        if len(self._face) != 3 or len(self._pair._face) != 3:
            return False

        # The vertices opposite the query edge may not belong to the same
        # face. Happens if one of the edge's endpoints is of degree three.
        v = self._next._target
        w = self._pair._next._target

        # if (v, w) in self._origin._mesh._halfs:
        #     return False

        # return True
        return not any(w in f for f in v._fiter())

    def _compute_loop_len(self):
        """ Length of halfedge loop.

        Length of the closed halfedge loop starting at ``self``. For a
        boundary halfedge this is the length of the corresponding boundary
        curve.

        Returns
        -------
        int
            Length of halfedge loop.
        """
        assert not self._deleted

        loop_len = 0
        h = self

        while True:
            loop_len += 1
            h = h._next

            if h is self:
                return loop_len

    def _check(self, contains_test):
        """ Perform sanity checks.

        Parameters
        ----------
        contains_test : bool
            Enable time consuming ``__contains__()`` tests.
        """
        assert not self._deleted
        assert not self._pair._deleted
        assert not self._origin._deleted
        assert not self._target._deleted

        assert self._pair._pair is self
        assert self._pair._origin is self._target
        assert self._pair._target is self._origin

        assert self._next._prev is self
        assert self._next._origin is self._target
        assert self._next._pair._target is self._target

        assert self._prev._next is self
        assert self._prev._target is self._origin
        assert self._prev._pair._origin is self._origin

        if self._face is not None:
            assert not self._face._deleted

            if contains_test:
                assert self in self._face
                assert self._origin in self._face
                assert self._target in self._face

        assert not (self._face is None and self._pair._face is None)

    def _invalidate(self):
        """ Reset all combinatorial attributes.
        """
        assert self._deleted

        self._origin = None
        self._target = None

        self._pair = None
        self._next = None
        self._prev = None
        self._face = None

    def _fiter_lnk(self):
        """ Vertex-adjacent face iterator.
        """
        assert not self._deleted
        h = self

        while True:
            if h._face is not None:
                yield h._face

            h = h._prev._pair

            if h is self:
                break

        h = self._pair._prev._pair

        while h is not self._next:
            if h._face is not None:
                yield h._face

            h = h._prev._pair


class Face:
    """ Face base class.

    In a halfedge based mesh representation a face is defined by the
    closed loop of halfedges starting at the :attr:`halfedge` attribute.

    Parameters
    ----------
    index : int
        Face index.


    The vertices of a face can be visited in several ways. Using
    :meth:`~m3sh.hds.Face.__len__` and :meth:`~m3sh.hds.Face.__getitem__`

    .. code-block:: python
       :linenos:

        for i in range(len(f)):
            print(f[i])

    is equivalent to using :meth:`~m3sh.hds.Face.__iter__`

    .. code-block:: python
       :linenos:

        for v in f:
            print(v)

    Note
    ----
    The latter is much more efficient and preferred.
    """

    def __init__(self, index):
        # Initialize combinatorial/topological attributes.
        self._idx = index
        self._halfedge = None

        # Initialize lazy attributes/properties.
        self._valence = None

        # Initialize internal state attributes.
        self._deleted = False
        self._flags = flags.FaceFlag(0)

    def __repr__(self):
        return f'Face({self._idx})'

    def __str__(self):
        face = '[None]' if self._deleted else str([int(v) for v in self])

        if self._flags:
            return f'f {self._idx} {face} {self._flags}'

        return f'f {self._idx} {face}'

    def __index__(self):
        """ Face index.

        Faces can be used directly as list and array indices.

        Returns
        -------
        int
            Face index.
        """
        return self._idx

    def __int__(self):
        """ Face index.

        The expression ``int(f)`` is equivalent to ``f.index``.

        Returns
        -------
        int
            Face index.
        """
        return self._idx

    def __len__(self):
        """ Face valence.

        The number of incident vertices.

        Returns
        -------
        int
            Number of vertices.
        """
        assert not self._deleted

        if self._valence is None:
            self._valence = self._compute_valence()
        else:
            assert self._valence == self._compute_valence()

        return self._valence

    def __bool__(self):
        return True

    def __array__(self, dtype=None, copy=None):
        """ Experimental NumPy support.

        There has been a change in the signature of __array__ from
        NumPy version 1.26 to 2.0 (the signature used here).

        Parameters
        ----------
        dtype : data-type, optional
            The desired data type for the array.
        copy : bool, optional
            If :obj:`True` then the array data is copied. If :obj:`None`,
            a copy will only be made if necessary. For :obj:`False` it
            raises a :class:`ValueError` if a copy cannot be avoided.

        Raises
        ------
        ValueError
            If `copy` evaluates to :obj:`False` and a copy cannot be
            avoided.

        Returns
        -------
        ~numpy.ndarray
            Array of vertex coordinates.
        """
        return np.array([v.point for v in self], dtype=dtype, copy=copy)

    def __contains__(self, item):
        """ Vertex and halfedge containment test.

        Parameters
        ----------
        item : Vertex or Halfedge
            Item to be tested for incidence with the face.

        Returns
        -------
        bool
            :obj:`True` if the given item belongs to the face.
        """
        return (item in self._viter()) or (item in self._hiter())

    def __iter__(self):
        """ Vertex iterator.

        The returned :term:`iterator` visits the vertices of ``self``
        starting with the ``self.halfedge.origin`` vertex.

        Yields
        ------
        Vertex
            Next vertex in counter-clockwise traversal.
        """
        assert not self._deleted
        assert self._halfedge is not None

        h = self._halfedge

        while True:
            yield h._origin
            h = h._next

            if h is self._halfedge:
                return

    def __getitem__(self, index):
        """ Vertex access.

        Vertex access by relative index, enumeration starts at
        ``self.halfedge.origin``.

        Parameters
        ----------
        index : int or slice
            Integer or slice object.

        Raises
        ------
        IndexError
            If `index` is outside the valid range.

        Returns
        -------
        Vertex
            The vertex at position `index`.
        """
        indices = range(len(self))[index]

        if isinstance(indices, int):
            return next(vert for i, vert in enumerate(self) if i == indices)
        else:
            if index.step is not None and index.step < 0:
                return [vert for i, vert in enumerate(self)
                        if i in indices][::-1]

            return [vert for i, vert in enumerate(self) if i in indices]

    @property
    def index(self):
        """ Face index.

        Position of the face in the list :attr:`~Mesh.faces` of all
        faces, same as ``int(self)``.

        :type: int
        """
        return self._idx

    @property
    def flags(self):
        """ Face flags.

        Read and write access to face flags.

        :type: FaceFlag
        """
        return self._flags

    @flags.setter
    def flags(self, value):
        self._flags = value

    @property
    def halfedge(self):
        """ Incident halfedge.

        One of the incident halfedges.

        :type: Halfedge
        """
        assert not self._deleted
        return self._halfedge

    @property
    def valence(self):
        """ Face valence.

        Number of incident vertices. Same as ``len(self)``.

        :type: int
        """
        return len(self)

    @property
    def deleted(self):
        """ Internal state.

        Topological mesh modification (e.g., edge conctractions) render
        faces as deleted when they do no longer contribute to a mesh's
        combinatorics.

        :type: bool

        Note
        ----
        Calling :meth:`~Mesh.clean` removes all deleted faces from the
        face container of a mesh.
        """
        return self._deleted

    @property
    def boundary(self):
        """ Topological state.

        A face is defined to be a boundary face if one of the incident
        edges is a boundary edge (an edge is a boundary edge if one of
        its two halfedges has this property).

        :type: bool

        Note
        ----
        A face only incident with boundary vertices is **not** classified
        as a boundary face.
        """
        return any(h.pair.boundary for h in self._hiter())

    @property
    def barycenter(self):
        """ Face barycenter.

        Arithmetic mean of vertex coordinates.

        :type: ~numpy.ndarray
        """
        return sum(v.point for v in self) / len(self)

    def _compute_valence(self):
        """
        """
        return sum(1 for _ in self._hiter())

    def _check(self, contains_test):
        """ Perform sanity checks.

        Parameters
        ----------
        contains_test : bool
            Enable time consuming ``__contains__()`` tests.
        """
        if not self._deleted:
            assert self._halfedge is not None
            assert not self._halfedge._deleted
            assert self._halfedge._face is self
            assert len(self) == self._compute_valence()

            if contains_test:
                assert self._halfedge in self

    def _invalidate(self):
        """ Reset all combinatorial attributes.
        """
        assert self._deleted

        self._idx = None
        self._halfedge = None
        self._valence = None

    def _viter(self):
        """ Incident vertex iterator.
        """
        assert not self._deleted
        assert self._halfedge is not None

        h = self._halfedge

        while True:
            yield h._origin
            h = h._next

            if h is self._halfedge:
                return

    def _hiter(self):
        """ Incident halfedge iterator.
        """
        assert not self._deleted
        assert self._halfedge is not None

        h = self._halfedge

        while True:
            yield h
            h = h._next

            if h is self._halfedge:
                return

    def _eiter(self):
        """ Generator expression.
        """
        return (h if h._origin._idx < h._target._idx else h.pair
                for h in self._hiter())

    def _fiter(self):
        """ Edge-adjacent face iterator.

        This iterator defines two faces to be adjacent if they share
        a common edge.
        """
        assert not self._deleted
        assert self._halfedge is not None

        h = self._halfedge

        while True:
            if h._pair._face is not None:
                yield h._pair._face

            h = h._next

            if h is self._halfedge:
                return

    def _fiter_lnk(self):
        """ Vertex-adjacent face iterator.

        This iterator defines two faces to be adjacent if they share
        a common vertex.
        """
        assert not self._deleted
        assert self._halfedge is not None

        h = self._halfedge._pair

        # Find an edge that points towards a vertex of the face that is
        # not an edge of the face.
        while h._pair._face is self:
            h = h._prev

        # Iterate over the incoming halfedges of vertices and yield the
        # adjacent face of such edges.
        while True:
            if h._face is not None:
                yield h._face

            h = h._pair._prev

            if h._pair is self._halfedge:
                return

            # We have reached a halfedge of the center face. Continue
            # with the incoming edges of the other vertex of the edge.
            while h._pair._face is self:
                h = h._prev

                if h._pair is self._halfedge:
                    return


class NonManifoldError(Exception):
    """ Manifold exception base class.

    Raised if an operation results in a topological configuration that
    violates the manifold condition.
    """

    pass
