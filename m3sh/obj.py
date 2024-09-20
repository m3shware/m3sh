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

""" OBJ file I/O.

Low-level functions to read and write OBJ files. Only a subset of the
OBJ standard is supported. Complete specifications can be found in the
`Advanced Visualizer Manual`.
"""

import numpy as np


def _array_append(array, item):
    """ Resize and append to array.

    Passing :obj:`None` as `item` will **not** initialize the newly
    added array entries. The `array` argument cannot be :obj:`None`
    in this case.

    Parameters
    ----------
    array : ~numpy.ndarray or None
        Array object to be augmented. A new array of shape
        ``(1, *item.shape)`` will be created if :obj:`None`.
    item : array_like or None
        Item to be added as new element of the first axis. The
        shapes ``array.shape[1:]`` and ``item.shape`` have to agree.

    Raises
    ------
    ValueError
        In case of dimension mismatch.

    Returns
    -------
    ~numpy.ndarray
        Reference to the enlarged array. This is a new array if the
        input array argument was :obj:`None`.
    """
    if isinstance(array, np.ndarray):
        if item is not None:
            if array[-1].shape != np.shape(item):
                msg = f'cannot add item with shape {np.shape(item)}'
                raise ValueError(msg)

        arr_shape = list(array.shape)
        arr_shape[0] += 1

        array.resize(arr_shape, refcheck=False)
    else:
        array = np.empty((1, *np.shape(item)))

    # Assign to the 'free' space at the end of the extended array.
    # The assignment itself should not trigger any exceptions.
    if item is not None:
        array[-1, ...] = item

    return array


def _array_clear(array):
    """ Collapse first axis of array.

    Parameters
    ----------
    array : ~numpy.ndarray
        Array with at least two axes.

    Returns
    -------
    ~numpy.ndarray
        The resized array.
    """
    arr_shape = list(array.shape)
    arr_shape[0] = 0

    array.resize(arr_shape, refcheck=False)

    return array


def _map_orientation(forward, up):
    """ Compute coordinate transformation matrix.

    Compute rotation that maps the given up direction to the z-axis
    of the world coordinate system and the forward direction to the
    y-axis.

    Parameters
    ----------
    forward : str
        Forward vector. Mapped to world :math:`y`-direction.
    up : str
        Up vector. Mapped to world :math:`z`-direction.

    Raises
    ------
    ValueError
        If ``up`` and ``forward`` values are invalid.

    Returns
    -------
    ~numpy.ndarray
        Matrix of shape (3, 3) that maps local/model coordinates to
        world coordinates given the local/model up and foward vectors.

    Note
    ----
    The values 'y' forward and 'z' up imply the identity matrix.
    Blender uses this particular forward/up directions by default.
    """
    up_sign = -1 if '-' in up else 1
    fw_sign = -1 if '-' in forward else 1

    if 'x' in up:
        z = np.array([up_sign, 0, 0])
    elif 'y' in up:
        z = np.array([0, up_sign, 0])
    elif 'z' in up:
        z = np.array([0, 0, up_sign])
    else:
        raise ValueError(f"invalid up direction '{up}'")

    if 'x' in forward:
        y = np.array([fw_sign, 0, 0])
    elif 'y' in forward:
        y = np.array([0, fw_sign, 0])
    elif 'z' in forward:
        y = np.array([0, 0, fw_sign])
    else:
        raise ValueError(f"invalid forward direction '{forward}'")

    x = np.cross(y, z)

    if np.dot(x, x) < 0.5:
        msg = (f"directions '{up}' and '{forward}' cannot be " +
                "used together as up and forward vectors")
        raise ValueError(msg)

    # Stacking the computed vectors as rows of matrix will create a
    # permutation matrix (rotation) that preserves the handedness of
    # the coordinate system.
    return np.stack((x, y, z))


def read(filename, *args):
    """ Read from file.

    Assumes an OBJ-like file structure, i.e., a text file where each
    line starts with a tag. Lines whose tag is contained in `args` are
    read. The returned data blocks store line data along their first
    axis. Data blocks are returned in the same order as given in `args`.
    If no corresponding data is found in the file the requested data
    block is represented as :obj:`None`.

    Parameters
    ----------
    filename : str
        Name of an OBJ file.
    *args
        Variable number of arguments of type :class:`str`.

    Raises
    ------
    ValueError
        If any argument is not of type :class:`str`.

    Returns
    -------
    object or tuple(object, ...)
        Data blocks corresponding to line tags given in `args`.


    To read vertices and vertex normals from an OBJ file do

    >>> v, vn = read('input-file.obj', 'v', 'vn')

    Data blocks are returend as objects of type :class:`~numpy.ndarray`.
    This assumes that data associated with a specific tag is homogeneous.
    The exception being the 'f' tag returning ``list[list[int]]``.
    """

    def parse(block):
        """ Parse vertex definition.

        Returned values can be negative (relative offsets). If positive,
        indices are 1-based.

        Parameters
        ----------
        block : str
            A v/vt/vn string representing a vertex definition as
            encountered when reading 'f' statements.

        Raises
        ------
        ValueError
            If the string could not be parsed.

        Returns
        -------
        v : int or None
            Vertex index.
        vt : int or None
            Vertex texture index.
        vn : int or None
            Vertex normal index.
        """
        # Default return values. If v could not be assigned there is
        # a problem with the input.
        v, vt, vn = None, None, None

        if '//' in block:
            bits = block.split('//')

            # A v//vn statement is split by // into exactly two parts.
            # The definition is invalid in all other cases.
            if len(bits) == 2:
                v, vn = (int(bit) for bit in bits)
            else:
                msg = 'invalid v//vn definition: ' + block
                raise ValueError(msg)
        elif '/' in block:
            bits = block.split('/')

            # A v/vt or v/vt/vn statement depending on how many parts
            # it gets split into by the / separator.
            if len(bits) == 2:
                v, vt = (int(bit) for bit in bits)
            elif len(bits) == 3:
                v, vt, vn = (int(bit) for bit in bits)
            else:
                msg = 'invalid v/vt/vn or v/vt definition: ' + block
                raise ValueError(msg)
        else:
            # Base case, only v given. This will raise ValueError if
            # block cannot be converted to an integer value.
            v = int(block)

        return v, vt, vn

    # At least one argument has to be given. Nothing to read otherwise.
    if not args:
        return None

    # All input arguments have to be of type str.
    if any((not isinstance(arg, str) for arg in args)):
        raise ValueError("arguments have to be of type 'str'")

    # Ouput data blocks for all arguments stored in a dictionary, keyed
    # on the corresponding tag.
    args_arr = {arg: [] if arg == 'f' else None for arg in args}

    # The number of encountered vertex coordinates. Needed to resolve
    # negative (relative) vertex indices.
    vcnt = 0

    with open(filename, 'r') as file:
        for line in file:
            # Extract runs of non-whitespace characters, the split()
            # method will also strip all whitespace ...
            blocks = line.split()

            # ... then process the contents of non-empty lines.
            if blocks:
                # Unconditional increment of the number of vertices
                # encountered up to this point.
                if blocks[0] == 'v':
                    vcnt += 1

                if blocks[0] in args:
                    if blocks[0] == 'f':
                        face = [parse(block)[0] for block in blocks[1:]]

                        for i in range(len(face)):
                            # Negative vertex indices are more complicated
                            # to handle. One needs to know the number of
                            # vertices read up this point...
                            if face[i] < 0:
                                face[i] = vcnt + face[i]
                            else:
                                face[i] -= 1

                        args_arr['f'].append(face)
                    else:
                        data = [float(block) for block in blocks[1:]]
                        arr = args_arr[blocks[0]]
                        args_arr[blocks[0]] = _array_append(arr, data)

    if len(args) == 1:
        return args_arr[args[0]]

    # This works as intended because dictionary values are iterated over
    # in insertion order (guaranteed since version 3.7 of Python).
    return tuple(args_arr.values())


def write(filename, *, f=None, **data):
    """ Write to file.

    Face data is expected as a nested list. Each entry of a face (a
    vertex definition) can be a single integer or a 3-tuple of integers.
    Tuple entries are interpreted as v/vt/vn triples, missing entries
    have to be specified with a :obj:`None` value.

    Parameters
    ----------
    filename : str
        Name of output file.
    f : list
        Face definitions.
    **data
        Keyword arguments.


    Data blocks to be stored in the file are passed via keyword arguments:

    >>> mesh.write('output-file.obj', line_tag=data_block)

    This assumes that ``data_block`` can be interpreted as a 2-dimensional
    array. The contents of each row are written to a line that starts
    with the given tag.
    """
    faces = [] if f is None else f

    with open(filename, 'w') as file:
        for key, value in data.items():
            for row in value:
                file.write(key)

                for element in row:
                    file.write(f' {element}')

                file.write('\n')

        for face in faces:
            file.write('f')

            for vertex in face:
                try:
                    v = vertex[0] + 1
                except TypeError:
                    file.write(f' {int(vertex) + 1}')
                else:
                    vt = '' if vertex[1] is None else vertex[1] + 1
                    vn = '' if vertex[2] is None else vertex[2] + 1

                    if vertex[2] is not None:
                        file.write(f' {v}/{vt}/{vn}')
                    else:
                        if vertex[1] is not None:
                            file.write(f' {v}/{vt}')
                        else:
                            file.write(f' {v}')

            file.write('\n')
