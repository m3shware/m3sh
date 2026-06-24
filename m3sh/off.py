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

""" Object file format input/output.

.. version-added:: 1.1.0
"""

from pathlib import Path


def read(filename, quiet=True):
    """ Read from OFF file.

    Parameters
    ----------
    filename : str
        Name of OFF file.
    quiet : bool, optional
        Pass :obj:`False` to print comments and summary.

    Returns
    -------
    verts : list
        List of vertex definitions. Each vertex is a list of three
        scalars that define vertex coordinates in 3-space.
    faces : list
        List of face definitions. Each face is a list integers.
    """
    with open(filename, 'r') as file:
        filepos = 1

        if file.readline().strip() != 'OFF':
            raise ValueError('file does not conform to OFF specification')

        verts, vread = [], 0
        faces, fread = [], 0

        while line := file.readline():
            if tokens := line.split():
                if tokens[0].startswith('#'):
                    if not quiet:
                        print(line.rstrip())

                    continue

                filepos += 1

                if filepos == 2:
                    vcnt, fcnt = int(tokens[0]), int(tokens[1])
                elif vread < vcnt:
                    verts.append([float(token) for token in tokens])
                    vread += 1
                else:
                    valence = int(tokens[0])
                    faces.append([int(tokens[i])
                                  for i in range(1, valence + 1)])
                    fread += 1

    assert vread == vcnt
    assert fread == fcnt

    if not quiet:
        name = Path(filename).name

        print(f"reading {'\33[1m'}{name}{'\33[0m'} results in")
        print(f'\t├─ {vcnt} vertex definitions')
        print(f'\t└─ {fcnt} face definitions')

    return verts, faces
