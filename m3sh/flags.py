# Copyright 2022-2024, m3sh76
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

""" Mesh item flags.

Note
----
Support for mesh item flags is still experimental and subject to change.
User defined flags are missing. Python enumerations cannot be subclassed.
"""

from enum import Flag
from enum import auto


class VertexFlag(Flag):
    """ Vertex flags enumeration.
    """

    FIXED = auto()
    """ Fixed flag.

    Indicates that algorithms should not change vertex coordinates
    when this flag is set."""

    CORNER = auto()
    """ Corner flag. """


class HalfedgeFlag(Flag):
    """ Halfedge flags enumeration.
    """

    CREASE = auto()
    """ Crease flag. """

    SEAM = auto()
    """ Seam flag. """


class FaceFlag(Flag):
    """ Face flags enumeration (reserved).
    """

    pass
