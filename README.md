# m3sh - a mutable halfedge mesh data structure

![mesh](/docs/source/figures/dragon_full.png "Halfedge mesh representation")

A pure Python implementation of a generic halfedge data structure for
orientable 2-manifold meshes - the discrete analogue of orientable surfaces
in Euclidean 3-space.

## Installation

In order to use m3sh its package folder `m3sh` needs to be in a location
that is searched by Python when loading modules. This can be achieved by
putting the package folder into your project folder, by putting it in a
location that is searched by default, or by adding the package location
to the search path. Assuming the directory structure

    projects/
    ├── project1/
    │   └── project1.py
    ├── project2/
    │
    ┆
    └── m3sh/

the latter can be achieved on a per project basis by adding the following
lines to a Python script (e.g. `project1.py`) before importing any modules
from the m3sh package:

    import os
    import sys

    sys.path.insert(0, os.path.abspath('../.'))

If the location of the m3sh package folder relative to your project
folder is different, the path needs to be adapted accordingly. See the
[sys.path](https://docs.python.org/3/library/sys_path_init.html)
documentation for more details. Alternatively you may set the
`PYTHONPATH` environment variable.

## Dependencies

The m3sh package has been succesfully used with Python 3.10 or higher and
NumPy 1.21 or higher. To use the visualization module `m3sh.vis`, a recent
VTK version has to be installed (VTK 9.1 or higher is recommended).
The computational capabilities of the m3sh package do not suffer from the
absence of VTK!

It is recommended to install packages in a dedicated environment. Using
`conda`, the necessary packages can be installed in an environment called
`m3sh` with

    conda create -n m3sh python numpy

## Documentation

Complete API documentation and quickstart examples can be found
[here](https://m3shware.github.io/m3sh).