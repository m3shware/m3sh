# m3sh - a mutable halfedge mesh data structure

![mesh](/docs/source/figures/dragon_full.png "Halfedge mesh representation")

A pure Python implementation of a generic halfedge data structure for
orientable 2-manifold meshes - the discrete analogue of orientable surfaces
in Euclidean 3-space.

## Quickstart

Download the code as a ZIP file (defaults to `m3sh-main.zip` or similar). After
extraction copy the `m3sh` package folder (the one containing `__init__.py`) to
a location that is searched by Python when importing modules. This can be achieved by putting the `m3sh` folder into your project folder:

    my-project/
    ├── project.py
    ┆
    └── m3sh/                   ← m3sh package folder
        ├── __init__.py
        ┆
        ├── hds.py              ← halfedge data structure
        └── vis.py              ← visualization module

The m3sh package can now be used in `project.py`. Make sure that your Python
installation provides all dependencies! Complete API documentation and more
quickstart examples can be found [here](https://m3shware.github.io/m3sh).

## Dependencies

Besides Python, the halfedge data structure provided by the m3sh package
depends only on NumPy 2.0 or higher. The visualization module `m3sh.vis`,
requires a recent VTK version (9.1 or higher is recommended).

> You can use the halfedge data structure without installing VTK. The
computational capabilities of the m3sh package do not suffer from the absence
of VTK!

It is recommended to work and install packages in a dedicated environment. Using
`conda`, all dependencies can be installed in an environment called `m3sh-env`
with

    conda create --name m3sh-env --channel conda-forge python numpy vtk

## Modifying the search path

To keep the m3sh package in a central location and make it accessible to
multiple projects one can add the package location to the search path. Assuming the directory structure

    projects/
    ├── project1/
    │   └── project1.py
    ├── project2/
    │
    ┆
    └── m3sh/                   ← m3sh package folder

this can be achieved by adding the following lines to a Python script (e.g. `project1.py`) before importing any modules from the m3sh package:

    import os
    import sys

    sys.path.insert(0, os.path.abspath('../.'))

If the location of the m3sh package folder relative to your project
folder is different, the path needs to be adapted accordingly. See the
[sys.path](https://docs.python.org/3/library/sys_path_init.html)
documentation for more details.