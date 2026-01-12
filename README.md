# m3sh - a mutable halfedge mesh data structure

![mesh](/docs/source/figures/dragon_full.png "Halfedge mesh representation")

A pure Python implementation of a generic halfedge data structure for
orientable 2-manifold meshes - the discrete analogue of orientable surfaces
in Euclidean 3-space. The package

- supports triangle and polygon meshes with dynamic connectivity
- import/export of object files (.obj) and
- includes a visualizer for rapid prototyping (optional, requires VTK).

## Quickstart

Copy the `m3sh` package folder (the one containing `__init__.py`) to a
location that is searched by Python when importing modules. This can be
achieved by putting the `m3sh` folder into your project folder:

    my-project/
    ├── project.py
    ┆
    └── m3sh/                   ← m3sh package folder
        ├── __init__.py
        ┆
        ├── hds.py              ← halfedge data structure
        └── vis.py              ← visualization module

The m3sh package can now be used in `project.py`. Complete API documentation
and examples can be found [here](https://m3shware.github.io/m3sh/usage.html).

## Dependencies

The halfedge data structure depends on NumPy 2.0 or higher. The visualization
module `m3sh.vis` requires a recent VTK version. It is recommended to install
packages in a dedicated environment. Using `conda`, an environment called
`m3sh-env` with all dependencies can be created and activated with

    conda create -n m3sh-env python numpy vtk
    conda activate m3sh-env

> VTK is only required if you want to use the `m3sh.vis` module. You can use
the full functionality of the halfedge data structure without installing VTK!

## Modifying the search path (optional)

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

If the location of the m3sh package folder relative to your `projects`
folder is different, the path needs to be adapted accordingly. See the
[sys.path](https://docs.python.org/3/library/sys_path_init.html)
documentation for more details.