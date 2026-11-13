API reference
=============

**Version:** |release|

.. currentmodule:: m3sh

The core halfedge data structure, i.e., the :mod:`~m3sh.hds` module, only 
depends on the :mod:`~m3sh.obj` and :mod:`~m3sh.off` modules to facilitate 
input/output operations.


Core modules
------------

.. autosummary::
   :toctree: api
   :template: module-toc.rst

   hds
   traits
   itertools
   
   
Auxiliary, linear algebra, I/O
------------------------------
   
.. autosummary::
   :toctree: api
   :template: module-toc.rst
   
   linalg
   heap
   obj
   off
   
   
Visualization
-------------
  
.. autosummary::
   :toctree: api
   :template: module-toc-vtk.rst
   
   vis
