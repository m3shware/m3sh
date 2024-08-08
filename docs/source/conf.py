# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath('../../.'))


# -- Project information -----------------------------------------------------

project = 'm3sh'
copyright = '2024, m3shware'
author = 'm3shware'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo'
]

# Set up intersphinx mapping to link to the python, numpy and scipy
# documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://scipy.github.io/devdocs', None)
}

# Turn on .rst file generation when using autosummary
autosummary_generate = True
autosummary_generate_overwrite = True

# Display todos by setting to True
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

toc_object_entries = False


# -- Options for AutoDoc output -------------------------------------------

# Members are listed as in the source file. Possible values are 
# 'alphabetical', 'groupwise', and 'bysource'.
# autodoc_member_order = 'bysource'
# autodoc_class_signature = 'separated'
# autoclass_content = 'both'

def skip(app, what, name, obj, skip, options):
    if name == "__init__":
        return True
    if name == "__new__":
        return True
    if 'vtkmodule' in str(obj):
        return True

    return None

def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'logo/logo-small.png'
html_favicon = 'icon/ghems_red.ico'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {
#    'navigation_depth': 3}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Do not offer to show html source files.
html_show_sourcelink = True
