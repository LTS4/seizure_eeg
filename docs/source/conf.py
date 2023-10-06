# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EEG pipeline for reproducible ML"
copyright = "2023, William Cappelletti"
author = "William Cappelletti"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

add_function_parentheses = False

# External sphinx doc referenced inside
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cupy": ("https://docs-cupy.chainer.org/en/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pandera": ("https://pandera.readthedocs.io/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []


master_doc = "index"
master_toc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "alabaster"
html_static_path = ["_static"]
html_static_path = ["_static"]
