# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add the project root to sys.path for autodoc.
# This file lives at docs/source/conf.py, so project root is two levels up.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Zephon"
copyright = "2026, DatologyAI"
author = "DatologyAI"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "Zephon"
html_css_files = ["theme.css", "figures.css"]
html_js_files = ["embeds.js"]

# Zephon's palette; dark values come from the figures, so a figure and the page
# around it agree. See _static/theme.css.
html_theme_options = {
    # The brand wordmark: mint for a dark ground, recoloured to teal for light.
    "light_logo": "zephon-wordmark-teal.png",
    "dark_logo": "zephon-wordmark-mint.png",
    "sidebar_hide_name": True,  # the wordmark already reads "zephon"
    "light_css_variables": {
        "color-brand-primary": "#096361",
        "color-brand-content": "#096361",
        "color-brand-visited": "#526e6b",
        "color-foreground-primary": "#112a2b",
        "color-foreground-secondary": "#526e6b",
        "color-background-secondary": "#f0f9f6",
        "color-background-border": "#cee2dd",
        "color-background-hover": "#e6f4f0",
        "color-table-border": "#cee2dd",
        "color-table-header-background": "#f0f9f6",
        "color-inline-code-background": "#f0f9f6",
        "color-highlight-on-target": "#d8f8f3",
        "color-api-name": "#096361",
        "color-api-pre-name": "#526e6b",
        "color-code-background": "#f0f9f6",
        "color-admonition-title--note": "#096361",
        "color-admonition-title-background--note": "#d8f8f3",
    },
    "dark_css_variables": {
        "color-brand-primary": "#00efb5",
        "color-brand-content": "#00efb5",
        "color-brand-visited": "#7e9b98",
        "color-background-secondary": "#15302f",
        "color-background-border": "#456260",
        "color-background-hover": "#1b3a38",
        "color-table-border": "#456260",
        "color-table-header-background": "#22403e",
        "color-inline-code-background": "#22403e",
        "color-highlight-on-target": "#1b3a38",
        "color-api-name": "#00efb5",
        "color-api-pre-name": "#7e9b98",
        "color-code-background": "#15302f",
        "color-admonition-title--note": "#00efb5",
        "color-admonition-title-background--note": "#1b3a38",
    },
}

# -- Extension configuration -------------------------------------------------

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 5

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    # Stays until the public API is documented: 76 public methods carry no
    # docstring, Pipeline.tokenize and .checkpoint among them.
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
# Only the parameters the docstring covers; the signature carries the rest.
autodoc_typehints_description_target = "documented_params"
autodoc_class_signature = "mixed"
autodoc_preserve_defaults = True
add_module_names = False
python_use_unqualified_type_names = True
toc_object_entries_show_parents = "hide"

# Napoleon configuration (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
# True un-skips every documented dunder, which put Pipeline.__getstate__ at the
# head of its page.
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Suppress warnings for missing references to optional dependencies
nitpicky = False
