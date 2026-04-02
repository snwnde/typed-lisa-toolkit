# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Typed LISA Toolkit"
copyright = "2024, Sen-wen Deng"
author = "Sen-wen Deng"

import importlib.metadata
import sys
from pathlib import Path

# Make _sphinx_handlers module available
sys.path.insert(0, str(Path(__file__).parent))

# The full version, including alpha/beta/rc tags
release = importlib.metadata.version("typed-lisa-toolkit")
# Take major.minor.patch
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "myst_nb",
    "sphinx.ext.mathjax",
    # "sphinx_autodoc_typehints",
]

# Source - https://stackoverflow.com/a/72658470
# Posted by Mikko Ohtamaa
# Retrieved 2026-04-02, License - CC BY-SA 4.0


python_use_unqualified_type_names = True

autosummary_generate = True
autosummary_ignore_module_all = False

autodoc_member_order = "bysource"
autodoc_typehints_description_target = "all"
autodoc_typehints = "signature"
autodoc_class_signature = "mixed"
autoclass_content = "class"

# Import autodoc handlers and transforms (see _sphinx_handlers.py)
from _sphinx_handlers import (
    TransformPrivateTypes,
    process_autodoc_docstring,
    process_autodoc_signature,
)

# Temporary solution for type aliases
# autodoc_type_aliases = {
#     "ChnName": "ChnName",
#     "Numeric": "Numeric",
#     "ArrayFunc": "ArrayFunc",
#     "Interpolator": "Interpolator",
#     "WaveformInChannel": "WaveformInChannel",
#     "Waveform": "Waveform",
#     "FormattedWaveform": "FormattedWaveform",
# }

extlinks = {
    "doi": ("https://dx.doi.org/%s", "doi:%s"),
}

napoleon_use_admonition_for_notes = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_preprocess_types = True

latex_engine = "xelatex"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_permalinks_icon = "§"
html_theme = "furo"


# -- Sphinx configuration ---------------------------------------------------


def setup(app):
    """Register autodoc event handlers and Sphinx transforms."""
    # app.connect("autodoc-process-signature", process_autodoc_signature)
    app.connect("autodoc-process-docstring", process_autodoc_docstring)
    app.add_transform(TransformPrivateTypes)
