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
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx_toolbox.more_autodoc.autoprotocol",
    "sphinx_toolbox.more_autodoc.autonamedtuple",
]

python_use_unqualified_type_names = True

autosummary_generate = True
autosummary_ignore_module_all = False

autodoc_member_order = "bysource"
autodoc_typehints_description_target = "all"
autodoc_typehints = "signature"
autodoc_class_signature = "mixed"
autoclass_content = "class"

# Temporary solution for type aliases
autodoc_type_aliases = {
    "ChnName": "ChnName",
    "ArrayFunc": "ArrayFunc",
    "TaperT": "TaperT",
    "WaveformInChannel": "WaveformInChannel",
    "Waveform": "Waveform",
    "FormattedWaveform": "FormattedWaveform",
}

napoleon_use_admonition_for_notes = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_preprocess_types = True

# We need to get the python version and the numpy version for intersphinx.
import platform
import numpy as np

py_version_tuple = platform.python_version_tuple()
py_version = f"{py_version_tuple[0]}.{py_version_tuple[1]}"
np_version_tuple = np.version.version.split(".")
np_version = f"{np_version_tuple[0]}.{np_version_tuple[1]}"


intersphinx_mapping = {
    "python": (f"https://docs.python.org/{py_version}", None),
    "numpy": (f"https://numpy.org/doc/{np_version}/", None),
}


latex_engine = "xelatex"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_permalinks_icon = "§"
html_theme = "insipid"
html_static_path = ["_static"]
http_theme_options = {
    "body_max_width": None,
}
