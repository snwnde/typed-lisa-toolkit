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
    'sphinx.ext.extlinks',
    "myst_nb",
    "sphinx.ext.mathjax",
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
    "Numeric": "Numeric",
    "ArrayFunc": "ArrayFunc",
    "Interpolator": "Interpolator",
    "WaveformInChannel": "WaveformInChannel",
    "Waveform": "Waveform",
    "FormattedWaveform": "FormattedWaveform",
}

extlinks = {
    'doi': ('https://dx.doi.org/%s', 'doi:%s'),
}

napoleon_use_admonition_for_notes = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_preprocess_types = True

# We need to get the python version and the numpy version for intersphinx.
# import platform
# import numpy as np
# import scipy.version  # type: ignore[import]

# py_version_tuple = platform.python_version_tuple()
# py_version = "{}.{}".format(*py_version_tuple[:2])
# np_version_tuple = np.version.version.split(".")
# np_version = "{}.{}".format(*np_version_tuple[:2])


# intersphinx_mapping = {
#     "python": (f"https://docs.python.org/{py_version}", None),
#     "numpy": (f"https://numpy.org/doc/{np_version}/", None),
#     "scipy": (
#         f"https://docs.scipy.org/doc/scipy-{scipy.version.version}/",
#         None,
#     ),
# }


latex_engine = "xelatex"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_permalinks_icon = "§"
html_theme = "furo"
