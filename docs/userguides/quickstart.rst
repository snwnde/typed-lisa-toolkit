Quick Start
-----------


Installation
~~~~~~~~~~~~


Use as a dependency
^^^^^^^^^^^^^^^^^^^

Add the following to the dependencies section of `pyproject.toml`:

.. code-block:: toml

   [project.dependencies]
   typed-lisa-toolkit = { git = "https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit.git" }

.. note:: 
   The project is planned to be released to PyPI once version 1.0.0 is reached.

Development setup
^^^^^^^^^^^^^^^^^

Using `uv` is recommended.

1. Clone the repository.
2. Change into the project directory.
3. Run the following command:

.. code-block:: bash

   uv sync --all-extras --all-groups


Load LDC data
~~~~~~~~~~~~~
LDC stands for `LISA Data Challenges <https://lisa-ldc.in2p3.fr/>`_,
which are a series of simulated datasets designed by the LISA Consortium.

The format of the LDC datasets is defined in the package
`lisa-data-challenge <https://gitlab.in2p3.fr/LISA/LDC>`_.
In this package, we provide loaders for the LDC datasets.

For instance, assuming you have exported the environment variable ``SANGRIA_TRAINING_DATA`` 
pointing to the Sangria data path, you can load the data as follows:

.. code-block:: python

   import os
   import pathlib
   import typed_lisa_toolkit as tlt
   sangria_data_path = pathlib.Path(os.environ.get('SANGRIA_TRAINING_DATA'))
   sangria_data = tlt.load_sangria(sangria_data_path)


Load Mojito data
~~~~~~~~~~~~~~~~
Mojito is a mock LISA data prepared by the DDPC. The downloading is handled
by the `mojito <https://mojito-e66317.io.esa.int/>`_ package. Downloaded
data are expected to be processed by `MojitoProcessor <https://ollieburke.github.io/MojitoProcessor/>`_,
which can then be loaded by the toolkit as follows:

.. code-block:: python

   import typed_lisa_toolkit as tlt
   processd_data = ...  # Process the downloaded Mojito data using MojitoProcessor
   mojito_data = tlt.load_mojito(processd_data)