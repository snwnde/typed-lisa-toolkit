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
LDC data stands for data in the format given by the
`LISA Data Challenge <https://gitlab.in2p3.fr/LISA/LDC>`_.
This includes `Sangria` and `Sangria HM` datasets.

Assuming you have exported the environment variable ``SANGRIA_TRAINING_DATA`` 
pointing to the LDC data path, you can load the data as follows:

.. code-block:: python

   import os
   import pathlib
   import typed_lisa_toolkit as tlt
   ldc_data_path = pathlib.Path(os.environ.get('SANGRIA_TRAINING_DATA'))
   ldc_data = tlt.load_ldc_data(ldc_data_path)