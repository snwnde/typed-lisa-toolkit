Quick Start
-----------

Install and add as a dependency to your project:

.. code-block:: bash

   uv add git+https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit

Import TLT from python:

.. code-block:: python

   import typed_lisa_toolkit as tlt

(See the :ref:`installation guide` for other installation methods.)

Load Mojito data
~~~~~~~~~~~~~~~~
Mojito is a mock LISA data prepared by the DDPC. The downloading is handled by the
`mojito`_ package. Downloaded data are expected to be processed by `MojitoProcessor`_,
which can then be loaded by TLT as follows:

.. _mojito: https://mojito-e66317.io.esa.int/
.. _MojitoProcessor: https://ollieburke.github.io/MojitoProcessor/

.. code-block:: python

   import typed_lisa_toolkit as tlt
   processd_data = ...  # Process the downloaded Mojito data using MojitoProcessor
   mojito_data = tlt.load_mojito(processd_data)

Load LDC data
~~~~~~~~~~~~~
LDC stands for `LISA Data Challenges <https://lisa-ldc.in2p3.fr/>`_,
which are a series of simulated datasets designed by the LISA Consortium.

The format of the LDC datasets is defined in the package
`lisa-data-challenge <https://gitlab.in2p3.fr/LISA/LDC>`_.
TLT provides loaders for the LDC datasets.

For instance, assuming you have exported the environment variable ``SANGRIA_TRAINING_DATA`` 
pointing to the Sangria data path, you can load the data as follows:

.. code-block:: python

   import os
   import pathlib
   import typed_lisa_toolkit as tlt
   sangria_data_path = pathlib.Path(os.environ.get('SANGRIA_TRAINING_DATA'))
   sangria_data = tlt.load_sangria(sangria_data_path)
