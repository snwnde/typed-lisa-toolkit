Quick Start
-----------


Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install ./typed-lisa-toolkit


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
   from typed_lisa_toolkit.containers import data
   ldc_data_path = pathlib.Path(os.environ.get('SANGRIA_TRAINING_DATA'))
   ldc_data = data.load_ldc_data(ldc_data_path)