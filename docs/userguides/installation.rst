.. _installation guide:

Installation
------------

We recommend using `uv <https://docs.astral.sh/uv/>`_ to manage your environment
and dependencies. In the venv, run the following command to install TLT:

.. code-block:: shell

   uv add typed-lisa-toolkit

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

Some features of TLT require additional dependencies. To install TLT with
an optional group of dependencies, you can use the ``--extra`` option:

.. code-block:: shell

   uv add typed-lisa-toolkit --extra jax

Or with another syntax:

.. code-block:: shell

   uv add 'typed-lisa-toolkit[wdm]'

The available extras are:

- ``jax``: JAX backend support.
- ``torch``: PyTorch backend support.
- ``wdm``: WDM transform support.

Development setup
~~~~~~~~~~~~~~~~~

If you want to contribute to the development of TLT, you can install it in
editable mode. With uv, you can follow these steps:

1. Clone the repository.

.. code-block:: shell

   git clone https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit.git

2. Change into the project directory.

.. code-block:: shell

   cd typed-lisa-toolkit

3. (*Optional*) Create a new virtual environment and activate it.

.. code-block:: shell

   uv venv
   source .venv/bin/activate

4. Run the following command:

.. code-block:: shell

   uv sync --all-extras --all-groups
