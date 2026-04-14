.. _installation guide:

Installation
------------

TLT is not available yet in PyPI (this is planned for version 1.0.0).
Currently you need to point your package manager to the git repository to install it.

Adding as a dependency
~~~~~~~~~~~~~~~~~~~~~~

If uv is your package manager, you can run the following command:

.. code-block:: shell

   uv add git+https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit

Alternatively, you can add the following to the dependencies section of ``pyproject.toml``
manually:

.. code-block:: toml

   [project.dependencies]
   typed-lisa-toolkit = { git = "https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit.git" }

This makes TLT a dependency of your project, 
and when you install your project, it will be installed in the
same environment as your project.

Using pip
~~~~~~~~~

You can also install TLT using pip, but note that
it is your responsibility to choose the correct environment.

.. code-block:: shell

   pip install git+https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit


Development setup
^^^^^^^^^^^^^^^^^

For development, it is recommended to use uv.

1. Clone the repository.
2. Change into the project directory.
3. (*Optional*) Create a new virtual environment and activate it.

.. code-block:: shell

   uv venv
   source .venv/bin/activate

4. Run the following command:

.. code-block:: shell

   uv sync --all-extras --all-groups

This will install the package in editable mode.
