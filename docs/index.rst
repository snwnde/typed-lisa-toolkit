Typed LISA Toolkit
==================

Typed LISA Toolkit (or TLT) is a collection of small, type-annotated python tools for
use in CU L2D. It is currently used in the `Gee-Moo`_ global fit prototype.

.. _Gee-Moo: https://gitlab.in2p3.fr/lisa-apc/global-fit

TLT is more than a data abstraction layer.
By basing your code on TLT objects, you get:

- A unified interface across different data array libraries
   Numpy, JAX, ...
- Semantically rich building blocks
   Get a TLT object and you are ready to work with it.
- No performance overhead
   Arrays remain first-class citizens and references are privileged over copies.
- Type safety and hints
   `Static type checking`_ improves code clarity and helps catch bugs.
   Your IDE guides you with type hints and autocompletion.
- Interoperability with other packages
   TLT follows the conventions defined in `l2d-interface`_.

.. _Static type checking: https://typing.python.org/en/latest/

.. _l2d-interface: https://l2d-interface-c43116.pages.in2p3.fr/index.html

.. toctree::
   :maxdepth: 2
   :caption: User Guides:

   userguides/quickstart
   userguides/installation
   userguides/tutorials

.. toctree::
   :maxdepth: 2
   :caption: Public API:
   :titlesonly:
   :hidden:

   api/toplevel
   api/types
   api/shop

.. toctree::
   :maxdepth: 2
   :caption: Internal API:
   :titlesonly:
   :hidden:

   api/internal/index
