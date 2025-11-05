"""Module for mixins.

.. currentmodule:: typed_lisa_toolkit.lib.mixins

Classes
-------

.. autoclass:: NDArrayMixin
    :members:
"""

import logging
from typing import Self


import numpy as np
import numpy.lib.mixins


log = logging.getLogger(__name__)


class NDArrayMixin(numpy.lib.mixins.NDArrayOperatorsMixin):
    """Mixin class to enable numpy ufuncs on subclasses."""

    def square(self, **kwargs) -> Self:
        """Return the element-wise square."""
        return np.square(self, **kwargs)

    def exp(self, **kwargs) -> Self:
        """Return the element-wise exponential."""
        return np.exp(self, **kwargs)

    def sqrt(self, **kwargs) -> Self:
        """Return the element-wise square root."""
        return np.sqrt(self, **kwargs)

    def conj(self, **kwargs) -> Self:
        """Return the element-wise complex conjugate."""
        return np.conj(self, **kwargs)

    def abs(self, **kwargs) -> Self:
        """Return the element-wise absolute value."""
        return np.abs(self, **kwargs)
