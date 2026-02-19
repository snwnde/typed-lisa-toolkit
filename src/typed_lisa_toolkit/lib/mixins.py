# ruff: noqa
"""Module for mixins.

.. currentmodule:: typed_lisa_toolkit.lib.mixins

Classes
-------

.. autoclass:: NDArrayMixin
    :members:
"""

import logging
import abc
import operator
from typing import Self, Any


log = logging.getLogger(__name__)


__all__ = ["NDArrayMixin"]


class NDArrayMixin:
    """Mixin class to enable array ufuncs on subclasses.

    The class exposes small abstract hooks that concrete subclasses must
    implement:
    - __array_namespace__() -> Any: return the array API / namespace (numpy, cupy, etc.)
    - _binary_op(...): perform a binary operation and return an instance of Self
    - _unary_op(...): perform a unary operation and return an instance of Self
    """

    @abc.abstractmethod
    def __xp__(self, api_version: str | None = None) -> Any: ...

    @abc.abstractmethod
    def _binary_op(
        self,
        other: object,
        op: Any,
        /,
        reflected: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Self: ...

    @abc.abstractmethod
    def _unary_op(self, op: Any, /, *args: Any, **kwargs: Any) -> Self: ...

    @property
    def xp(self):
        """The underlying array namespace (numpy-like module)."""
        return self.__xp__()

    # comparisons
    def __lt__(self, other: object) -> Self:
        return self._binary_op(other, operator.lt)

    def __le__(self, other: object) -> Self:
        return self._binary_op(other, operator.le)

    def __eq__(self, other: object) -> Any:
        return self._binary_op(other, operator.eq)

    def __ne__(self, other: object) -> Any:
        return self._binary_op(other, operator.ne)

    def __gt__(self, other: object) -> Self:
        return self._binary_op(other, operator.gt)

    def __ge__(self, other: object) -> Self:
        return self._binary_op(other, operator.ge)

    # numeric methods
    def __add__(self, other: object) -> Self:
        return self._binary_op(other, operator.add)

    def __radd__(self, other: object) -> Self:
        return self._binary_op(other, operator.add, reflected=True)

    def __iadd__(self, other: object) -> Self:
        return self._binary_op(other, operator.iadd, inplace=True)

    def __sub__(self, other: object) -> Self:
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: object) -> Self:
        return self._binary_op(other, operator.sub, reflected=True)

    def __isub__(self, other: object) -> Self:
        return self._binary_op(other, operator.isub, inplace=True)

    def __mul__(self, other: object) -> Self:
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: object) -> Self:
        return self._binary_op(other, operator.mul, reflected=True)

    def __imul__(self, other: object) -> Self:
        return self._binary_op(other, operator.imul, inplace=True)

    def __matmul__(self, other: object) -> Self:
        return self._binary_op(other, operator.matmul)

    def __rmatmul__(self, other: object) -> Self:
        return self._binary_op(other, operator.matmul, reflected=True)

    def __imatmul__(self, other: object) -> Self:
        return self._binary_op(other, operator.imatmul, inplace=True)

    def __truediv__(self, other: object) -> Self:
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other: object) -> Self:
        return self._binary_op(other, operator.truediv, reflected=True)

    def __itruediv__(self, other: object) -> Self:
        return self._binary_op(other, operator.itruediv, inplace=True)

    def __floordiv__(self, other: object) -> Self:
        return self._binary_op(other, operator.floordiv)

    def __rfloordiv__(self, other: object) -> Self:
        return self._binary_op(other, operator.floordiv, reflected=True)

    def __ifloordiv__(self, other: object) -> Self:
        return self._binary_op(other, operator.ifloordiv, inplace=True)

    def __mod__(self, other: object) -> Self:
        return self._binary_op(other, operator.mod)

    def __rmod__(self, other: object) -> Self:
        return self._binary_op(other, operator.mod, reflected=True)

    def __imod__(self, other: object) -> Self:
        return self._binary_op(other, operator.imod, inplace=True)

    def __pow__(self, other: object) -> Self:
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other: object) -> Self:
        return self._binary_op(other, operator.pow, reflected=True)

    def __ipow__(self, other: object) -> Self:
        return self._binary_op(other, operator.ipow, inplace=True)

    def __lshift__(self, other: object) -> Self:
        return self._binary_op(other, operator.lshift)

    def __rlshift__(self, other: object) -> Self:
        return self._binary_op(other, operator.lshift, reflected=True)

    def __ilshift__(self, other: object) -> Self:
        return self._binary_op(other, operator.ilshift, inplace=True)

    def __rshift__(self, other: object) -> Self:
        return self._binary_op(other, operator.rshift)

    def __rrshift__(self, other: object) -> Self:
        return self._binary_op(other, operator.rshift, reflected=True)

    def __irshift__(self, other: object) -> Self:
        return self._binary_op(other, operator.irshift, inplace=True)

    def __and__(self, other: object) -> Self:
        return self._binary_op(other, operator.and_)

    def __rand__(self, other: object) -> Self:
        return self._binary_op(other, operator.and_, reflected=True)

    def __iand__(self, other: object) -> Self:
        return self._binary_op(other, operator.iand, inplace=True)

    def __xor__(self, other: object) -> Self:
        return self._binary_op(other, operator.xor)

    def __rxor__(self, other: object) -> Self:
        return self._binary_op(other, operator.xor, reflected=True)

    def __ixor__(self, other: object) -> Self:
        return self._binary_op(other, operator.ixor, inplace=True)

    # unary methods
    def __neg__(self) -> Self:
        return self._unary_op(operator.neg)

    def __pos__(self) -> Self:
        return self._unary_op(operator.pos)

    def __abs__(self) -> Self:
        return self._unary_op(operator.abs)

    def __invert__(self) -> Self:
        return self._unary_op(operator.invert)

    # convenience elementwise methods that delegate to the array namespace
    def square(self, **kwargs: Any) -> Self:
        return self._unary_op(self.xp.square, **kwargs)

    def exp(self, **kwargs: Any) -> Self:
        return self._unary_op(self.xp.exp, **kwargs)

    def sqrt(self, **kwargs: Any) -> Self:
        return self._unary_op(self.xp.sqrt, **kwargs)

    @property
    def conj(self) -> Self:
        return self._unary_op(self.xp.conj)

    @property
    def real(self) -> Self:
        return self._unary_op(self.xp.real)

    @property
    def imag(self) -> Self:
        return self._unary_op(self.xp.imag)

    def abs(self, **kwargs: Any) -> Self:
        return self._unary_op(self.xp.abs, **kwargs)

    def angle(self, **kwargs: Any) -> Self:
        return self._unary_op(self.xp.angle, **kwargs)

    def unwrap(self, **kwargs: Any) -> Self:
        return self._unary_op(self.xp.unwrap, **kwargs)
