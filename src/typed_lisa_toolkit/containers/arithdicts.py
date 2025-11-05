"""Module for arithmetic dictionaries.

Arithmetic dictionaries are dictionaries that support arithmetic operations
between them and with numeric values or arrays. For example, a dictionary
that maps gravitational wave modes to their waveforms can be multiplied by a
dictionary that maps these modes to some scaling factors. To this end, we use
have :class:`.ModeDict`. In LISA data analysis, we can multiply a dictionary
that maps channels to simplified responses to strain waveforms to get TDI
waveforms. For this, we have :class:`.ChannelDict`.

.. currentmodule:: typed_lisa_toolkit.containers.arithdicts

Types
-----
.. autoclass:: ArithT
.. autoclass:: ArithTb
.. autoclass:: ModeT
.. autoprotocol:: SupportsArithmetic
.. autoclass:: ChnName

Entities
--------
.. autoclass:: ArithDict
   :members:
   :member-order: groupwise
   :exclude-members: listify
   :special-members: __mul__, __rmul__, __add__, __sub__, __truediv__, __rtruediv__, __neg__

.. autoclass:: ModeDict
   :members:
   :member-order: groupwise
   :show-inheritance:
   :inherited-members: UserDict

.. autoclass:: ChannelDict
   :members:
   :member-order: groupwise
   :show-inheritance:
   :inherited-members: UserDict
"""

from __future__ import annotations
from collections import UserDict
from collections.abc import Mapping, Sequence
import logging
from typing import TypeVar, Protocol, Self, Generic, Callable, Union, cast
import numbers
import numpy as np
import numpy.typing as npt

from .. import lib


log = logging.getLogger(__name__)

KT = TypeVar("KT")
"""Key type."""

# ArithT = TypeVar(
#     "ArithT", bound=Union["SupportsArithmetic", npt.NDArray, numbers.Number]
# )
# """Arithmetic type."""

ArithT = TypeVar("ArithT")
"""Arithmetic type."""

ArithTb = TypeVar("ArithTb")
"""Arithmetic type (bis)."""

# ArithTb = TypeVar(
#     "ArithTb", bound=Union["SupportsArithmetic", npt.NDArray, numbers.Number]
# )
# """Arithmetic type (bis)."""

ModeT = TypeVar("ModeT", bound=tuple[int, ...])
"""Mode type."""

ChnName = str


class SupportsArithmetic(Protocol):
    """A protocol for supporting arithmetic operations."""

    def __array_ufunc__(self, ufunc: np.ufunc, method: lib.MethodT, *inputs, **kwargs):
        """Support arithmetic operations via numpy ufuncs."""


class ArithDict(UserDict[KT, ArithT], lib.mixins.NDArrayMixin):
    """A dictionary of values that support arithmetic operations."""

    def _check_keys(self, other: ArithDict):
        if set(other.keys()) != set(self.keys()):
            raise ValueError("Cannot operate on two ArithDicts with different keys.")
        return True

    def _unwrap(self, x: object, k: KT):
        try:
            for_type = self._type_copy(x)
            self._check_keys(cast(ArithDict, x))
            return cast(ArithDict, x)[k], for_type
        except TypeError:
            return x, x

    def _type_copy(self, other: object):
        type_self = type(self)
        type_other = type(other)
        if isinstance(other, type_self):
            return other
        if isinstance(self, type_other):
            return self
        for base in type(self).__mro__:
            if issubclass(ArithDict, base):
                continue
            if issubclass(type_other, base):
                return base
        raise TypeError(
            f"Cannot determine common type for {type_self} and {type_other}."
        )

    def __array_ufunc__(self, ufunc: np.ufunc, method: lib.MethodT, *inputs, **kwargs):
        """Support arithmetic operations via numpy ufuncs."""
        if method == "reduce":
            return NotImplemented

        if method == "accumulate":
            return NotImplemented

        if method == "outer":
            return NotImplemented

        if method == "reduceat":
            return NotImplemented

        if method == "at":
            return NotImplemented

        if method == "__call__":
            unwrapped: dict[KT, list[object]] = {}
            for_type = self
            for k in self.keys():
                unwrapped[k] = []
                for inp in inputs:
                    val, val_for_type = self._unwrap(inp, k)
                    unwrapped[k].append(val)
                    try:
                        for_type = for_type._type_copy(val_for_type)
                    except TypeError:
                        # If no common type is found, we assume
                        # val is a number or an array
                        # and we keep common_type as is
                        pass

            # unwrapped = {
            #     k: [self._unwrap(inp, k) for inp in inputs] for k in self.keys()
            # }
            out_arg = kwargs.get("out", None)

            if (
                out_arg is not None
                and len(out_arg) == 1
                and out_arg[0] is self
                and len(inputs) == 2
                and inputs[0] is self
                and ufunc in (np.add, np.subtract)
            ):
                other = inputs[1]
                for k in self.keys():
                    rhs, _ = self._unwrap(other, k)
                    if ufunc is np.add:
                        self[k] = self[k].__iadd__(rhs)  # type: ignore[attr-defined]
                    else:
                        self[k] = self[k].__iadd__(-rhs)  # type: ignore[attr-defined]
                return self

            if out_arg is None:
                new_data = {k: ufunc(*unwrapped[k], **kwargs) for k in self.keys()}
                return for_type.create_new(new_data)

            out_unwrapped = {
                k: [self._unwrap(o, k)[0] for o in out_arg] for k in self.keys()
            }
            for k in self.keys():
                kwargs["out"] = tuple(out_unwrapped[k])
                ufunc(*unwrapped[k], **kwargs)
            return out_arg[0]

    def _type_copy(self, other: object):
        type_self = type(self)
        type_other = type(other)
        if isinstance(other, type_self):
            return other
        if isinstance(self, type_other):
            return self
        for base in type(self).__mro__:
            if issubclass(ArithDict, base):
                continue
            if issubclass(type_other, base):
                return base
        raise TypeError(
            f"Cannot determine common type for {type_self} and {type_other}."
        )

    def __array_ufunc__(self, ufunc: np.ufunc, method: lib.MethodT, *inputs, **kwargs):
        """Support arithmetic operations via numpy ufuncs."""
        if method == "reduce":
            return NotImplemented

        if method == "accumulate":
            return NotImplemented

        if method == "outer":
            return NotImplemented

        if method == "reduceat":
            return NotImplemented

        if method == "at":
            return NotImplemented

        if method == "__call__":
            unwrapped: dict[KT, list[object]] = {}
            for_type = self
            for k in self.keys():
                unwrapped[k] = []
                for inp in inputs:
                    val, val_for_type = self._unwrap(inp, k)
                    unwrapped[k].append(val)
                    try:
                        for_type = for_type._type_copy(val_for_type)
                    except TypeError:
                        # If no common type is found, we assume
                        # val is a number or an array
                        # and we keep common_type as is
                        pass

            # unwrapped = {
            #     k: [self._unwrap(inp, k) for inp in inputs] for k in self.keys()
            # }
            out_arg = kwargs.get("out", None)
            if out_arg is None:
                new_data = {k: ufunc(*unwrapped[k], **kwargs) for k in self.keys()}
                return for_type.create_new(new_data)

            out_unwrapped = {
                k: [self._unwrap(o, k)[0] for o in out_arg] for k in self.keys()
            }
            for k in self.keys():
                kwargs["out"] = tuple(out_unwrapped[k])
                ufunc(*unwrapped[k], **kwargs)
            return out_arg[0]

    def _create_new(self, data: Mapping[KT, ArithTb]):
        """Create a new instance of the class."""
        return ArithDict(data)

    def create_new(self, data: Mapping[KT, ArithT]) -> Self:
        """Create a new instance of the class."""
        return type(self)(data)

    def pass_through(self, func: Callable[[ArithT], ArithTb]):
        """Pass the dictionary through a function.

        Essentially, this is a map operation on the values of the dictionary,
        returning a new dictionary with the same keys, like:

        .. code-block:: python

            self.create_new({k: func(v) for k, v in self.items()})

        In case `v` is itself a :class:`.ArithDict`, the function is applied recursively.
        """

        def _func(v: ArithT) -> ArithTb:
            if isinstance(v, ArithDict):
                return v.pass_through(func)  # type: ignore[return-value]
            return func(v)

        return self._create_new({k: _func(v) for k, v in self.items()})

    def listify(self, keys: KT | Sequence[KT]) -> list[KT]:
        """Convert the keys to a list."""
        del keys
        raise NotImplementedError("This method should be implemented in child classes.")

    def pick(self, keys: KT | Sequence[KT]) -> Self:
        """Return a new instance with only the given keys.

        Parameters
        ----------
        keys :
            The keys to keep. Either a single key or a sequence of keys.
        """
        # pylint: disable=assignment-from-no-return, not-an-iterable
        keys = self.listify(keys)
        return self.create_new({key: self[key] for key in keys})

    def drop(self, keys: KT | Sequence[KT]) -> Self:
        """Return a new instance dropping the given keys.

        Parameters
        ----------
        keys :
            The keys to drop. Either a single key or a sequence of keys.
        """
        # pylint: disable=assignment-from-no-return, unsupported-membership-test
        keys = self.listify(keys)
        return self.create_new(
            {key: self[key] for key in self.keys() if key not in keys}
        )

    def sum(self) -> ArithT:
        """Return the sum of all the values in the dictionary."""
        # TODO optimize the typing here
        return sum(self.values())  # type: ignore[return-value,arg-type]

    @property
    def real(self):
        """Return the real part of the dictionary values."""
        return self.create_new({k: getattr(v, "real") for k, v in self.items()})

    @property
    def imag(self):
        """Return the imaginary part of the dictionary values."""
        return self.create_new({k: getattr(v, "imag") for k, v in self.items()})


class ModeDict(ArithDict[ModeT, ArithT], Generic[ModeT, ArithT]):
    """A dictionary of modes."""

    @property
    def modes(self) -> tuple[ModeT, ...]:
        """Return the modes."""
        return tuple(self.keys())

    def listify(self, keys: ModeT | Sequence[ModeT]) -> list[ModeT]:  # noqa: D102
        # Pay attention as ModeT is a tuple so a Sequence
        if isinstance(keys, tuple) and isinstance(keys[0], int):
            # Then I know it is not a sequence of modes
            keys = cast(ModeT, keys)
            return [keys]
        # Otherwise, it is a sequence of modes
        keys = cast(Sequence[ModeT], keys)
        return list(keys)

    def _create_new(self, data: Mapping[ModeT, ArithTb]):
        """Create a new instance of the class."""
        return ModeDict(data)

class ChannelDict(ArithDict[ChnName, ArithT], Generic[ArithT]):
    """A dictionary of channels."""

    @property
    def _cls_binary_op(self):
        return ChannelDict

    @property
    def channel_names(self) -> tuple[ChnName, ...]:
        """Return the channel names."""
        return tuple(self.keys())

    def listify(self, keys: ChnName | Sequence[ChnName]) -> list[ChnName]:  # noqa: D102
        if isinstance(keys, ChnName):
            return [keys]
        return list(keys)

    def _create_new(self, data: Mapping[ChnName, ArithTb]):
        """Create a new instance of the class."""
        return ChannelDict(data)
