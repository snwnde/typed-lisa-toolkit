"""Module for arithmetic dictionaries."""

from __future__ import annotations
from collections import UserDict
from collections.abc import Mapping, Sequence
import logging
from typing import TypeVar, Protocol, Self, Generic, Callable, Union, cast, Type


log = logging.getLogger(__name__)

KT = TypeVar("KT")  # Key type
ArithT = TypeVar("ArithT", bound=Union["SupportsArithmetic", "ArithDict"])
ArithTb = TypeVar("ArithTb", bound=Union["SupportsArithmetic", "ArithDict"])
ChnName = str  # Channel name
ModeT = TypeVar("ModeT", bound=tuple[int, ...])  # Mode type


class SupportsArithmetic(Protocol):
    """A protocol for supporting arithmetic operations."""

    def __add__(self: ArithT, other: ArithT, /) -> SupportsArithmetic: ...  # noqa: D105
    def __sub__(self: ArithT, other: ArithT, /) -> ArithT: ...  # noqa: D105
    def __mul__(self: ArithT, other: ArithT, /) -> ArithT: ...  # noqa: D105
    def __truediv__(self: ArithT, other: ArithT, /) -> SupportsArithmetic: ...  # noqa: D105
    def __neg__(self: ArithT) -> ArithT: ...  # noqa: D105


class ArithDict(UserDict[KT, ArithT]):
    """A dictionary of values that support arithmetic operations."""

    @classmethod
    def _get_null_value(cls) -> ArithT:
        """Get the null value for the class."""

        class NullValue(cls):  # type: ignore
            # `cls` is a class, so it should be used as a type.
            # Ignoring the error that I got from mypy.

            def __add__(self, other: ArithTb):
                return other

        return NullValue()

    def __array__(self):  # noqa: D105
        raise TypeError("""If you see this in a multiplication between a numpy array and an ArithDict,
            most likely the numpy array is on the left side of the multiplication. Try to put it on the right side.""")

    def create_new(self, data: Mapping[KT, ArithTb]):
        """Create a new instance of the class."""
        return type(self)(data)

    def __mul_arithdict__(self, other: ArithDict[KT, ArithTb]):
        """Multiply two ArithDicts."""
        return self.create_new({k: self[k] * other[k] for k in self.keys()})

    def __mul_value__(self, value: ArithTb):
        """Multiply a ArithDict by a value."""
        return self.create_new({k: self[k] * value for k in self.keys()})

    def __mul__(self, other: ArithTb):  # noqa: D105
        if isinstance(other, ArithDict):
            try:
                return self.__mul_arithdict__(other)
            except KeyError as ke:
                log.debug(
                    f"Trying to multiply {self} by {other} as two ArithDict. Got KeyError: {ke}"
                )
        try:
            return self.__mul_value__(other)
        except TypeError as e:
            raise TypeError(f"Cannot multiply {type(self)} by {type(other)}.") from e

    def __rmul__(self, value: ArithTb):
        """Multiply a ArithDict by a value."""
        return self.__mul_value__(value)

    def __truediv__(self, other: ArithTb):
        """Divide a ArithDict by another ArithDict or a value."""
        try:
            return self.create_new({k: self[k] / other for k in self.keys()})
        except TypeError:
            if isinstance(other, ArithDict):
                return self.create_new({k: self[k] / other[k] for k in self.keys()})
        raise TypeError(f"Cannot divide {type(self)} by {type(other)}")

    def __rtruediv__(self, value: ArithTb):
        """Divide a ArithDict by a value."""
        return self.create_new({k: value / self[k] for k in self.keys()})

    def __add_arithdict__(self, other: ArithDict[KT, ArithTb]):
        """Add two ArithDicts."""
        return self.create_new({k: self[k] + other[k] for k in self.keys()})

    def __add_value__(self, value: ArithTb):
        """Add a ArithDict by a value."""
        return self.create_new({k: self[k] + value for k in self.keys()})

    def __add__(self, other: ArithTb):  # noqa: D105
        if isinstance(other, ArithDict):
            try:
                return self.__add_arithdict__(other)
            except KeyError:
                pass
        try:
            return self.__add_value__(other)
        except TypeError as e:
            raise TypeError(f"Cannot add {type(self)} by {type(other)}.") from e

    def __neg__(self):
        """Negate a ArithDict."""
        return self.create_new({k: -self[k] for k in self.keys()})

    def __sub__(self, other: ArithTb):
        """Subtract two ArithDicts."""
        return self + (-other)

    def pass_through(self, func: Callable[[ArithT], ArithTb]):
        """Pass the dictionary through a function."""

        def _func(v: ArithT) -> ArithTb:
            if isinstance(v, ArithDict):
                return v.pass_through(func)
            return func(v)

        return self.create_new({k: _func(v) for k, v in self.items()})

    def listify(self, keys: KT | Sequence[KT]) -> list[KT]:
        """Convert the keys to a list."""
        del keys
        raise NotImplementedError("This method should be implemented in child classes.")

    def pick(self, keys: KT | Sequence[KT]) -> Self:
        """Pick only the given keys.

        Parameters
        ----------
        keys : KT | Sequence[KT]
            The keys to keep.

        Returns
        -------
        Self
            The new dictionary with only the given keys.
        """
        keys = self.listify(keys)
        return self.create_new({key: self[key] for key in keys})

    def drop(self, keys: KT | Sequence[KT]) -> Self:
        """Drop the given keys.

        Parameters
        ----------
        keys : KT | Sequence[KT]
            The keys to drop.

        Returns
        -------
        Self
            The new dictionary without the given keys.
        """
        keys = self.listify(keys)
        return self.create_new(
            {key: self[key] for key in self.keys() if key not in keys}
        )


class ModeDict(ArithDict[ModeT, ArithT], Generic[ModeT, ArithT]):
    """A dictionary of modes."""

    @property
    def modes(self):
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

    def sum(self):
        """Sum all the modes."""
        _sum = sum(self.values(), self._get_null_value())
        return _sum


class ChannelDict(ArithDict[ChnName, ArithT], Generic[ArithT]):
    """A dictionary of channels."""

    @property
    def channel_names(self):
        """Return the channel names."""
        return tuple(self.keys())

    def listify(self, keys: ChnName | Sequence[ChnName]) -> list[ChnName]:  # noqa: D102
        if isinstance(keys, ChnName):
            return [keys]
        return list(keys)
