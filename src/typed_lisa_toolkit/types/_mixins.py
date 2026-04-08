# ruff: noqa
"""Module for mixins.

.. currentmodule:: typed_lisa_toolkit.lib.mixins

Classes
-------

.. autoclass:: NDArrayMixin
    :members:
"""

import abc
import logging
import operator
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Self, cast, overload

import array_api_compat as xpc

from .misc import AnyGrid, Array


if TYPE_CHECKING:
    from . import representations as reps

    AnyReps = reps.Representation[AnyGrid]


log = logging.getLogger(__name__)


__all__ = ["NDArrayMixin"]


class NDArrayMixin(abc.ABC):
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


class ChannelMapping[RepT: "AnyReps"](Mapping[str, RepT], NDArrayMixin):
    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the channel names."""
        return self._channel_names

    def __init__(
        self,
        representation: "AnyReps",
        channels: tuple[str, ...],
        name: str | None = None,
    ):
        self._channel_names: tuple[str, ...] = tuple(channels)
        self._init_repr(representation)
        entries = self.representation.entries
        if entries.shape[1] != len(self.channel_names):
            raise ValueError(
                "Channel count mismatch between representation entries and channel names."
            )
        if entries.shape[2] != 1:
            raise ValueError(
                "Data containers require n_harmonics=1 in the representation entries."
            )
        self._channel_to_idx: dict[str, int] = {
            chn: i for i, chn in enumerate(channels)
        }
        self.name: str | None = name

    @property
    def representation(self) -> RepT:
        """Return the underlying representation."""
        return cast(RepT, self._representation)

    def _init_repr(self, representation: "AnyReps"):
        self._representation: AnyReps = representation

    def create_new(self, representation: "AnyReps", channels: tuple[str, ...]) -> Self:
        """Create a new instance with a representation and channels."""
        return type(self)(representation, channels, self.name)

    def create_like(self, entries: "Array", channels: tuple[str, ...]) -> Self:
        """Create a new instance with different entries but the same grid and type."""
        return type(self)(self.representation.create_like(entries), channels, self.name)

    def __xp__(self, api_version: str | None = None) -> Any:
        """Get the array namespace from the representation entries."""
        return xpc.get_namespace(self.representation.entries, api_version=api_version)

    def _binary_op(
        self,
        other: object,
        op: Callable[[object, object], object],
        /,
        reflected: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Apply binary operation using native array ops on representations."""
        del kwargs  # Unused

        if isinstance(other, ChannelMapping):
            if self.channel_names != other.channel_names:
                raise ValueError("Cannot operate on data with different channel sets.")
            if reflected:
                new_repr = op(other.representation, self.representation)
            else:
                new_repr = op(self.representation, other.representation)
        else:
            # Scalar or array-like broadcast
            if reflected:
                new_repr = op(other, self.representation)
            else:
                new_repr = op(self.representation, other)

        if inplace:
            self._representation = cast("AnyReps", new_repr)
            return self

        return self.create_new(cast(RepT, new_repr), self.channel_names)

    def _unary_op(self, op: Callable[[object], object], /, **kwargs: Any) -> Self:
        """Apply unary operation using native array ops."""
        new_repr = op(self.representation, **kwargs)
        return self.create_new(cast(RepT, new_repr), self.channel_names)

    def pick(self, channels: str | tuple[str, ...]) -> Self:
        """Pick specific channels.

        Parameters
        ----------
        channels : str or tuple[str, ...]
            Channel name(s) to pick.

        Returns
        -------
        Self
            New instance with only the specified channels.
        """
        if isinstance(channels, str):
            channels = (channels,)

        indices = tuple([self._channel_to_idx[chn] for chn in channels])
        # Slice entries to pick only these channels (canonical shape: n_batches, n_channels, ...)
        xp = xpc.get_namespace(self.representation.entries)
        picked_entries = xp.asarray(self.representation.entries)[:, indices, ...]
        picked_repr = self.representation.create_like(picked_entries)
        return self.create_new(picked_repr, channels)

    @classmethod
    def from_dict[RT: "AnyReps"](
        cls, data_dict: Mapping[str, RT], **additions: Any
    ) -> Self:
        """Create a new instance from a dictionary of channel names to representations."""
        if len(data_dict) == 0:
            raise ValueError("Cannot build data container from an empty mapping.")
        channels = tuple(data_dict.keys())
        xp = xpc.get_namespace(*(cast(Any, data_dict[chn]).entries for chn in channels))
        # Concatenate entries along the channel dimension (canonical shape: n_batches, n_channels, ...)
        entries = xp.concatenate([data_dict[chn].entries for chn in channels], axis=1)
        # Assume all representations have the same grid and type
        first = next(iter(data_dict.values()))
        return cls(first.create_like(entries), channels, **additions)

    def set_name(self, name: str | None) -> Self:
        """Set the name of the data container.

        .. note:: The name is only used for labeling the data container in plots.

        .. note:: This method returns `self` to allow for fluent method chaining.
        """
        self.name = name
        return self

    # Implement Mapping protocol
    def __getitem__(self, key: str) -> RepT:
        """Get a channel by name as a view with preserved channel dimension (size 1)."""
        idx = self._channel_to_idx[key]
        entries_view = self.representation.entries[:, idx : idx + 1, 0:1, ...]
        return self.representation.create_like(entries_view)

    def __iter__(self) -> Iterator[str]:
        """Iterate over channel names."""
        return iter(self.channel_names)

    def __len__(self) -> int:
        """Return the number of channels."""
        return len(self.channel_names)

    def __repr__(self):
        items = {key: self[key] for key in self}
        if self.name is not None:
            return f"{self.__class__.__name__}(name={self.name!r}, items={items!r})"
        return f"{self.__class__.__name__}({items!r})"

    @overload
    def grid[GridT: AnyGrid](  # pyright: ignore[reportInconsistentOverload]
        self: "ChannelMapping[reps.Representation[GridT]]",
    ) -> GridT: ...

    @property
    def grid(self):
        """Return the grid."""
        return self.representation.grid

    @property
    def domain(self):
        """Physical domain shared by all channels."""
        return self.representation.domain

    @property
    def kind(self):
        """Semantic kind shared by all channels."""
        return self.representation.kind

    @property
    def entries(self) -> Any:
        """Return the raw entries array for direct matrix operations.

        Shape is (n_channels, ...) where ... depends on the representation type.
        Use this for complicated array operations with numpy/jax.
        """
        return self.representation.entries

    def get_kernel(self):
        """Return kernel entries in conventional shape."""
        return self.representation.entries
