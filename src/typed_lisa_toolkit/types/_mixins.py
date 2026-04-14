# ruff: noqa
"""Module for mixins.

.. currentmodule:: typed_lisa_toolkit.lib.mixins

Classes
-------

.. autoclass:: NDArrayMixin
    :members:
"""

import abc
from types import ModuleType
import logging
import operator
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Self, cast, Protocol

import array_api_compat as xpc
import l2d_interface.validators as l2dv
from l2d_interface.contract import LinspaceLike

import numpy as np

from .misc import AnyGrid, Array, Domain, Linspace, Axis
from . import modes
from .. import utils


Mode = modes.Harmonic | modes.QNM


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


class BinaryUnaryOpMixin(NDArrayMixin, abc.ABC):
    entries: "Array"

    @abc.abstractmethod
    def create_like(self, entries: "Array") -> Self: ...

    @abc.abstractmethod
    def _unwrap(self, other: object) -> object: ...

    def _binary_op(
        self,
        other: object,
        op: Callable[[Any, Any], Any],
        /,
        reflected: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ):
        del kwargs  # Unused

        if not reflected:
            entries = op(self.entries, self._unwrap(other))
        else:
            entries = op(self._unwrap(other), self.entries)

        if inplace:
            # del entries # This works for Numpy but fails for JAX
            # The following works for both
            self.entries = entries
            return self

        return self.create_like(entries)

    def _unary_op(self, op: Callable[..., Any], /, **kwargs: Any) -> Self:
        out_arg = kwargs.get("out", None)
        if out_arg is not None:
            kwargs["out"] = self._unwrap(out_arg)
        entries = op(self.entries, **kwargs)
        return self.create_like(entries)


def check_grid_compatibility(grid1: "AnyGrid", grid2: "AnyGrid") -> bool:
    if len(grid1) != len(grid2):
        return False
    for g1, g2 in zip(grid1, grid2):
        if isinstance(g1, Linspace) and isinstance(g2, Linspace):
            if g1 != g2:
                return False
        else:
            xp = xpc.get_namespace(g1)
            if not xp.array_equal(g1, g2):
                return False
    return True


class _GridProperty:
    def __get__[GridT: AnyGrid](
        self,
        instance: "ChannelMapping[reps.Representation[GridT]]",
        owner: Any,
    ) -> GridT: ...


class ChannelMapping[RepT: "AnyReps"](Mapping[str, RepT], BinaryUnaryOpMixin, abc.ABC):
    _REP_TYPE: type[RepT]

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the channel names."""
        return self._channel_names

    def __init__(
        self,
        grid: "AnyGrid|None" = None,
        entries: "Array|None" = None,
        channels: tuple[str, ...] | None = None,
        *,
        name: str | None = None,
        _mapping: Mapping[str, "AnyReps"] | None = None,
        _rep_type: type["AnyReps"] | None = None,
    ):
        if _mapping is None:
            assert grid is not None and entries is not None and channels is not None, (
                "Must provide grid, entries, and channels when not initializing from an existing mapping."
            )
            self._grid: "AnyGrid" = grid
            self.entries: "Array" = entries
            self._channel_names: tuple[str, ...] = tuple(channels)
            try:
                self._rep_type: type[RepT] = self._REP_TYPE
            except AttributeError:
                if _rep_type is None:
                    raise ValueError(
                        "Must provide _rep_type when not initializing from an existing mapping and _REP_TYPE is not defined."
                    )
                self._rep_type = cast(type[RepT], _rep_type)
        else:
            assert grid is None and entries is None and channels is None, (
                "Must not provide grid, entries, or channels when initializing from an existing mapping."
            )
            if len(_mapping) == 0:
                raise ValueError("Cannot initialize from an empty mapping.")
            self._grid = next(iter(_mapping.values())).grid
            self._channel_names = tuple(_mapping.keys())
            xp = xpc.get_namespace(
                *(_mapping[chn].entries for chn in self._channel_names)
            )
            # Concatenate entries along the channel dimension (canonical shape: n_batches, n_channels, ...)
            self.entries = xp.concatenate(
                [_mapping[chn].entries for chn in self._channel_names], axis=1
            )
            self._rep_type = type(
                next(iter(cast(Mapping[str, RepT], _mapping).values()))
            )
        self._channel_to_idx: dict[str, int] = {
            chn: i for i, chn in enumerate(self._channel_names)
        }
        self._refresh_mapping()
        self.set_name(name)

    def _refresh_mapping(self) -> None:
        self._mapping: Mapping[str, RepT] = {
            chn: self.__get_repr_by_channel(chn) for chn in self._channel_names
        }

    def create_like(self, entries: "Array") -> Self:
        return type(self)(
            self.grid,
            entries,
            self.channel_names,
            name=self.name,
            _rep_type=self._rep_type,
        )

    def __xp__(self, api_version: str | None = None) -> Any:
        """Get the array namespace from the _representation entries."""
        return xpc.get_namespace(self.entries, api_version=api_version)

    def _unwrap(self, other: object):
        if hasattr(other, "grid") and hasattr(other, "entries"):
            other = cast(ChannelMapping["AnyReps"], other)
            if check_grid_compatibility(self.grid, other.grid):
                return other.entries
            raise ValueError(f"Grid mismatch: expected {self.grid}, got {other.grid}.")
        return other

    def _binary_op(
        self,
        other: object,
        op: Callable[[Any, Any], Any],
        /,
        reflected: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Self:
        result = super()._binary_op(
            other,
            op,
            reflected=reflected,
            inplace=inplace,
            **kwargs,
        )
        if inplace:
            self._refresh_mapping()
        return result

    def pick(self, channels: str | tuple[str, ...]) -> Self:
        """Return a new instance containing only the specified channels."""
        if isinstance(channels, str):
            channels = (channels,)

        indices = tuple([self._channel_to_idx[chn] for chn in channels])
        # Slice entries to pick only these channels (canonical shape: n_batches, n_channels, ...)
        picked_entries = self.xp.asarray(self.entries)[:, indices, ...]
        return type(self)(self.grid, picked_entries, channels, name=self.name)

    @classmethod
    def from_dict(cls, data_dict: Mapping[str, "AnyReps"], **kwargs: Any) -> Self:
        """Create a new instance from a dictionary of channel names to representations."""
        return cls(_mapping=data_dict, **kwargs)

    def set_name(self, name: str | None) -> Self:
        """Set the name of the data container.

        .. note:: The name is only used for labeling the data container in plots.

        .. note:: This method returns ``self`` to allow for fluent method chaining.
        """
        self.name: str | None = name
        return self

    def __get_repr_by_channel(self, channel: str) -> RepT:
        """Get the representation for a given channel."""
        idx = self._channel_to_idx[channel]
        entries_view = self.entries[:, idx : idx + 1, 0:1, ...]
        return self._rep_type(self._grid, entries_view)

    # Implement Mapping protocol
    def __getitem__(self, key: str) -> RepT:
        """Get a channel by name as a view with preserved channel dimension (size 1)."""
        return self._mapping[key]

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

    @property
    def grid(self):  # pyright: ignore[reportRedeclaration]
        """Return the grid."""
        return self._grid

    grid: _GridProperty  # For correct type hinting

    @property
    def domain(self) -> Domain:
        """Physical domain shared by all channels."""
        return self[self.channel_names[0]].domain

    @property
    @abc.abstractmethod
    def kind(self) -> str | None:
        """Semantic kind."""

    def get_kernel(self) -> Array:
        """Return kernel entries in conventional shape."""
        return self.entries


def validate_maps_to_reps(mapping: Mapping[Any, "AnyReps"]):
    """Validate that a mapping maps to valid representations."""
    for key, rep in mapping.items():
        try:
            l2dv.validate_representation(rep)
        except ValueError as error:
            raise ValueError(f"Invalid representation for key {key!r}.") from error


def to_array(ary: "Axis", xp: ModuleType = np) -> "Array":
    """Convert an axis to an array if it is a Linspace, otherwise return it as is."""
    if isinstance(ary, LinspaceLike):
        return Linspace.make(ary).asarray(xp)
    return ary


def embed_entries_to_grid[GT: "AnyGrid"](
    source_grid: "AnyGrid",
    source_entries: "Array",
    embedding_grid: GT,
    *,
    known_slices: tuple[slice, ...] | None = None,
) -> tuple[GT, "Array"]:
    """Embed entries from source grid into a target grid."""
    _embedding_grid = tuple(to_array(eg) for eg in embedding_grid)
    _source_grid = tuple(to_array(sg) for sg in source_grid)
    entries = utils.extend_to(_embedding_grid, known_slices=known_slices)(
        _source_grid, source_entries
    )
    return embedding_grid, entries


class HasDomain(Protocol):
    @property
    def domain(self) -> Domain: ...


class HasXP(Protocol):
    def __xp__(self, api_version: str | None = None) -> ModuleType: ...


class _HasXPAndDomain(HasXP, HasDomain, Protocol): ...


class ModeMapping[ModeT: Mode, VT: _HasXPAndDomain](Mapping[ModeT, VT], NDArrayMixin):
    def __init__(self, mapping: Mapping[ModeT, Any]):
        self._mapping: Mapping[ModeT, Any] = mapping

    # Implement Mapping protocol
    def __getitem__(self, key: ModeT) -> VT:
        """Get a channel by name as a view with preserved channel dimension (size 1)."""
        return self._mapping[key]

    def __iter__(self) -> Iterator[ModeT]:
        """Iterate over harmonic modes."""
        return iter(self._mapping)

    def __len__(self) -> int:
        """Return the number of harmonic modes."""
        return len(self._mapping)

    def __repr__(self):
        items = {key: self[key] for key in self}
        return f"{self.__class__.__name__}({items!r})"

    def pick(self, modes: ModeT | tuple[ModeT, ...]) -> Self:
        """Return a new instance containing only the specified modes."""
        try:
            return self._pick(modes)  # type: ignore[arg-type]
        except KeyError:
            return self._pick((modes,))  # type: ignore[arg-type]

    def _pick(self, modes: tuple[ModeT, ...]) -> Self:
        new_mapping = {mode: self[mode] for mode in modes}
        return type(self)(new_mapping)

    @property
    def harmonics(self):
        """All harmonic modes and their order."""
        return tuple(self._mapping.keys())

    @property
    def domain(self):
        """Physical domain shared by all harmonics."""
        return self[next(iter(self))].domain

    def __xp__(self, api_version: str | None = None) -> ModuleType:
        """Array namespace from the first harmonic."""
        return self[next(iter(self))].__xp__(api_version=api_version)

    def _binary_op(
        self,
        other: object,
        op: Callable[[Any, Any], Any],
        /,
        reflected: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ):
        del kwargs  # Unused

        if isinstance(other, ModeMapping):
            _other_harmonics = cast(ModeMapping[ModeT, Any], other).harmonics
            if set(self.harmonics) != set(_other_harmonics):
                raise ValueError(
                    f"Harmonic mode mismatch: expected {self.harmonics}, got {_other_harmonics}."
                )
            if not reflected:
                _mapping = {
                    mode: op(self[mode], other[mode]) for mode in self.harmonics
                }
            else:
                _mapping = {
                    mode: op(other[mode], self[mode]) for mode in self.harmonics
                }
        else:
            if not reflected:
                _mapping = {mode: op(self[mode], other) for mode in self.harmonics}
            else:
                _mapping = {mode: op(other, self[mode]) for mode in self.harmonics}

        if inplace:
            self.__init__(_mapping)
            return self

        return type(self)(_mapping)

    def _unary_op(self, op: Callable[..., Any], /, **kwargs: Any) -> Self:
        _mapping = {mode: op(rep, **kwargs) for mode, rep in self._mapping.items()}
        return type(self)(_mapping)
