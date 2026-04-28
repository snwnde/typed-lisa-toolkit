"""Miscellaneous utility types."""

import copy
import logging
from collections.abc import Callable
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
    cast,
    final,
    overload,
)

import array_api_compat as xpc
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import jax
    import jax.typing as jpt

    type JaxArrayLike = jpt.ArrayLike
    type JaxArray = jax.Array
else:
    type JaxArrayLike = npt.ArrayLike
    type JaxArray = npt.NDArray[np.number]

type ArrayLike = JaxArrayLike | npt.ArrayLike
type Array = JaxArray | npt.NDArray[np.number]
"""An array from any array library supporting the Python Array API standard.

Currently only NumPy and JAX arrays have been tested, but in principle any array
library that implements the Python Array API standard should be compatible.
"""


ArrayFunc = Callable[[Array], Array]
"""A callable that takes an :class:`array <.Array>` as input and returns an :class:`array <.Array>` as output."""  # noqa: E501

Interpolator = Callable[[Array, Array], ArrayFunc]
"""A callable providing interpolation functionality.


The callable takes two :class:`arrays <.Array>` as input,
representing the x and y coordinates of the known data points,
and returns a function that can be used to interpolate values
at new x coordinates.
"""


log = logging.getLogger(__name__)
_slice = slice  # Alias for slice


@final
class Linspace:
    """A lazy representation of a uniformly spaced array.

    .. note::
        To construct a Linspace, use :func:`~typed_lisa_toolkit.linspace`,
        :func:`~typed_lisa_toolkit.linspace_from_step`,
        or :func:`~typed_lisa_toolkit.linspace_from_array`.

    .. attention::

        This class is designed to represent a uniform grid by
        three numbers. It does not try to implement the full
        interface of an array, but only a subset of it that is
        relevant for our use cases.
    """

    def __init__(self, start: float, step: float, num: int):
        if num <= 0:
            msg = "num must be at least 1"
            raise ValueError(msg)
        num = int(num)
        # The float conversion is necessary to avoid issues with JAX scalars
        self._start = float(start)
        self._step = float(step)
        self._num = num
        self._shape = (num,)
        self._stop = self.start + self.step * (num - 1)

    @property
    def start(self) -> float:
        """The first point of the array."""
        return self._start

    @property
    def step(self) -> float:
        """The step of the array."""
        return self._step

    @property
    def num(self) -> int:
        """The number of points in the grid."""
        return self._num

    @property
    def shape(self) -> tuple[int]:
        """The shape of the array."""
        return self._shape

    @property
    def stop(self) -> float:
        """The last point of the array."""
        return self._stop

    def __hash__(self) -> int:
        """Return the hash."""
        return hash((self.start, self.step, self.num))

    def __eq__(self, other: object) -> bool:
        """Check if two Linspace instances are equal."""
        if not isinstance(other, Linspace):
            msg = f"Cannot compare Linspace with {type(other)}."
            raise TypeError(msg)
        if not self.start == other.start:
            return False
        if not self.step == other.step:
            return False
        if not len(self) == len(other):  # noqa: SIM103
            return False
        return True

    def __len__(self) -> int:
        """Return the length of the array."""
        return self.num

    def __repr__(self):
        """Return the string representation of the array."""
        return f"Linspace(start={self.start}, step={self.step}, num={self.num})"

    def __array__(
        self,
        dtype: "npt.DTypeLike | None" = None,
        *,
        copy: bool | None = None,
    ) -> "npt.NDArray[np.floating]":
        """Return the grid as a numpy array."""
        grid = self.start + self.step * np.arange(self.num, dtype=dtype)
        if copy is False:
            return grid
        return np.array(grid, copy=True)

    @overload
    def __getitem__(self, slice: _slice, /) -> Self: ...

    @overload
    def __getitem__(self, idx: int, /) -> float: ...

    def __getitem__(self, sli: Any, /):
        """Return a subset of the array."""
        if isinstance(sli, _slice):
            slice_idx = sli.indices(self.num)
            start = self.start + self.step * slice_idx[0]
            step = self.step * slice_idx[2]
            num = len(range(*slice_idx))
            return type(self)(start=start, step=step, num=num)
        if isinstance(sli, int):
            if sli < 0:
                sli += self.num
            if sli < 0 or sli >= self.num:
                msg = f"Index {sli} out of bounds for Linspace of length {self.num}."
                raise IndexError(msg)
            return self.start + self.step * sli
        msg = f"Invalid index {sli} for Linspace.Must be an integer or a slice."
        raise TypeError(msg)

    @classmethod
    def from_array(cls, array: "ArrayLike"):
        """Create a Linspace from an array."""
        return linspace_from_array(array)

    @classmethod
    def make(cls, array: "ArrayLike | Linspace") -> "Linspace":
        """Create a Linspace from a numpy array or return the input if already Linspace."""  # noqa: E501
        if isinstance(array, Linspace):
            return array
        return linspace_from_array(array)

    @classmethod
    def get_step(cls, grid: "ArrayLike | Linspace") -> float:
        """Return the step of the uniform grid."""
        if isinstance(grid, Linspace):
            return grid.step
        return linspace_from_array(grid).step

    def asarray(
        self,
        xp: ModuleType,
        *,
        dtype: "npt.DTypeLike | jax.typing.DTypeLike | None" = None,
    ) -> "Array":
        """Return the linspace as an array in the specified array library."""
        return xp.asarray(self, dtype=dtype)


@final
class Axis[T: Array | Linspace]:
    """An axis of a grid."""

    def __init__(self, ax: ArrayLike | Linspace, *, xp: ModuleType):
        if isinstance(ax, Linspace):
            self._ax = ax
        else:
            self._ax = xp.asarray(ax)
        self.xp: ModuleType = xp

    def __repr__(self) -> str:
        """Return the string representation of the axis."""
        return f"Axis({self.ax!r})"

    def __hash__(self) -> int:
        """Return the hash of the axis."""
        if isinstance(self.ax, Linspace):
            return hash(self.ax)
        return hash(self.ax.tobytes())

    def __eq__(self, other: object) -> bool:
        """Check if two Axis instances are equal."""
        if isinstance(other, Axis):
            return hash(self) == hash(other)
        return False

    def __array__(
        self,
        dtype: "npt.DTypeLike | jax.typing.DTypeLike | None" = None,
        *,
        copy: bool | None = None,
    ) -> "Array":
        """Return the axis as an array."""
        if isinstance(self.ax, Linspace):
            return self.ax.__array__(dtype=dtype, copy=copy)
        return np.asarray(self.ax, dtype=dtype, copy=copy)

    def __deepcopy__(self, memo: dict[int, Any]):
        """Return a deepcopy of the axis without copying the array namespace."""
        copied = type(self)(copy.deepcopy(self.ax, memo), xp=self.xp)
        memo[id(self)] = copied
        return copied

    @property
    def ax(self) -> T:
        """The axis as an array or Linspace."""
        return cast("T", self._ax)

    def __len__(self) -> int:
        """Return the length of the axis."""
        return len(self.ax)

    @overload
    def __getitem__(self, slice: _slice, /) -> Self: ...

    @overload
    def __getitem__(self, idx: int, /) -> float: ...

    def __getitem__(self, sli: Any, /):
        """Return a subset of the array."""
        if isinstance(sli, _slice):
            return type(self)(self.ax[sli], xp=self.xp)
        if isinstance(sli, int):
            return float(self.ax[sli])
        msg = f"Invalid index {sli} for AnyAxis. Must be an integer or a slice."
        raise TypeError(msg)

    @property
    def start(self) -> float:
        """The first point of the axis."""
        return float(self.ax[0])

    @property
    def stop(self) -> float:
        """The last point of the axis."""
        return float(self.ax[-1])

    def asarray(
        self,
        xp: ModuleType | None = None,
        *,
        dtype: "npt.DTypeLike | jax.typing.DTypeLike | None" = None,
    ) -> "Array":
        """Return the axis as an array in the specified array library."""
        xp = self.xp if xp is None else xp
        return xp.asarray(self.ax, dtype=dtype)


AxLike = Array | Linspace
AnyAxis = Axis[AxLike]

# AnyAxis = _ArrayAxis | Linspace
# """An axis of a grid, which can be either an :class:`array <.Array>` or a :class:`Linspace`."""  # noqa: E501


def linspace_from_step(start: float, step: float, num: int) -> Linspace:
    """Create a :class:`~types.Linspace` instance from a start, step, and num."""
    return Linspace(start=start, step=step, num=num)


def linspace(start: float, stop: float, num: int) -> Linspace:
    """Create a :class:`~types.Linspace` instance."""
    step = (stop - start) / (num - 1) if num > 1 else 0
    return linspace_from_step(start=start, step=step, num=num)


def linspace_from_array(array: ArrayLike) -> Linspace:
    """Create a :class:`~types.Linspace` instance from an array."""
    min_size = 2
    xp = xpc.get_namespace(array)
    _array = xp.asarray(array)
    if len(_array) < min_size:
        msg = f"Array must have at least {min_size} elements to create Linspace."
        raise ValueError(msg)
    diff = xp.diff(_array)
    if not xp.allclose(diff, diff[0], rtol=1e-8, atol=0):
        msg = "Array must have uniform spacing to create Linspace."
        raise ValueError(msg)
    return linspace(start=float(_array[0]), stop=float(_array[-1]), num=len(_array))


def axis[AT: Array | Linspace](ax: AT, /) -> Axis[AT]:
    """Create an :class:`~types.Axis` instance."""
    if isinstance(ax, Linspace):
        return Axis[AT](ax, xp=np)
    xp = xpc.get_namespace(ax)
    return Axis[AT](ax, xp=xp)


class Grid2DSparse[Axis0: AnyAxis, Axis1: AnyAxis]:
    """Class for a sparse 2D grid."""

    indices: Array
    """
    The indices of the non-empty points in the grid, represented as an array of shape
    ``(n_sparse_points, 2)`` where each row is a pair of indices corresponding to the
    positions in ``axis0`` and ``axis1``.
    """

    @property
    def axis0(self) -> Axis0:
        """The first axis of the grid."""
        return cast("Axis0", self._axis0)

    @property
    def axis1(self) -> Axis1:
        """The second axis of the grid."""
        return cast("Axis1", self._axis1)

    def __init__(self, axis0: AnyAxis, axis1: AnyAxis, *, sparse_indices: Array):
        self._axis0: AnyAxis = axis0
        self._axis1: AnyAxis = axis1
        self.indices = sparse_indices

    @overload
    def __getitem__(self, idx: Literal[0]) -> Axis0: ...

    @overload
    def __getitem__(self, idx: Literal[1]) -> Axis1: ...

    @overload
    def __getitem__(self, idx: int) -> Axis0 | Axis1: ...

    def __getitem__(self, idx: int):
        """Return the axis at the given index."""
        if idx == 0:
            return self.axis0
        if idx == 1:
            return self.axis1
        msg = f"Invalid index {idx} for Grid2DSparse."
        raise IndexError(msg)

    def __len__(self):
        """Return the number of axes."""
        return 2

    def __iter__(self):
        """Return an iterator over the axes."""
        yield self.axis0
        yield self.axis1

    def __repr__(self):
        """Return the string representation of the grid."""
        return f"Grid2DSparse({self.axis0!r}, {self.axis1!r}, indices={self.indices!r})"

    def __hash__(self) -> int:
        """Return the hash of the grid."""
        _idx = self.indices.tobytes()
        return hash((self.axis0, self.axis1, _idx))

    def __eq__(self, other: object) -> bool:
        """Check if two Grid2DSparse instances are equal."""
        return hash(self) == hash(other)


type Grid1D[AxisT: "AnyAxis"] = tuple[AxisT]
"""A tuple containing a single axis, representing a 1D grid."""

type Grid2DCartesian[Axis0: AnyAxis, Axis1: AnyAxis] = tuple[Axis0, Axis1]
"""A tuple containing two axes, representing a dense 2D grid."""


type Grid2D[Axis0: AnyAxis, Axis1: AnyAxis] = (
    Grid2DCartesian[Axis0, Axis1] | Grid2DSparse[Axis0, Axis1]
)
"""A 2D grid, which is either :class:`.Grid2DCartesian` or :class:`.Grid2DSparse`."""


UniformGrid2D = Grid2D[Axis[Linspace], Axis[Linspace]]
"""A tuple containing two :class:`Linspace` axes, representing a uniform 2D grid."""

AnyGrid = Grid1D["AnyAxis"] | Grid2D["AnyAxis", "AnyAxis"]
"""A grid that can be either 1D or 2D."""

Domain = Literal["time", "frequency", "time-frequency"]
"""A type representing the physical domain of a representation, which can be either "time", "frequency", or "time-frequency"."""  # noqa: E501


@overload
def build_grid2d[Axis0: AnyAxis, Axis1: AnyAxis](
    axis0: Axis0,
    axis1: Axis1,
    /,
    *,
    sparse_indices: None = None,
) -> Grid2DCartesian[Axis0, Axis1]: ...


@overload
def build_grid2d[Axis0: AnyAxis, Axis1: AnyAxis](
    axis0: Axis0,
    axis1: Axis1,
    /,
    *,
    sparse_indices: Array,
) -> Grid2DSparse[Axis0, Axis1]: ...


@overload
def build_grid2d[A0: AxLike, A1: AxLike](
    axis0: A0,
    axis1: A1,
    /,
    *,
    sparse_indices: None = None,
) -> Grid2DCartesian[Axis[A0], Axis[A1]]: ...


@overload
def build_grid2d[A0: AxLike, Axis1: AnyAxis](
    axis0: A0,
    axis1: Axis1,
    /,
    *,
    sparse_indices: None = None,
) -> Grid2DCartesian[Axis[A0], Axis1]: ...


@overload
def build_grid2d[A0: AxLike, A1: AxLike](
    axis0: A0,
    axis1: A1,
    /,
    *,
    sparse_indices: Array,
) -> Grid2DSparse[Axis[A0], Axis[A1]]: ...


@overload
def build_grid2d[Axis0: AnyAxis, A1: AxLike](
    axis0: Axis0,
    axis1: A1,
    /,
    *,
    sparse_indices: Array,
) -> Grid2DSparse[Axis0, Axis[A1]]: ...


def build_grid2d(
    axis0: AxLike | AnyAxis,
    axis1: AxLike | AnyAxis,
    /,
    *,
    sparse_indices: Array | None = None,
):
    """Build a :class:`~typed_lisa_toolkit.types.misc.Grid2D`, either :class:`dense <typed_lisa_toolkit.types.misc.Grid2DCartesian>` or :class:`sparse <typed_lisa_toolkit.types.misc.Grid2DSparse>`."""  # noqa: E501
    _axis0 = axis(axis0) if not isinstance(axis0, Axis) else axis0
    _axis1 = axis(axis1) if not isinstance(axis1, Axis) else axis1
    if sparse_indices is None:
        return (_axis0, _axis1)
    return Grid2DSparse[Any, Any](_axis0, _axis1, sparse_indices=sparse_indices)
