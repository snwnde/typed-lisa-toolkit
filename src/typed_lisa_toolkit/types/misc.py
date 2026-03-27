"""Miscellaneous utility types."""

import logging
from types import ModuleType
from typing import (
    Callable,
    Literal,
    Self,
    cast,
    final,
    overload,
)

import array_api_compat as xpc
import numpy as np
import numpy.typing as npt
from l2d_interface.contract import LinspaceLike

try:
    import jax
    import jax.typing as jpt
except ImportError:
    type ArrayLike = npt.ArrayLike
    type Array = npt.NDArray[np.number]
else:
    type ArrayLike = jpt.ArrayLike | npt.ArrayLike  # pyright: ignore[reportRedeclaration]
    type Array = jax.Array | npt.NDArray[np.number]  # pyright: ignore[reportRedeclaration]
    """An array from any array library supporting the Python Array API standard.

    Currently only NumPy and JAX arrays have been tested, but in principle any array 
    library that implements the Python Array API standard should be compatible.
    """

ArrayFunc = Callable[[Array], Array]
"""A callable that takes an :class:`array <.Array>` as input and returns an :class:`array <.Array>` as output."""

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
        To construct a Linspace, use :func:`~typed_lisa_toolkit.linspace`
        or :func:`~typed_lisa_toolkit.linspace_from_array`.

    .. attention::

        This class is designed to represent a uniform grid by
        three numbers. It does not try to implement the full
        interface of an array, but only a subset of it that is
        relevant for our use cases.
    """

    def __init__(self, start: float, step: float, num: int):
        if num <= 0:
            raise ValueError("num must be at least 1")
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

    def __eq__(self, other: object) -> bool:
        """Check if two Linspace instances are equal."""
        if not isinstance(other, LinspaceLike):
            raise TypeError(f"Cannot compare Linspace with {type(other)}.")
        if not self.start == other.start:
            return False
        if not self.step == other.step:
            return False
        if not len(self) == len(other):
            return False
        return True

    def __len__(self) -> int:
        """Return the length of the array."""
        return self.num

    def __repr__(self):
        """Return the string representation of the array."""
        return f"Linspace(start={self.start}, step={self.step}, num={self.num})"

    def __array__(
        self, dtype: "npt.DTypeLike | None" = None, copy: bool | None = None
    ) -> "npt.NDArray[np.floating]":
        """Return the grid as a numpy array."""
        grid = self.start + self.step * np.arange(self.num, dtype=dtype)
        if copy is False:
            return grid
        return np.array(grid, copy=True)

    def __getitem__(self, slice: object) -> Self:
        """Return a subset of the array."""
        if not isinstance(slice, _slice):
            raise TypeError(f"Invalid index type: expected slice, got {type(slice)}.")
        slice_idx = slice.indices(self.num)
        start = self.start + self.step * slice_idx[0]
        step = self.step * slice_idx[2]
        num = len(range(*slice_idx))
        return type(self)(start=start, step=step, num=num)

    @classmethod
    def from_array(cls, array: "ArrayLike"):
        """Create a Linspace from an array."""
        return linspace_from_array(array)

    @classmethod
    def make(cls, array: "ArrayLike | LinspaceLike") -> "Linspace":
        """Create a Linspace from a numpy array or return the input if already Linspace."""
        if isinstance(array, Linspace):
            return array
        if isinstance(array, LinspaceLike):
            return linspace(start=array.start, step=array.step, num=len(array))
        return linspace_from_array(array)

    @classmethod
    def get_step(cls, grid: "ArrayLike | LinspaceLike") -> float:
        """Return the step of the uniform grid."""
        if isinstance(grid, LinspaceLike):
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


Axis = Array | Linspace
"""An axis of a grid, which can be either an :class:`array <.Array>` or a :class:`Linspace`."""


def linspace(start: float, step: float, num: int) -> Linspace:
    """Create a :class:`~types.Linspace` instance."""
    return Linspace(start=start, step=step, num=num)


def linspace_from_array(array: ArrayLike) -> Linspace:
    """Create a :class:`~types.Linspace` instance from an array."""
    xp = xpc.get_namespace(array)
    _array = xp.asarray(array)
    if len(_array) < 2:
        raise ValueError("Array must have at least two elements to create Linspace.")
    diff = xp.diff(_array)
    if not xp.allclose(diff, diff[0], rtol=1e-8, atol=0):
        raise ValueError("Array must have uniform spacing to create Linspace.")
    return linspace(start=float(_array[0]), step=float(diff[0]), num=len(_array))


class Grid2DSparse[Axis0: Axis, Axis1: Axis]:
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
        return cast(Axis0, self._axis0)

    @property
    def axis1(self) -> Axis1:
        """The second axis of the grid."""
        return cast(Axis1, self._axis1)

    def __init__(self, axis0: Axis, axis1: Axis, *, sparse_indices: Array):
        self._axis0: Axis = axis0
        self._axis1: Axis = axis1
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
        elif idx == 1:
            return self.axis1
        else:
            raise IndexError(f"Invalid index {idx} for Grid2DSparse.")

    def __len__(self):
        """Return the number of axes."""
        return 2

    def __iter__(self):
        """Return an iterator over the axes."""
        yield self.axis0
        yield self.axis1


type Grid1D[AxisT: "Axis"] = tuple[AxisT]
"""A tuple containing a single axis, representing a 1D grid."""

type Grid2DCartesian[Axis0: Axis, Axis1: Axis] = tuple[Axis0, Axis1]
"""A tuple containing two axes, representing a dense 2D grid."""


type Grid2D[Axis0: Axis, Axis1: Axis] = (
    Grid2DCartesian[Axis0, Axis1] | Grid2DSparse[Axis0, Axis1]
)
"""A 2D grid, which can be either :class:`.Grid2DCartesian` or :class:`.Grid2DSparse`."""


UniformGrid2D = Grid2D["Linspace", "Linspace"]
"""A tuple containing two :class:`Linspace` axes, representing a uniform 2D grid."""

AnyGrid = Grid1D["Axis"] | Grid2D["Axis", "Axis"]
"""A grid that can be either 1D or 2D."""

Domain = Literal["time", "frequency", "time-frequency"]
"""A type representing the physical domain of a representation, which can be either "time", "frequency", or "time-frequency"."""


@overload
def build_grid2d[Axis0: Axis, Axis1: Axis](
    axis0: Axis0, axis1: Axis1, /, *, sparse_indices: None = None
) -> Grid2DCartesian[Axis0, Axis1]: ...


@overload
def build_grid2d[Axis0: Axis, Axis1: Axis](
    axis0: Axis0, axis1: Axis1, /, *, sparse_indices: Array
) -> Grid2DSparse[Axis0, Axis1]: ...


def build_grid2d[Axis0: Axis, Axis1: Axis](
    axis0: Axis0, axis1: Axis1, /, *, sparse_indices: Array | None = None
) -> Grid2DCartesian[Axis0, Axis1] | Grid2DSparse[Axis0, Axis1]:
    """Build a :class:`~typed_lisa_toolkit.types.misc.Grid2D`, either :class:`dense <typed_lisa_toolkit.types.misc.Grid2DCartesian>` or :class:`sparse <typed_lisa_toolkit.types.misc.Grid2DSparse>`."""
    if sparse_indices is None:
        return (axis0, axis1)
    else:
        return Grid2DSparse[Axis0, Axis1](axis0, axis1, sparse_indices=sparse_indices)
