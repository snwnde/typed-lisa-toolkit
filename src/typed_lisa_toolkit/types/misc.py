"""Miscellaneous utility types."""

import logging
from types import ModuleType
from typing import (
    Callable,
    Literal,
    Self,
    final,
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
    def from_array(cls, array: "ArrayLike") -> Self:
        """Create a Linspace from an array."""
        xp = xpc.get_namespace(array)
        _array = xp.asarray(array)
        if len(_array) < 2:
            raise ValueError(
                "Array must have at least two elements to create Linspace."
            )
        diff = xp.diff(_array)
        if not xp.allclose(diff, diff[0], rtol=1e-8, atol=0):
            raise ValueError("Array must have uniform spacing to create Linspace.")
        return cls(start=float(_array[0]), step=float(diff[0]), num=len(_array))

    @classmethod
    def make(cls, array: "ArrayLike | LinspaceLike") -> "Linspace":
        """Create a Linspace from a numpy array or return the input if already Linspace."""
        if isinstance(array, Linspace):
            return array
        if isinstance(array, LinspaceLike):
            return Linspace(start=array.start, step=array.step, num=len(array))
        return cls.from_array(array)

    @classmethod
    def get_step(cls, grid: "ArrayLike | LinspaceLike") -> float:
        """Return the step of the uniform grid."""
        if isinstance(grid, LinspaceLike):
            return grid.step
        return cls.from_array(grid).step

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

type Grid1D[AxisT: "Axis"] = tuple[AxisT]
"""A 1D grid, represented as a tuple containing a single axis."""

type Grid2D[Axis0: "Axis", Axis1: "Axis"] = tuple[Axis0, Axis1]
"""A 2D grid, represented as a tuple containing two axes."""

UniformGrid2D = Grid2D["Linspace", "Linspace"]
"""A 2D grid where both axes are uniformly spaced, represented as a tuple of two :class:`Linspace` instances."""

AnyGrid = Grid1D["Axis"] | Grid2D["Axis", "Axis"]
"""A grid that can be either 1D or 2D, represented as a tuple of one or two axes."""

Domain = Literal["time", "frequency", "time-frequency"]
"""A type representing the physical domain of a representation, which can be either "time", "frequency", or "time-frequency"."""


def linspace(start: float, step: float, num: int) -> Linspace:
    """Create a :class:`~types.misc.Linspace` instance."""
    return Linspace(start=start, step=step, num=num)
