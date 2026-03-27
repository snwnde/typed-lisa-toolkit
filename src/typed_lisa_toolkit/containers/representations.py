"""Module for representations.

**Multi-Backend Support:**

The underlying arrays can be from any array library that supports the Python "Array" API standard,
including NumPy and JAX.

.. currentmodule:: typed_lisa_toolkit.containers.representations

.. autoclass:: Linspace
   :members:
   :special-members: __eq__, __len__, __array__, __getitem__

.. autofunction:: frequency_series
.. autofunction:: time_series
.. autofunction:: phasor
.. autofunction:: stft
.. autofunction:: wdm


.. autoclass:: TimeSeries
   :members:
   :member-order: bysource

.. autoclass:: FrequencySeries
   :members:
   :member-order: bysource
   :inherited-members:

.. autoclass:: Phasor
   :members:
   :member-order: bysource
   :inherited-members:

.. autoclass:: STFT
    :members:
    :member-order: bysource
    :inherited-members:

.. autoclass:: WDM
    :members:
    :member-order: bysource
    :inherited-members:
"""

from __future__ import annotations

import abc
import logging
import warnings
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Protocol,
    Self,
    Union,
    cast,
    final,
    overload,
)

import array_api_compat as xpc
import numpy as np
import scipy.signal
from l2d_interface.contract import LinspaceLike

if TYPE_CHECKING:
    import numpy.typing as npt

    from .. import utils
    from ..viz import plotters

    Array = utils.Array
    Axis = Union[Array, "Linspace"]
    type Grid1D[AxisT: Axis] = tuple[AxisT]
    type Grid2D[Axis0: Axis, Axis1: Axis] = tuple[Axis0, Axis1]
    UniformGrid2D = Grid2D["Linspace", "Linspace"]
    AnyGrid = Grid1D[Axis] | Grid2D[Axis, Axis]
    Domain = Literal["time", "frequency", "time-frequency"]

    from types import ModuleType

    from l2d_interface import contract

    class Representation[GridT: AnyGrid](
        contract.Representation[Domain, GridT, str | None], Protocol
    ):
        """Protocol for any representation type."""

        entries: Array  # type: ignore[assignment] # Necessary due to missing data array API

        @property
        def grid(self) -> GridT: ...  # noqa: D102

        def __init__(
            self,
            grid: AnyGrid,
            entries: "Array",
        ): ...

        def create_like(self, entries: Any) -> Self:
            """Create a new instance with the same grid and type but different entries."""
            ...


from pywavelet import set_backend as _pyw_set_backend
from pywavelet.transforms import from_freq_to_wavelet as _pyw_f2w
from pywavelet.transforms import from_wavelet_to_freq as _pyw_w2f
from pywavelet.types import FrequencySeries as pywFS
from pywavelet.types import Wavelet as pywWDM

# NOTE We could also import the individual transformation routines from
# pywavelet (written in numpy, cupy, jax) for finer control

# temporary: force backend to be numpy. This should be removed when
# tlt is updated to use multiple array backends.
_pyw_set_backend("numpy", "float64")
# pyw_set_precision("float64")  # by default pywavelet uses float32.


from .. import lib, utils
from . import tapering

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..viz import plotters


_slice = slice  # Alias for slice


@final
class Linspace:
    """Class for a uniform grid.

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
        """The start of the grid."""
        return self._start

    @property
    def step(self) -> float:
        """The step of the grid."""
        return self._step

    @property
    def num(self) -> int:
        """The number of points in the grid."""
        return self._num

    @property
    def shape(self) -> tuple[int]:
        """The shape of the grid."""
        return self._shape

    @property
    def stop(self) -> float:
        """The stop of the grid."""
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
        """Return the length of the grid."""
        return self.num

    def __repr__(self):
        """Return the string representation of the grid."""
        return f"Linspace(start={self.start}, step={self.step}, num={self.num})"

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[np.floating]:
        """Return the grid as a numpy array."""
        grid = self.start + self.step * np.arange(self.num, dtype=dtype)
        if copy is False:
            return grid
        return np.array(grid, copy=True)

    def __getitem__(self, slice: object) -> Self:
        """Return a subset of the grid."""
        if not isinstance(slice, _slice):
            raise TypeError(f"Invalid index type: expected slice, got {type(slice)}.")
        slice_idx = slice.indices(self.num)
        start = self.start + self.step * slice_idx[0]
        step = self.step * slice_idx[2]
        num = len(range(*slice_idx))
        return type(self)(start=start, step=step, num=num)

    @classmethod
    def from_array(cls, array: "Array") -> Self:
        """Create a Linspace from an array."""
        xp = xpc.get_namespace(array)
        if len(array) < 2:
            raise ValueError(
                "Array must have at least two elements to create Linspace."
            )
        diff = xp.diff(array)
        if not xp.allclose(diff, diff[0], rtol=1e-8, atol=0):
            raise ValueError("Array must have uniform spacing to create Linspace.")
        return cls(start=float(array[0]), step=float(diff[0]), num=len(array))

    @classmethod
    def make(cls, array: "Array | LinspaceLike") -> "Linspace":
        """Create a Linspace from a numpy array or return the input if already Linspace."""
        if isinstance(array, Linspace):
            return array
        if isinstance(array, LinspaceLike):
            return Linspace(start=array.start, step=array.step, num=len(array))
        return cls.from_array(array)

    @classmethod
    def get_step(cls, grid: "Array | LinspaceLike") -> float:
        """Return the step of the uniform grid."""
        if isinstance(grid, LinspaceLike):
            return grid.step
        return cls.from_array(grid).step


def _get_entry_grid_shape(entries: "Array") -> tuple[int, ...]:
    return entries.shape[4:]  # Remove batch, channels, harmonics, features dimensions


def _check_entry_grid_compatibility(grid: AnyGrid, entries: "Array") -> None:
    grid_shape = tuple(len(g) for g in grid)
    entry_grid_shape = _get_entry_grid_shape(entries)
    if grid_shape != entry_grid_shape:
        raise ValueError(
            f"Incompatible grid and entries shapes: expected {grid_shape}, got {entry_grid_shape}."
        )


def _get_full_slice(grid_slices: tuple[_slice, ...]) -> tuple[_slice, ...]:
    """Return the slice tuple for the canonical entries array given the grid slices."""
    return (
        slice(None),
    ) * 4 + grid_slices  # batch, channels, harmonics, features stay intact


def _take_subset[GridT: AnyGrid](
    grid: GridT,
    entries: "Array",
    grid_slices: tuple[_slice, ...],
) -> tuple[GridT, "Array"]:
    if len(grid) != len(grid_slices):
        raise ValueError(
            f"Number of slices {len(grid_slices)} does not match number of grid dimensions {len(grid)}."
        )
    _check_entry_grid_compatibility(grid, entries)
    # Slice each grid dimension
    _grid = tuple(g[s] for g, s in zip(grid, grid_slices))
    entries_sliced = entries[_get_full_slice(grid_slices)]
    return cast(GridT, _grid), entries_sliced


def _get_subset_slice(
    grid1d: "Axis",
    *,
    interval: tuple[float, float] | None = None,
    slice: _slice | None = None,
) -> _slice:
    _grid1d = grid1d if not isinstance(grid1d, LinspaceLike) else np.array(grid1d)
    if interval is None:
        # Note that slice(None) is not None
        if slice is None:
            return _slice(None)
        # Otherwise we use the input slice
    else:
        if slice is not None:
            raise ValueError("Only one of `interval` and `slice` should be provided.")
        slice = utils.get_subset_slice(_grid1d, interval[0], interval[1])
    # slice is always a slice object at this point
    return slice


def _set_value(entries: "Array", slice: _slice, value: Any) -> None:
    try:
        entries[slice] = value
    except TypeError:
        entries = cast("Array", entries.at[slice].set(value))  # type: ignore[assignment, union-attr]


def to_array(ary: Axis, xp: ModuleType = np) -> "Array":
    """Convert an axis to an array if it is a Linspace, otherwise return it as is."""
    if isinstance(ary, LinspaceLike):
        return xp.array(Linspace.make(ary))
    return ary


def _to_linspace_if_possible(ary: Union["Array", LinspaceLike]):
    try:
        return Linspace.make(ary)
    except ValueError:
        return ary


def _get_axis_onset(axis: Axis) -> float:
    try:
        return axis.start  # type: ignore[union-attr]
    except AttributeError:
        return float(axis[0])  # type: ignore[union-index, arg-type]


def _get_axis_end(axis: Axis) -> float:
    try:
        return axis.stop  # type: ignore[union-attr]
    except AttributeError:
        return float(axis[-1])  # type: ignore[union-index, arg-type]


class _BinaryUnaryOpMixin(lib.mixins.NDArrayMixin, abc.ABC):
    entries: "Array"

    @abc.abstractmethod
    def create_like(self, entries: Any) -> Self: ...

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
            del entries
            return self

        return self.create_like(entries)

    def _unary_op(self, op: Callable[..., Any], /, **kwargs: Any) -> Self:
        out_arg = kwargs.get("out", None)
        if out_arg is not None:
            kwargs["out"] = self._unwrap(out_arg)
        entries = op(self.entries, **kwargs)
        return self.create_like(entries)


class _InitMixin[GridT: AnyGrid](abc.ABC):
    @property
    def grid(self) -> GridT:
        return cast(GridT, self._grid)

    def __init__(
        self,
        grid: AnyGrid,  # on purpose not GridT to allow more flexible input types
        entries: "Array",
    ):
        self._grid = tuple(_to_linspace_if_possible(g) for g in grid)  # pyright: ignore[reportUnannotatedClassAttribute]
        self.entries: "Array" = entries

    def __repr__(self) -> str:
        return f"{type(self).__name__}(grid={self.grid!r}, entries={self.entries!r}, {self.entries.dtype!r})"

    @property
    @abc.abstractmethod
    def domain(self) -> Domain:
        """Physical domain of the representation."""

    @property
    @abc.abstractmethod
    def kind(self) -> str | None:
        """Optional semantic kind of representation."""

    @property
    def n_batches(self) -> int:
        """Return the number of batches."""
        return self.entries.shape[0]

    @property
    def n_channels(self) -> int:
        """Return the number of channels."""
        return self.entries.shape[1]

    @property
    def n_harmonics(self) -> int:
        """Return the number of harmonics."""
        return self.entries.shape[2]

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return self.entries.shape[3]

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Return the shape of the grid dimensions."""
        return _get_entry_grid_shape(self.entries)


class _Subset1DMixin[GridT: "Grid1D[Axis]"](_InitMixin[GridT], abc.ABC):
    def get_subset(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: _slice | None = None,
        copy: bool = True,
    ) -> Self:
        """Return the subset as a new instance."""
        _slice = _get_subset_slice(self.grid[0], interval=interval, slice=slice)
        grid, entries = _take_subset(self.grid, self.entries, (_slice,))
        entries = entries.copy() if copy else entries
        return type(self)(grid=grid, entries=entries)

    def __getitem__(self, slice: _slice) -> Self:
        """Return the view of a subset of the series."""
        return self.get_subset(slice=slice, copy=False)

    def __setitem__(self, slice: _slice, value: Any) -> None:
        """Set entries at slice location."""
        # NOTE this method does not check the compatibility of the grids,
        # and assumes that the slice is correct.
        _set_value(self.entries, slice, value)


def _embed_entries_to_grid[GT: "AnyGrid"](
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


class _ArithmeticReprOnGrid[GridT: "AnyGrid"](
    _BinaryUnaryOpMixin, _InitMixin[GridT], abc.ABC
):
    # Provides implementations for arithmetic operations

    def create_like(self, entries: "Array"):
        """Create a new instance with the same grid as the current one."""
        return type(self)(grid=self.grid, entries=entries)

    def __xp__(self, api_version: str | None = None):
        return xpc.get_namespace(self.entries, api_version=api_version)

    @property
    def xp(self):
        return self.__xp__()

    def _check_grid_compatibility(self, other: object) -> bool:
        if not hasattr(other, "grid"):
            return False
        other = cast(_ArithmeticReprOnGrid[GridT], other)
        other_grid = other.grid
        if len(self.grid) != len(other_grid):
            return False
        for g1, g2 in zip(self.grid, other_grid):
            if isinstance(g1, Linspace) and isinstance(g2, Linspace):
                if g1 != g2:
                    return False
            else:
                xp = xpc.get_namespace(g1)
                if not xp.array_equal(g1, g2):
                    return False
        return True

    def _unwrap(self, other: object):
        if hasattr(other, "grid") and hasattr(other, "entries"):
            other = cast(_ArithmeticReprOnGrid[GridT], other)
            if self._check_grid_compatibility(other):
                return other.entries
            raise ValueError(
                f"AnyGrid mismatch: expected {self.grid}, got {other.grid}."
            )
        return other

    def add(self, other: Self, slice: _slice, inplace: bool = False) -> Self:
        """Add another series on a sub-grid with known slice.

        This method adds another series on a sub-grid of the current series
        with a known slice, which is used to select the entries of the current
        series to be added with.

        If `inplace` is True, the current series is modified in place
        and returned (equivalent to calling :meth:`.iadd`). Otherwise,
        a new series is returned with the result of the addition.
        Default is False.

        See Also
        --------
        :meth:`.iadd`
        """
        if inplace:
            return self.iadd(other, slice)
        self_copy = self.create_like(self.entries.copy())
        self_copy.iadd(other, slice)
        return self_copy

    def iadd(self, other: Self, slice: _slice) -> Self:
        """Add another series on a sub-grid with known slice in place.

        See Also
        --------
        :meth:`.add`
        """
        try:
            _set_value(self.entries, slice, self.entries[slice] + other.entries)
        except ValueError as e:
            raise ValueError(
                "You may want to first embed the series instances to super-grids before "
                + "adding them, if their grids are not compatible."
            ) from e
        return self

    def __iadd__(self, other: object) -> Self:
        """Add another series in place.

        Note
        ----
        Compared to :meth:`.iadd`, this method computes automatically the slice
        of the subgrid to apply with a generic algorithm. This is not the most
        efficient in some cases. If you have a specialised and more efficient
        way of computing the slice, you should use the :meth:`.iadd` method
        instead.

        See Also
        --------
        :meth:`.iadd`
        :meth:`.add`
        :meth:`.__add__`
        """
        if isinstance(other, type(self)):
            other_grid_1d = other.grid[0]
            if isinstance(other_grid_1d, LinspaceLike):
                start, stop = other_grid_1d.start, other_grid_1d.stop
            else:
                start, stop = float(other_grid_1d[0]), float(other_grid_1d[-1])

            if len(self.grid) < len(other.grid):
                raise ValueError(
                    "In-place addition requires the series to add to "
                    + "be a sub-grid of the current one. Expect `other.grid` "
                    + "to be shorter than `self.grid`."
                )
            _slice = utils.get_subset_slice(to_array(self.grid[0]), start, stop)
            return self.iadd(other, slice=_slice)
        return super().__iadd__(other)


class _Uniform1DMixin(abc.ABC):
    @property
    @abc.abstractmethod
    def grid(self) -> "Grid1D[Linspace]": ...

    @property
    def resolution(self) -> float:
        return self.grid[0].step


@overload
def frequency_series(
    frequencies: "Linspace",
    entries: "Array",
) -> "UniformFrequencySeries": ...


@overload
def frequency_series[AxisT: Axis](
    frequencies: AxisT,
    entries: "Array",
) -> "FrequencySeries[AxisT]": ...


def frequency_series[AxisT: Axis](
    frequencies: AxisT,
    entries: "Array",
) -> Union["FrequencySeries[AxisT]", "UniformFrequencySeries"]:
    """Build a :class:`FrequencySeries` or :class:`UniformFrequencySeries`."""
    try:
        _frequencies = Linspace.make(frequencies)
    except ValueError:
        return FrequencySeries[AxisT]((frequencies,), entries)
    else:
        return UniformFrequencySeries((_frequencies,), entries)


@overload
def time_series(
    times: "Linspace",
    entries: "Array",
) -> "UniformTimeSeries": ...


@overload
def time_series[AxisT: Axis](
    times: AxisT,
    entries: "Array",
) -> "TimeSeries[AxisT]": ...


def time_series[AxisT: Axis](
    times: AxisT,
    entries: "Array",
) -> Union["TimeSeries[AxisT]", "UniformTimeSeries"]:
    """Build a :class:`TimeSeries` or :class:`UniformTimeSeries`."""
    try:
        _times = Linspace.make(times)
    except ValueError:
        return TimeSeries[AxisT]((times,), entries)
    else:
        return UniformTimeSeries((_times,), entries)


def phasor[AxisT: Axis](
    frequencies: AxisT,
    amplitudes: "Array",
    phases: "Array",
) -> "Phasor[AxisT]":
    """Build a :class:`Phasor`."""
    return Phasor[AxisT].make(
        frequencies=frequencies, amplitudes=amplitudes, phases=phases
    )


def stft[FreqAxisT: Axis, TimeAxisT: Axis](
    frequencies: FreqAxisT,
    times: TimeAxisT,
    entries: "Array",
) -> "STFT[FreqAxisT, TimeAxisT]":
    """Build an :class:`STFT`."""
    return STFT[FreqAxisT, TimeAxisT]((frequencies, times), entries)


def wdm(
    frequencies: Axis,
    times: Axis,
    entries: "Array",
) -> "WDM":
    """Build a :class:`WDM`."""
    return WDM((frequencies, times), entries)


class _1DSeries[AxisT: Axis](  # pyright: ignore[reportUnsafeMultipleInheritance]
    _ArithmeticReprOnGrid["Grid1D[AxisT]"],
    _Subset1DMixin["Grid1D[AxisT]"],
    abc.ABC,
): ...


class FrequencySeries[AxisT: Axis](_1DSeries[AxisT]):
    """A series of numbers on a frequency grid."""

    @property
    def domain(self) -> Literal["frequency"]:
        """The physical domain of the representation."""
        return "frequency"

    @property
    def kind(self) -> None:
        """The semantic kind of the representation."""
        return None

    def angle(self, **kwargs: Any):
        """Return the angle of the series."""
        return super().angle(**kwargs).unwrap(period=2 * self.xp.pi)

    @property
    def frequencies(self):
        """The frequencies of the series."""
        return self.grid[0]

    @property
    def f_min(self) -> float:
        """The minimum frequency of the series."""
        return _get_axis_onset(self.frequencies)

    @property
    def f_max(self) -> float:
        """The maximum frequency of the series."""
        return _get_axis_end(self.frequencies)

    def get_time_shifted(self, shift: float) -> Self:
        """Shift the series in time."""
        return self * self.xp.exp(
            -2j * self.xp.pi * self.xp.array(self.frequencies) * shift
        )

    def get_embedded[AT: "Axis"](
        self,
        embedding_grid: "Grid1D[AT]",
        *,
        known_slices: tuple[slice, ...] | None = None,
    ):
        """Return the series embedded in a new 1D grid."""
        grid, entries = _embed_entries_to_grid(
            self.grid,
            self.entries,
            embedding_grid,
            known_slices=known_slices,
        )
        return frequency_series(grid[0], entries)

    def get_plotter(self):
        """Return the plotter for the series."""
        from ..viz import plotters

        return plotters.FSPlotter(self)


class UniformFrequencySeries(FrequencySeries[Linspace], _Uniform1DMixin):
    """A frequency series on a uniform frequency grid."""

    @property
    def df(self) -> float:
        """The frequency spacing. Alias for :attr:`.resolution`."""
        return self.resolution

    def irfft(
        self,
        time_grid: "Array",
        *args: tapering.Tapering | None,
        tapering: tapering.Tapering | None = None,
    ):
        """Inverse real FFT of the series."""
        if len(args) > 1:
            raise TypeError("irfft() accepts at most one positional optional argument.")
        if len(args) == 1:
            if tapering is not None:
                raise TypeError(
                    "irfft() received `tapering` as both positional and keyword arguments."
                )
            warnings.warn(
                "Passing `tapering` positionally to `irfft` is deprecated and will be removed "
                + "in 0.7.0; pass it as a keyword argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            tapering = args[0]
        self_frequencies = to_array(self.frequencies)
        tapering_window = tapering(self_frequencies) if tapering is not None else 1.0
        dt = float(time_grid[1] - time_grid[0])
        nyquist_dt = 1 / (2 * self_frequencies[-1])
        if dt < nyquist_dt and not self.xp.isclose(dt, nyquist_dt):
            # FIXME spurious warning for odd, small n
            # probably related to the (n-1)/2 in the last frequency in rfftfreq
            warnings.warn("The time grid is denser than the Nyquist limit.")

        return UniformTimeSeries(
            grid=(time_grid,),
            entries=self.xp.fft.irfft(
                self.entries * tapering_window / dt, n=len(time_grid)
            ),
        )

    def to_wdm(
        self,
        /,
        *,
        Nf: int | None = None,
        Nt: int | None = None,
        nx: float = 4.0,
    ) -> WDM:
        """Transform the frequency series to a WDM representation.

        This method performs a forward wavelet transform, converting a
        frequency series into a wavelet representation.

        At least one of `Nf` and `Nt` must be provided.

        .. warning::

            The WDM transform on discrete-time or discrete-frequency data
            is inherently lossy, since WDM is a Wilson basis conceived for
            continuous-time functions. The smaller ``(Nf, Nt)`` are,
            the more lossy the transform is. You must make sure
            they are large enough for your needs. Use the inverse transform
            :meth:`.WDM.to_frequency_series` to quantify the loss.
            Even numbers for ``Nf`` and ``Nt`` are recommended.

        .. note::

            This method first transforms the frequency series to :class:`pywavelet.types.FrequencySeries` object,
            then leverages the forward wavelet transform implemented in pywavelet to get the WDM representation.
            Note that in pywavelet, the degree of freedom of a frequency series of length ``K`` is ``2 * (K - 1)``,
            even though the length of the original time series could be ``2*K - 1`` as well.

        Parameters
        ----------
        Nf : int | None
            The number of frequency bins in the WDM representation. Note that this
            is smaller than the number of frequency bins in the original frequency series.

        Nt : int | None
            The number of time bins in the WDM representation.

        nx : float
            Shape parameter controling the width of the wavelets.
        """
        ndof = 2 * (len(self.frequencies) - 1)  # pywavelet's convention
        dt = 1 / (ndof * self.df)
        fs = pywFS(
            data=self.entries.squeeze() / dt,
            freq=self.xp.array(self.frequencies),
            t0=0.0,
        )
        return WDM.from_pywWDM(_pyw_f2w(fs, Nf=Nf, Nt=Nt, nx=nx))

    def to_WDM(
        self,
        /,
        *,
        Nf: int | None = None,
        Nt: int | None = None,
        nx: float = 4.0,
    ) -> WDM:
        """Return :meth:`to_wdm` while warning about deprecation.

        This alias will be removed in 0.7.0.
        """
        warnings.warn(
            "`to_WDM` is deprecated and will be removed in 0.7.0; "
            + "use `to_wdm` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_wdm(Nf=Nf, Nt=Nt, nx=nx)


class TimeSeries[AxisT: Axis](_1DSeries[AxisT]):
    """A series of numbers on a time grid."""

    @property
    def domain(self) -> Literal["time"]:
        """The physical domain of the representation."""
        return "time"

    @property
    def kind(self) -> None:
        """The semantic kind of the representation."""
        return None

    @property
    def times(self) -> Axis | Linspace:
        """The times of the series. Alias for :attr:`.grid`."""
        return self.grid[0]

    @property
    def t_start(self) -> float:
        """The onset time of the series."""
        return _get_axis_onset(self.times)

    @property
    def t_end(self) -> float:
        """The end time of the series."""
        return _get_axis_end(self.times)

    def get_plotter(self):
        """Return the plotter for the series."""
        from ..viz import plotters

        return plotters.TSPlotter(self)

    def get_embedded[AT: "Axis"](
        self,
        embedding_grid: "Grid1D[AT]",
        *,
        known_slices: tuple[slice, ...] | None = None,
    ):
        """Return the series embedded in a new 1D grid."""
        grid, entries = _embed_entries_to_grid(
            self.grid,
            self.entries,
            embedding_grid,
            known_slices=known_slices,
        )
        return time_series(grid[0], entries)


class UniformTimeSeries(TimeSeries[Linspace], _Uniform1DMixin):
    """A time series on a uniform time grid."""

    @property
    def dt(self) -> float:
        """The time step. Alias for :attr:`.resolution`."""
        return self.resolution

    def rfft(
        self,
        *args: tapering.Tapering | None,
        tapering: tapering.Tapering | None = None,
    ):
        """Fast Fourier transform of the series.

        .. note::
            Unlike the inverse transform :meth:`.FrequencySeries.irfft`, this method does
            not allow taking a frequency grid as input. Time series are considered as
            primary representations for DATA, in the sense that they are the most directly
            related to what we measure.
        """
        if len(args) > 1:
            raise TypeError("rfft() accepts at most one positional optional argument.")
        if len(args) == 1:
            if tapering is not None:
                raise TypeError(
                    "rfft() received `tapering` as both positional and keyword arguments."
                )
            warnings.warn(
                "Passing `tapering` positionally to `rfft` is deprecated and will be removed "
                + "in 0.7.0; pass it as a keyword argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            tapering = args[0]
        self_times = self.xp.array(self.times)
        tapering_window = (
            tapering(self_times)
            if tapering is not None
            else self.xp.ones_like(self_times)
        )
        return UniformFrequencySeries(
            grid=(self.xp.fft.rfftfreq(len(self.times), d=self.dt),),
            entries=self.xp.fft.rfft(self.entries * tapering_window * self.dt),
        )

    # NOTE win cannot be a Tapering object. It's probably worth designing this
    # interface to be consistent with the rest of tlt and close to scipy's stft
    def stfft(self, win: npt.NDArray[np.floating], hop: int):
        """Short-time Fourier transform of the series."""
        SFT = scipy.signal.ShortTimeFFT(win=win, hop=hop, fs=1 / self.dt)
        times = SFT.t(len(self.times)) + np.asarray(self.times)[0]
        freqs = SFT.f
        Sx = SFT.stft(np.asarray(self.entries) * self.dt)
        return stft(freqs, times, Sx)

    # def to_WDM(
    #     self,
    #     /,
    #     *,
    #     Nf: int | None = None,
    #     Nt: int | None = None,
    #     nx: float = 4.0,
    #     mult: int = 32,
    # ):
    #     """Transform the time series to a WDM representation.

    #     This method performs a forward wavelet transform, converting a
    #     time series into a wavelet representation.

    #     At least one of `Nf` and `Nt` must be provided.

    #     .. warning::

    #         The WDM transform on discrete-time or discrete-frequency data
    #         is inherently lossy, since WDM is a Wilson basis conceived for
    #         continuous-time functions. The smaller ``(Nf, Nt)`` are,
    #         the more lossy the transform is. You must make sure
    #         they are large enough for your needs. Use the inverse transform
    #         :meth:`.WDM.to_time_series` to quantify the loss.
    #         Even numbers for ``Nf`` and ``Nt`` are recommended.

    #     .. note::

    #         This method first transforms the time series to :class:`pywavelet.types.TimeSeries` object,
    #         then leverages the forward wavelet transform implemented in pywavelet to get the WDM representation.

    #     Parameters
    #     ----------
    #     Nf : int | None
    #         The number of frequency points in the WDM representation.
    #     Nt : int | None
    #         The number of time points in the WDM representation.
    #     nx : float
    #         Shape parameter controling the width of the wavelets.
    #     mult : int
    #         Number of time points to use for the wavelet transform.
    #         Ensure `mult` is not larger than half of the number of time points `Nt`.
    #     """
    #     # pywavelet's time series' last point is not the end of the time grid,
    #     # but the start of the last time bin.
    #     fs = pywTS(data=self.entries, t=np.array(self.times))
    #     return WDM.from_pywWDM(_pyw_t2w(fs, Nf=Nf, Nt=Nt, nx=nx, mult=mult))


class Phasor[AxisT: Axis](
    _Subset1DMixin["Grid1D[AxisT]"],
):
    """Phasor representation.

    A phasor is a couple of amplitude and phase that represent a complex number.
    This class encapsulates a sequence of phasors at different frequencies, which
    can be used to represent a waveform. This representation is useful for
    interpolating waveforms generated on a sparse grid of frequencies to a dense
    grid of frequencies.

    The input phases are expected to be smooth, without zigzags, so as the real
    and imaginary parts of the amplitudes. This is crucial for the interpolation
    to work properly.

    .. note:: The so-called amplitude is itself complex number in general.
    """

    @property
    def domain(self) -> Literal["frequency"]:
        """The physical domain of the representation."""
        return "frequency"

    @property
    def kind(self):
        """The semantic kind of the representation."""
        return "phasor"

    @property
    def phases(self) -> "Array":
        """The phases of the phasors."""
        return self.entries[..., slice(1, 2), :]

    @property
    def amplitudes(self) -> "Array":
        """The amplitudes of the phasors."""
        return self.entries[..., slice(0, 1), :]

    @property
    def frequencies(self) -> Axis | Linspace:
        """The frequencies of the phasors. Alias for :attr:`.grid`."""
        return self.grid[0]

    @property
    def f_min(self) -> float:
        """The minimum frequency of the series."""
        return _get_axis_onset(self.frequencies)

    @property
    def f_max(self) -> float:
        """The maximum frequency of the series."""
        return _get_axis_end(self.frequencies)

    @classmethod
    def make(
        cls,
        *,
        frequencies: "Axis",
        amplitudes: "Array",
        phases: "Array",
    ):
        """Create a phasor from amplitudes and phases."""
        xp = xpc.get_namespace(amplitudes, phases)
        # We assume input amplitudes and phases are 1D arrays
        return cls(
            grid=(frequencies,),
            entries=xp.stack((amplitudes, phases), axis=0)[None, None, None, ...],
        )

    def __setitem__(self, slice: _slice, value: Any) -> None:
        """Set the entries and phases of a subset of the phasor."""
        _set_value(self.entries, slice, value)

    def create_like(self, entries: "Array"):
        """Create a new series with the same grid as the current one."""
        return type(self)(grid=self.grid, entries=entries)

    def get_embedded[AT: "Axis"](
        self,
        embedding_grid: "Grid1D[AT]",
        *,
        known_slices: tuple[slice, ...] | None = None,
    ) -> Self:
        """Return the phasor embedded in a new 1D grid."""
        grid, entries = _embed_entries_to_grid(
            self.grid,
            self.entries,
            embedding_grid,
            known_slices=known_slices,
        )
        return type(self)(grid=cast("Grid1D[AxisT]", grid), entries=entries)

    def get_interpolated[AT: Axis](
        self,
        frequencies: AT,
        interpolator: Callable[[Any, Any], Callable[[Any], Any]],
    ) -> "Phasor[AT]":
        """Get the phasors interpolated to the given frequencies."""
        xp = xpc.get_namespace(self.amplitudes, self.phases)
        _frequencies = to_array(frequencies, xp=xp)
        self_freq = to_array(self.frequencies, xp=xp)
        if self.entries.shape != (1, 1, 1, 2, len(self_freq)):
            raise ValueError(
                f"Only 1D phasors with shape (1, 1, 1, 2, {len(self_freq)}) are supported"
                + f"for interpolation, but got shape {self.entries.shape}."
            )
        amp_real = self.amplitudes.real.squeeze()
        amp_imag = self.amplitudes.imag.squeeze()
        amplitudes_real = interpolator(self_freq, amp_real)(_frequencies)
        amplitudes_imag = interpolator(self_freq, amp_imag)(_frequencies)
        amplitudes = amplitudes_real + 1j * amplitudes_imag
        phases = interpolator(self_freq, self.phases.squeeze())(_frequencies)
        return Phasor[AT].make(
            frequencies=_frequencies, amplitudes=amplitudes, phases=phases
        )

    @overload
    def to_frequency_series(self: "Phasor[Linspace]") -> "UniformFrequencySeries": ...

    @overload
    def to_frequency_series[AT: "Axis"](
        self: "Phasor[AT]",
    ) -> "FrequencySeries[AT]": ...

    def to_frequency_series(self):
        """Get the :class:`.FrequencySeries` representation of the waveform."""
        xp = xpc.get_namespace(self.amplitudes, self.phases)
        return frequency_series(
            self.frequencies,
            self.amplitudes * xp.exp(1j * self.phases),
        )

    def get_plotter(self):
        """Return the plotter for the phasor."""
        from ..viz import plotters

        return plotters.PhasorPlotter(self)


class STFT[
    FreqAxisT: Axis,
    TimeAxisT: Axis,
](
    _ArithmeticReprOnGrid["Grid2D[FreqAxisT, TimeAxisT]"],
):
    """Time-frequency representation."""

    @property
    def domain(self) -> Literal["time-frequency"]:
        """The physical domain of the representation."""
        return "time-frequency"

    @property
    def kind(self) -> str:
        """The semantic kind of the representation."""
        return "stft"

    @property
    def times(self) -> TimeAxisT:
        """The time grid of the time-frequency representation."""
        return self.grid[1]

    @property
    def t_start(self) -> float:
        """The onset time of the series."""
        return _get_axis_onset(self.times)

    @property
    def t_end(self) -> float:
        """The end time of the series."""
        return _get_axis_end(self.times)

    @property
    def frequencies(self) -> FreqAxisT:
        """The frequency grid of the time-frequency representation."""
        return self.grid[0]

    @property
    def f_min(self) -> float:
        """The minimum frequency of the series."""
        return _get_axis_onset(self.frequencies)

    @property
    def f_max(self) -> float:
        """The maximum frequency of the series."""
        return _get_axis_end(self.frequencies)

    @classmethod
    def make(
        cls,
        *,
        times: "Array",
        frequencies: "Array",
        entries: "Array",
    ) -> Self:
        """Create a time-frequency representation from time and frequency grids and entries."""
        return cls(grid=(frequencies, times), entries=entries)

    def get_embedded[AT0: "Axis", AT1: "Axis"](
        self,
        embedding_grid: "Grid2D[AT0, AT1]",
        *,
        known_slices: tuple[slice, ...] | None = None,
    ):
        """Return the representation embedded in a new 2D grid."""
        grid, entries = _embed_entries_to_grid(
            self.grid,
            self.entries,
            embedding_grid,
            known_slices=known_slices,
        )
        return stft(grid[0], grid[1], entries)

    # TODO for consistency: receive slices, receive copy bool arg
    def get_subset(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
    ) -> Self:
        """Return the subset as a new instance."""
        if time_interval is not None:
            time_slice = utils.get_subset_slice(
                to_array(self.times, xp=self.xp), time_interval[0], time_interval[1]
            )
        else:
            time_slice = _slice(None)
        if freq_interval is not None:
            freq_slice = utils.get_subset_slice(
                to_array(self.frequencies, xp=self.xp),
                freq_interval[0],
                freq_interval[1],
            )
        else:
            freq_slice = _slice(None)
        time_grid_sliced = to_array(self.times, xp=self.xp)[time_slice]
        freq_grid_sliced = to_array(self.frequencies, xp=self.xp)[freq_slice]
        return type(self)(
            grid=(freq_grid_sliced, time_grid_sliced),
            entries=self.entries[_get_full_slice((freq_slice, time_slice))],
        )

    def get_plotter(self):
        """Return the plotter for the representation."""
        from ..viz import plotters

        return plotters.STFTPlotter(self)


class WDM(_ArithmeticReprOnGrid["UniformGrid2D"]):
    """
    Wilson-Daubechies-Meyer (WDM) time-frequency representation.

    This represents data using an evenly-spaced 2D grid in the
    time-frequency plane with shape (Nf, Nt). Each "pixel" has size
    ΔF ΔT = 1/2. The times range approximately from 0 to the final
    observation time, while the frequencies range from 0 to the
    Nyquist limit (half the sampling rate).

    Currently, transformations to/from FrequencySeries are allowed,
    but only for full series --- all frequencies and all times.

    .. warning::
        Elsewhere in this codebase, a grid of N points is considered to have N-1 bins,
        since the first point is the start of the first bin and the last point is
        the end of the last bin. However, in the WDM representation, due to the
        convention of `pywavelet` which is the working horse for the WDM transform,
        a grid of N points is considered to have N bins. This needs reviewing
        and fixing in the future, but for now users should be aware of this inconsistency.

    Parameters
    ----------
    times: real 1D array
        "Array" of evenly-spaced times with separation ΔT and size `Nt`.

    frequencies: real 1D array
        "Array" of evenly-spaced frequencies with separation ΔF and size `Nf`.

    entries: real 2D array
        "Array" of data entries, with shape `(Nf, Nt)`.
    """

    @property
    def domain(self) -> Literal["time-frequency"]:
        """The physical domain of the representation."""
        return "time-frequency"

    @property
    def kind(self) -> str:
        """The semantic kind of the representation."""
        return "WDM"

    @property
    def times(self) -> Linspace:
        """The time grid of the time-frequency representation."""
        return self.grid[1]

    @property
    def frequencies(self) -> Linspace:
        """The frequency grid of the time-frequency representation."""
        return self.grid[0]

    @property
    def dT(self) -> float:
        """Time resolution (ΔT) of the time-frequency grid."""
        return self.times.step

    @property
    def dF(self) -> float:
        """Frequency resolution (ΔF) of the time-frequency grid."""
        return self.frequencies.step

    @classmethod
    def make(
        cls,
        *,
        times: Union["Array", Linspace],
        frequencies: Union["Array", Linspace],
        entries: "Array",
    ):
        """Create a WDM representation from time and frequency grids and entries."""
        return cls(
            grid=(Linspace.make(frequencies), Linspace.make(times)), entries=entries
        )

    def is_critically_sampled(self):
        """Return True if :attr:`.dT` * :attr:`.dF` = 1/2."""
        # I don't like how this method is implemented,
        # but I don't see a better way for now.
        return self.xp.isclose(self.dT * self.dF, 1 / 2)

    @property
    def Nt(self) -> int:
        """Number of time points.

        .. note::
            Throughout this codebase, a grid of N points is considered to have N-1 bins,
            since the first point is the start of the first bin and the last point is
            the end of the last bin.
        """
        return self.times.num

    @property
    def Nf(self) -> int:
        """Number of frequency points.

        .. note::
            Throughout this codebase, a grid of N points is considered to have N-1 bins,
            since the first point is the start of the first bin and the last point is
            the end of the last bin.
        """
        return self.frequencies.num

    @property
    def ND(self) -> int:
        """Total number of data points in the time-frequency plane."""
        return self.times.num * self.frequencies.num

    @property
    def duration(self) -> float:
        """Total signal duration."""
        # return self.times.stop - self.times.start
        # Given that a grid of N points has N bins in this class
        times = self.times
        return len(times) * times.step

    @property
    def sample_interval(self) -> float:
        """
        Time resolution of a TimeSeries corresponding to this WDM.

        Smaller than the wavelet time bin :attr:`.dT`.
        """
        return self.duration / self.ND

    dt: property = sample_interval
    """Alias for :attr:`.sample_interval`."""

    @property
    def df(self) -> float:
        """
        Frequency resolution of a FrequencySeries corresponding to this WDM.

        Smaller than the wavelet frequency bin :attr:`.dF`.
        """
        return 1 / self.duration

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (:attr:`.Nf`, :attr:`.Nt`) of the wavelet grid."""
        return self.Nf, self.Nt

    @property
    def sample_rate(self) -> float:
        """Sampling rate."""
        # We can verify that this is twice (self.frequencies.stop + self.frequencies.step)
        # which is the true maximum frequency in the WDM representation, again due to
        # the special convention in this class that a grid of N points has N bins.
        return 1 / self.sample_interval

    fs: property = sample_rate
    """Alias for :attr:`.sample_rate`."""

    def to_frequency_series(
        self, *, nx: float = 4.0, mask: npt.NDArray[np.bool_] | None = None
    ) -> UniformFrequencySeries:
        """Perform an inverse wavelet transform to the frequency domain.

        Parameters
        ----------
        nx : float
            Shape parameter controling the width of the wavelets, defaults to 4.0.
        mask : npt.NDArray[np.bool] | None
            Mask to apply on the frequencies and entries of the result, useful
            to avoid singularities. Defaults to None.
        """
        pywwv = self._to_pywWDM()
        # Is there a reason why pywavelet accepts the time step
        # instead of computing it from the WDM representation itself?
        pywfs = cast(Any, _pyw_w2f(pywwv, self.dt, nx))
        freqs = pywfs.freq
        entries = pywfs.data * pywfs.dt
        # To see if we keep or not
        # https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit/-/merge_requests/4#note_576666
        if mask is not None:
            freqs = np.ma.masked_where(mask, freqs)  # type: ignore
            entries = np.ma.masked_where(mask, entries)  # type: ignore
        return UniformFrequencySeries((freqs,), entries[None, None, None, None, ...])  # type: ignore

    def to_freqseries(
        self, *, nx: float = 4.0, mask: npt.NDArray[np.bool_] | None = None
    ) -> UniformFrequencySeries:
        """Return :meth:`to_frequency_series` while warning about deprecation.

        This alias will be removed in 0.7.0.
        """
        warnings.warn(
            "`to_freqseries` is deprecated and will be removed in 0.7.0; "
            + "use `to_frequency_series` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_frequency_series(nx=nx, mask=mask)

    @property
    def nyquist(self) -> float:
        """Nyquist frequency (half the sampling rate)."""
        # I don't like this property name
        return self.sample_rate / 2

    @classmethod
    def from_pywWDM(cls, pywwv: pywWDM, /) -> Self:
        """Convert a pywWDM object to a WDM."""
        pywwv_any = cast(Any, pywwv)
        entries = pywwv_any.data[None, None, None, None, ...]
        times = Linspace(
            pywwv_any.time[0],
            pywwv_any.time[1] - pywwv_any.time[0],
            len(pywwv_any.time),
        )
        frequencies = Linspace(
            pywwv_any.freq[0],
            pywwv_any.freq[1] - pywwv_any.freq[0],
            len(pywwv_any.freq),
        )
        return cls.make(times=times, frequencies=frequencies, entries=entries)

    def _to_pywWDM(self) -> pywWDM:
        """Convert self to a pywWDM object."""
        xp = xpc.get_namespace(self.entries)
        return pywWDM(
            data=self.entries.squeeze(),
            time=xp.array(self.times),
            freq=xp.array(self.frequencies),
        )

    def get_plotter(self) -> plotters.WDMPlotter:
        """Return the plotter for the WDM representation."""
        from ..viz import plotters

        return plotters.WDMPlotter(self)

    def get_embedded[AT0: "Axis", AT1: "Axis"](
        self,
        embedding_grid: "Grid2D[AT0, AT1]",
        *,
        known_slices: tuple[slice, ...] | None = None,
    ):
        """Return the representation embedded in a new 2D grid."""
        grid, entries = _embed_entries_to_grid(
            self.grid,
            self.entries,
            embedding_grid,
            known_slices=known_slices,
        )
        return wdm(grid[0], grid[1], entries)

    # TODO for consistency: receive slices, receive copy bool arg
    def get_subset(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
    ) -> Self:
        """Return the subset as a new instance."""
        if time_interval is not None:
            time_slice = utils.get_subset_slice(
                to_array(self.times, xp=self.xp), time_interval[0], time_interval[1]
            )
        else:
            time_slice = _slice(None)
        if freq_interval is not None:
            freq_slice = utils.get_subset_slice(
                to_array(self.frequencies, xp=self.xp),
                freq_interval[0],
                freq_interval[1],
            )
        else:
            freq_slice = _slice(None)
        time_grid_sliced = Linspace.from_array(
            to_array(self.times, xp=self.xp)[time_slice]
        )
        freq_grid_sliced = Linspace.from_array(
            to_array(self.frequencies, xp=self.xp)[freq_slice]
        )
        return type(self)(
            grid=(freq_grid_sliced, time_grid_sliced),
            entries=self.entries[_get_full_slice((freq_slice, time_slice))],
        )
