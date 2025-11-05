"""
Module for representations.

A representation is a format in which detector data are displayed.
The most natural representation for LISA data is a set of time series
obtained upon postprocessing the beat-note signals. Besides, we also
need to wrap waveform templates in a compatible representation with
the data representation, so that we can compare them and perform
operations between them.

In this module, we consider representations for data or signals in a
single channel (and a single mode, if applicable). For the combination
of multiple channels or modes, we use arithmetic dictionaries (see
:mod:`.arithdicts`, :mod:`.data` and :mod:`.waveforms`).

We have two main types of representations, either in a single domain
(time or frequency) or in both domains. Conceptually, we can think of
a grid (1D or 2D) and values loaded on the grid.
In the first case, we implement data classes for the series
(:class:`.TimeSeries`, :class:`.FrequencySeries` and :class:`.Phasor`),
where both the grid and the values are 1D arrays and stored in the
instances. In the second case, we do not store the 2D grid directly
to save memory, but we store the arrays of times, frequencies and
the values for the time-frequency matrices (:class:`.TimeFrequency`,
:class:`.WDM`).

We model these representations as immutable data classes that support
arithmetic operations between them and with numeric values or arrays.
We implement generic classes that accept a certain :class:`numpy.dtype`
as type parameter. This allows us to specify and keep track of the data
type of the values in the representation.

.. currentmodule:: typed_lisa_toolkit.containers.representations

Types
-----
.. autoclass:: Numeric
.. autoclass:: NPNumberT_co
.. autoclass:: NPFloatingT
.. autoclass:: NPTBitT
.. autoclass:: Interpolator

Entities
--------
.. autoprotocol:: Representation

..autoclass:: Linspace
   :members:
   :special-members:

.. autoclass:: TimeSeries
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
   :special-members: __mul__, __rmul__, __add__, __sub__, __truediv__, __rtruediv__, __neg__, __add__, __iadd__

.. autoclass:: FrequencySeries
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
   :special-members: __add__, __iadd__

.. autoclass:: Phasor
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
   :special-members: __add__, __iadd__

.. autoclass:: TimeFrequency
    :members:
    :member-order: groupwise
    :undoc-members:
    :inherited-members:

.. autoclass:: WDM
    :members:
    :member-order: groupwise
    :inherited-members:
    :show-inheritance:
"""

from __future__ import annotations
from collections.abc import Callable
import dataclasses as dc
import logging
from typing import TypeVar, Self, Protocol, TYPE_CHECKING, Any
import warnings
import numpy as np
import numpy.typing as npt
import scipy.signal  # type: ignore[import-untyped]

from pywavelet.types import (  # type: ignore[import-untyped]
    FrequencySeries as pywFS,
    TimeSeries as pywTS,
    Wavelet as pywWDM,
)
from pywavelet import set_backend as _pyw_set_backend  # type: ignore[import-untyped]
from pywavelet.backend import (  # type: ignore[import-untyped]
    cuda_is_available as _pyw_cuda_is_available,
    jax_is_available as _pyw_jax_is_available,
    set_precision as _pyw_set_precision,
)
from pywavelet.transforms import (  # type: ignore[import-untyped]
    from_freq_to_wavelet as _pyw_f2w,
    from_time_to_wavelet as _pyw_t2w,
    from_wavelet_to_freq as _pyw_w2f,
    from_wavelet_to_time as _pyw_w2t,
)

# NOTE We could also import the individual transformation routines from
# pywavelet (written in numpy, cupy, jax) for finer control

# temporary: force backend to be numpy. This should be removed when
# tlt is updated to use multiple array backends.
_pyw_set_backend("numpy", "float64")
# pyw_set_precision("float64")  # by default pywavelet uses float32.


from .. import utils, lib
from . import tapering

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..viz import plotters


PyNum = int | float | complex  # Union[int, float, complex]
Numeric = PyNum | np.number | npt.NDArray[np.number]

NPNumberT_co = TypeVar("NPNumberT_co", bound=np.number, covariant=True)
"""Covariant numpy number data type."""

NPFloatingT = TypeVar("NPFloatingT", bound=np.floating)
"""Numpy floating data type."""

NPTBitT = TypeVar("NPTBitT", bound=npt.NBitBase)
"""Numpy bit data type."""

ArrayFunc = Callable[[npt.NDArray[NPFloatingT]], npt.NDArray[NPFloatingT]]
Interpolator = Callable[[npt.NDArray[NPFloatingT], npt.NDArray[NPFloatingT]], ArrayFunc]

_slice = slice  # Alias for slice


class Representation(Protocol):
    """A format in which a GW signal is represented."""

    entries: npt.NDArray

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: lib.MethodT, *inputs, **kwargs
    ) -> Any:
        """Support arithmetic operations via numpy ufuncs."""
        ...


class Linspace:
    """Class for a uniform grid defined by linspace."""

    # FIXME confusing interface: semantics differ from np.linspace (step not stop)
    # Either change this or rename the class
    def __init__(self, start: float, step: float, num: int):
        """Initialize a uniform grid.

        Attention: this does not work like `numpy.linspace`.

        Parameters
        ----------
        start : float
            The start of the grid.
        step : float
            The step of the grid.
        num : int
            The number of points in the grid.

        Attributes
        ----------
        start : float
            The start of the grid.

        step : float
            The step of the grid.

        num : int
            The number of points in the grid.

        shape : tuple[int]
            The shape of the grid.

        stop : float
            The stop of the grid.
        """
        if num <= 0:
            raise ValueError("num must be at least 1")
        num = int(num)
        self.start = start
        self.step = step
        self.num = num
        self.shape = (num,)
        self.stop = start + step * (num - 1)

    def __eq__(self, other) -> bool:
        """Check equality of start, step, num, shape and stop."""
        if not isinstance(other, Linspace):
            return False
        a1 = self.start == other.start
        a2 = self.step == other.step
        a3 = self.num == other.num
        a4 = self.shape == other.shape
        a5 = self.stop == other.stop
        return a1 and a2 and a3 and a4 and a5

    def __len__(self) -> int:
        """Return the length of the grid."""
        return self.num

    def __repr__(self):
        """Return the string representation of the grid."""
        return f"Linspace(start={self.start}, step={self.step}, num={self.num})"

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool = True
    ) -> npt.NDArray[np.floating]:
        """Return the grid as a numpy array."""
        del copy  # Unused
        return self.start + self.step * np.arange(self.num, dtype=dtype)

    def __getitem__(self, slice: _slice) -> Self:
        """Return a subset of the grid."""
        return self.from_array(np.array(self)[slice])

    @classmethod
    def from_array(cls, array: npt.NDArray[np.floating]) -> Self:
        """Create a Linspace from a numpy array."""
        if len(array) < 2:
            raise ValueError(
                "Array must have at least two elements to create Linspace."
            )
        diff = np.diff(array)
        if not np.allclose(diff, diff[0], rtol=1e-8, atol=0):
            raise ValueError("Array must have uniform spacing to create Linspace.")
        return cls(start=array[0], step=diff[0], num=len(array))

    @classmethod
    def make(cls, array: Self | npt.NDArray[np.floating]):
        """Create a Linspace from a numpy array or return the input if already Linspace."""
        if isinstance(array, Linspace):
            return array
        return cls.from_array(array)

    @classmethod
    def get_step(cls, grid: "Linspace" | npt.NDArray[np.floating]) -> float:
        """Return the step of the uniform grid."""
        if isinstance(grid, Linspace):
            return grid.step
        return cls.from_array(grid).step


@dc.dataclass(slots=True, frozen=False)
class _Series(lib.mixins.NDArrayMixin):
    """A series of numbers on a grid."""

    grid: "Linspace" | npt.NDArray[np.floating]
    """ The grid of the representation. """
    entries: npt.NDArray[np.number]
    """ The values loaded on the grid. """

    def __init__(
        self,
        grid: "Linspace" | npt.NDArray[np.floating],
        entries: npt.NDArray[np.number],
    ):
        """Initialize the series.

        Parameters
        ----------
        grid :
            The grid of the representation.
        entries :
            The values loaded on the grid.
        """
        self.entries = entries
        self.grid = Linspace.make(grid)

    def is_consistent(self) -> bool:
        """Check if the grid and entries have the same shape."""
        return self.grid.shape == self.entries.shape

    def create_like(self, entries: npt.NDArray[np.number]):
        """Create a new series with the same grid as the current one."""
        return type(self)(grid=self.grid, entries=entries)

    @property
    def resolution(self) -> float:
        """The resolution of the grid."""
        return Linspace.get_step(self.grid)

    def _check_series(self, other: object, raise_error: bool = False) -> bool:
        """Check if the other series is of the same type."""
        if type(other) is type(self):
            return True
        if raise_error:
            if isinstance(other, _Series):
                raise TypeError(
                    f"Cannot operate between different series types: {type(self)} and {type(other)}."
                )
            raise TypeError(
                f"Cannot operate between series and non-series types: {type(self)} and {type(other)}."
            )
        return False

    def _check_grid(self, other: _Series, raise_error: bool = False) -> bool:
        """Check if the other series has the same grid."""
        flag = np.array_equal(self.grid, other.grid)
        if not raise_error:
            return flag
        if not flag:
            raise ValueError(
                "Series grid mismatch: expected {}, got {}".format(
                    self.grid, other.grid
                )
            )
        return flag

    def __array_ufunc__(self, ufunc: np.ufunc, method: lib.MethodT, *inputs, **kwargs):
        cls = type(self)

        def _unwrap(x):
            if self._check_series(x, raise_error=False):
                if self._check_grid(x, raise_error=True):
                    return x.entries
            return x

        if method == "reduce":
            return NotImplemented

        if method == "accumulate":
            return NotImplemented

        if method == "outer":
            return NotImplemented

        if method == "reduceat":
            if len(inputs) < 2:
                return NotImplemented
            assert inputs[0] is self, "What is going on?"
            indices = inputs[1]
            entries = ufunc.reduceat(self.entries, indices, *inputs[2:], **kwargs)
            grid = np.array(self.grid)[indices]
            return cls(grid, entries)

        if method == "at":
            if len(inputs) < 2:
                return NotImplemented
            assert inputs[0] is self, "What is going on?"
            indices = inputs[1]
            ufunc.at(self.entries, indices, *inputs[2:], **kwargs)
            return None

        if method == "__call__":
            unwrapped = [_unwrap(inp) for inp in inputs]
            out_arg = kwargs.get("out", None)
            if out_arg is None:
                return cls(
                    self.grid,
                    ufunc(*unwrapped, **kwargs),
                )
            out_unwrapped = [_unwrap(o) for o in out_arg]
            kwargs["out"] = tuple(out_unwrapped)
            ufunc(*unwrapped, **kwargs)
            return out_arg[0]

    def __getitem__(self, slice: _slice) -> Self:
        """Return the view of a subset of the series."""
        return self.get_subset(slice=slice, copy=False)

    def __setitem__(self, slice: _slice, value: Self) -> None:
        # NOTE this does NOT check whether the grids match
        self.entries[slice] = value.entries

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
        :meth:`.__iadd__`
        :meth:`.__add__`

        Note
        ----
        It is also possible to perform the addition using numpy syntax:

        ```python
        series[slice] + other
        ```

        where `slice` is the slice of the sub-grid to add on,
        and `other` is either a series defined on the sub-grid
        or a numeric value or array compatible with the sub-grid.
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
        :meth:`.__iadd__`
        :meth:`.add`
        :meth:`.__add__`

        Note
        ----
        It is also possible to perform the in-place addition using numpy syntax:

        ```python
        series[slice] += other
        ```

        where `slice` is the slice of the sub-grid to add on,
        and `other` is either a series defined on the sub-grid
        or a numeric value or array compatible with the sub-grid.
        """
        try:
            self.entries[slice] += other.entries  # type: ignore[misc]
        except ValueError as e:
            raise ValueError(
                "You may want to first embed the series instances to super-grids before "
                "adding them, if their grids are not compatible."
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
            if isinstance(other.grid, Linspace):
                start, stop = other.grid.start, other.grid.stop
            else:
                start, stop = other.grid[0], other.grid[-1]

            if len(self.grid) < len(other.grid):
                raise ValueError(
                    "In-place addition requires the series to add to "
                    "be a sub-grid of the current one. Expect `other.grid` "
                    "to be shorter than `self.grid`."
                )
            _slice = utils.get_subset_slice(np.array(self.grid), start, stop)
            return self.iadd(other, slice=_slice)
        return super().__iadd__(other)

    def _get_subset_slice(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: _slice | None = None,
    ) -> slice:
        """Return the subset as a new instance."""
        if interval is None:
            # Note that slice(None) is not None
            if slice is None:
                return _slice(None)
            # Otherwise we use the input slice
        else:
            if slice is not None:
                raise ValueError(
                    "Only one of `interval` and `slice` should be provided."
                )
            slice = utils.get_subset_slice(
                np.array(self.grid), interval[0], interval[1]
            )
        return slice

    def get_subset(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: _slice | None = None,
        copy: bool = True,
    ) -> Self:
        """Return the subset as a new instance."""
        slice = self._get_subset_slice(interval=interval, slice=slice)
        entries = self.entries[slice].copy() if copy else self.entries[slice]
        return type(self)(grid=self.grid[slice], entries=entries)

    def get_embedded(self, embedding_grid: Linspace | npt.NDArray[np.floating]) -> Self:
        """Return the series embedded in a new grid."""
        entries = utils.extend_to(np.array(embedding_grid))(
            np.array(self.grid), self.entries
        )
        return type(self)(grid=embedding_grid, entries=entries)

    def get_plotter(self):
        """Return the plotter for the series."""
        raise NotImplementedError("This method needs to be implemented in subclasses.")


@dc.dataclass(slots=True, frozen=False)
class FrequencySeries(_Series):
    """A series of numbers on a frequency grid."""

    @staticmethod
    def _angle(
        complex_numbers: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
    ) -> npt.NDArray[np.floating[NPTBitT]]:
        return np.unwrap(np.angle(complex_numbers), period=2 * np.pi)

    @staticmethod
    def _real(
        complex_numbers: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
    ) -> npt.NDArray[np.floating[NPTBitT]]:
        return np.real(complex_numbers)

    @staticmethod
    def _imag(
        complex_numbers: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
    ) -> npt.NDArray[np.floating[NPTBitT]]:
        return np.imag(complex_numbers)

    def angle(self):
        """Return the angle of the series."""
        return self.create_like(
            self._angle(self.entries)  # pyright: ignore[reportArgumentType]
        )

    @property
    def real(self):
        """Return the real part of the series."""
        return self.create_like(
            self._real(self.entries)  # pyright: ignore[reportArgumentType]
        )

    @property
    def imag(self):
        """Return the imaginary part of the series."""
        return self.create_like(
            self._imag(self.entries)  # pyright: ignore[reportArgumentType]
        )

    @property
    def frequencies(self):
        """The frequencies of the series. Alias for :attr:`.grid`."""
        return self.grid

    @property
    def df(self):
        """The frequency spacing. Alias for :attr:`.resolution`."""
        return self.resolution

    @property
    def frequencies(self):
        """The frequencies of the series. Alias for :attr:`.grid`."""
        return self.grid

    @property
    def df(self):
        """The frequency spacing. Alias for :attr:`.resolution`."""
        return self.resolution

    def irfft(
        self,
        time_grid: npt.NDArray[np.floating],
        tapering: tapering.Tapering | None = None,
    ):
        """Inverse real FFT of the series."""
        self_frequencies = np.array(self.frequencies)
        tapering_window = (
            tapering(self_frequencies)
            if tapering is not None
            else np.ones_like(self.frequencies)
        )
        dt: np.floating = time_grid[1] - time_grid[0]

        nyquist_dt = 1 / (2 * self_frequencies[-1])
        if dt < nyquist_dt and not np.isclose(dt, nyquist_dt):
            # FIXME spurious warning for odd, small n
            # probably related to the (n-1)/2 in the last frequency in rfftfreq
            warnings.warn("The time grid is denser than the Nyquist limit.")

        return TimeSeries(
            grid=time_grid,
            entries=np.fft.irfft(self.entries * tapering_window / dt, n=len(time_grid)),
        )

    def get_time_shifted(self, shift: float):
        """Shift the series in time."""
        return self * np.exp(-2j * np.pi * np.array(self.frequencies) * shift)

    def get_plotter(self) -> plotters.FSPlotter:
        """Return the plotter for the series."""
        from ..viz import plotters

        return plotters.FSPlotter(self)

    def to_WDM(
        self,
        /,
        *,
        Nf: int | None = None,
        Nt: int | None = None,
        nx: float = 4.0,
    ):
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
        fs = pywFS(data=self.entries / dt, freq=np.array(self.frequencies), t0=0.0)
        return WDM.from_pywWDM(_pyw_f2w(fs, Nf=Nf, Nt=Nt, nx=nx))


@dc.dataclass(slots=True, frozen=False)
class TimeSeries(_Series):
    """A series of numbers on a time grid."""

    @property
    def times(self):
        """The times of the series. Alias for :attr:`.grid`."""
        return self.grid

    @property
    def dt(self):
        """The time step. Alias for :attr:`.resolution`."""
        return self.resolution

    # TODO why is there no frequency grid parameter here? (symmetry with irfft)
    def rfft(self, tapering: tapering.Tapering | None = None):
        """Fast Fourier transform of the series."""
        self_times = np.array(self.times)
        tapering_window = (
            tapering(self_times) if tapering is not None else np.ones_like(self_times)
        )
        return FrequencySeries(
            grid=np.fft.rfftfreq(n=len(self.times), d=self.dt),
            entries=np.fft.rfft(self.entries * tapering_window * self.dt),
        )

    def get_plotter(self) -> plotters.TSPlotter:
        """Return the plotter for the series."""
        from ..viz import plotters

        return plotters.TSPlotter(self)

    # NOTE win cannot be a Tapering object. It's probably worth designing this
    # interface to be consistent with the rest of tlt and close to scipy's stft
    def stfft(self, win: npt.NDArray[np.floating], hop: int):
        """Short-time Fourier transform of the series."""
        SFT = scipy.signal.ShortTimeFFT(win=win, hop=hop, fs=1 / self.dt)
        times = SFT.t(len(self.entries)) + np.array(self.times)[0]
        freqs = SFT.f
        Sx = SFT.stft(self.entries * self.dt)
        return TimeFrequency(times=times, frequencies=freqs, entries=Sx)

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


@dc.dataclass(slots=True, frozen=False)
class Phasor(FrequencySeries):
    """Phasor representation. Subclass of :class:`.FrequencySeries`.

    A phasor is a couple of amplitude and phase that represent a complex number.
    This class encapsulates a sequence of phasors at different frequencies, which
    can be used to represent a waveform. This representation is useful for
    interpolating waveforms generated on a sparse grid of frequencies to a dense
    grid of frequencies.

    The input phases are expected to be smooth, without zigzags, so as the real
    and imaginary parts of the amplitudes. This is crucial for the interpolation
    to work properly.

    Note
    ----

    The so-called amplitude is itself complex number in general.
    The attribute :attr:`.amplitudes` gives the complex amplitudes,
    while the method :meth:`.abs` gives the absolute values of the amplitudes.
    Similarly, the attribute :attr:`.phases` gives the phases of the phasors
    as input, while the method :meth:`.angle` gives the angles of the full
    complex numbers as their argument.
    """

    phases: npt.NDArray[np.floating]

    @classmethod
    def make(
        cls,
        frequencies: npt.NDArray[NPFloatingT],
        amplitudes: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
        phases: npt.NDArray[np.floating[NPTBitT]],
    ) -> Phasor:
        """Create a phasor from amplitudes and phases."""
        entries = cls.phasor_to_cplx(amplitudes, phases)
        return cls(grid=frequencies, entries=entries, phases=phases)

    def _guard_binary_op(self, other):
        if isinstance(other, Phasor):
            raise ValueError("Binary operations between phasors are not supported.")

    def __setitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, slice: _slice, value: Self
    ) -> None:
        """Set the entries and phases of a subset of the phasor."""
        self.entries[slice] = value.entries
        self.phases[slice] = value.phases

    def create_like(self, entries: npt.NDArray[np.number]):
        """Create a new series with the same grid as the current one."""
        return type(self)(grid=self.grid, entries=entries, phases=self.phases)

    def get_subset(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: _slice | None = None,
        copy: bool = True,
    ) -> Self:
        """Return the subset as a new instance."""
        slice = self._get_subset_slice(interval=interval, slice=slice)
        entries = self.entries[slice].copy() if copy else self.entries[slice]
        phases = self.phases[slice].copy() if copy else self.phases[slice]
        return type(self)(grid=self.grid[slice], entries=entries, phases=phases)

    @staticmethod
    def reim_to_cplx(
        real_parts: npt.NDArray[np.floating[NPTBitT]],
        imag_parts: npt.NDArray[np.floating[NPTBitT]],
    ) -> npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]]:
        """Convert real and imaginary parts to complex numbers."""
        return real_parts + 1j * imag_parts

    @staticmethod
    def cplx_to_reim(
        complex_numbers: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
    ) -> tuple[npt.NDArray[np.floating[NPTBitT]], npt.NDArray[np.floating[NPTBitT]]]:
        """Convert complex numbers to real and imaginary parts."""
        return np.real(complex_numbers), np.imag(complex_numbers)

    @staticmethod
    def phasor_to_cplx(
        amplitudes: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
        phases: npt.NDArray[np.floating[NPTBitT]],
    ) -> npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]]:
        """Convert phasors to complex numbers."""
        return amplitudes * np.exp(1j * phases)

    @property
    def amplitudes(self):
        """The amplitudes of the phasors.

        Note
        ----

        The amplitudes are complex numbers in general.
        """
        return self.entries * np.exp(-1j * self.phases)

    def get_interpolated(
        self,
        frequencies: npt.NDArray[NPFloatingT],
        interpolator: Interpolator,
    ):
        """Get the phasors interpolated to the given frequencies."""
        self_freq = np.array(self.frequencies)
        amp_real, amp_imag = self.cplx_to_reim(
            self.amplitudes  # pyright: ignore[reportArgumentType]
        )
        amplitudes_real = interpolator(self_freq, amp_real)(frequencies)
        amplitudes_imag = interpolator(self_freq, amp_imag)(frequencies)
        amplitudes = self.reim_to_cplx(amplitudes_real, amplitudes_imag)
        phases = interpolator(self_freq, self.phases)(frequencies)
        return type(self).make(frequencies, amplitudes, phases)

    def to_frequency_series(self) -> FrequencySeries:
        """Get the :class:`.FrequencySeries` representation of the waveform."""
        return FrequencySeries(
            self.frequencies,
            self.entries,
        )

    def get_plotter(self) -> plotters.PhasorPlotter:
        """Return the plotter for the phasor."""
        from ..viz import plotters

        return plotters.PhasorPlotter(self)


@dc.dataclass(slots=True, frozen=False)
class TimeFrequency(lib.mixins.NDArrayMixin):
    """Time-frequency representation.

    The entries are stored in a 2D array, where the first axis corresponds to
    the frequencies and the second axis corresponds to the times. The time and
    frequency grids are stored as 1D arrays.

    The time-frequency representation could contain the spectrogram or the
    scalogram of a signal, depending on how the entries are computed.
    """

    times: "Linspace" | npt.NDArray[np.floating]
    """ The time grid of the time-frequency representation. """

    frequencies: "Linspace" | npt.NDArray[np.floating]
    """ The frequency grid of the time-frequency representation. """

    entries: npt.NDArray
    """ The data values in the time-frequency representation. """

    @property
    def dT(self) -> float:
        """Time resolution (ΔT) of the time-frequency grid."""
        return Linspace.get_step(self.times)

    @property
    def dF(self) -> float:
        """Frequency resolution (ΔF) of the time-frequency grid."""
        return Linspace.get_step(self.frequencies)

    # TODO for consistency with _Series: receive slices, receive copy bool arg
    def get_subset(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
    ) -> Self:
        """Return the subset as a new instance."""
        if time_interval is not None:
            time_slice = utils.get_subset_slice(
                np.array(self.times), time_interval[0], time_interval[1]
            )
        else:
            time_slice = _slice(None)
        if freq_interval is not None:
            freq_slice = utils.get_subset_slice(
                np.array(self.frequencies), freq_interval[0], freq_interval[1]
            )
        else:
            freq_slice = _slice(None)
        return type(self)(
            times=self.times[time_slice],
            frequencies=self.frequencies[freq_slice],
            entries=self.entries[freq_slice, time_slice],
        )

    def create_like(self, entries: npt.NDArray[np.number]):
        """Create a new series with different entries but the same grid."""
        return type(self)(
            times=self.times,
            frequencies=self.frequencies,
            entries=entries,
        )

    def get_plotter(self) -> plotters.TimeFrequencyPlotter:
        """Return the plotter for the time-frequency representation."""
        from ..viz import plotters

        return plotters.TimeFrequencyPlotter(self)

    def _check_representation(self, other: object, raise_error: bool = False) -> bool:
        """Check if the other representation is of the same type."""
        if type(other) is type(self):
            return True
        if raise_error:
            if isinstance(other, TimeFrequency):
                raise TypeError(
                    "Cannot operate between different time-frequency"
                    f"representation types: {type(self)} and {type(other)}."
                )
            raise TypeError(
                "Cannot operate between time-frequency and non time-frequency"
                f"representation types: {type(self)} and {type(other)}."
            )
        return False

    def _check_grid(self, other: TimeFrequency, raise_error: bool = False) -> bool:
        """Check if the other representation has the same grid."""
        time_flag = np.array_equal(self.times, other.times)
        freq_flag = np.array_equal(self.frequencies, other.frequencies)
        flag = time_flag and freq_flag
        if not raise_error:
            return flag
        if not time_flag:
            raise ValueError(
                "Time-frequency representation time grid mismatch: expected {}, got {}".format(
                    self.times, other.times
                )
            )
        if not freq_flag:
            raise ValueError(
                "Time-frequency representation frequency grid mismatch: expected {}, got {}".format(
                    self.frequencies, other.frequencies
                )
            )
        return flag

    def __array_ufunc__(self, ufunc: np.ufunc, method: lib.MethodT, *inputs, **kwargs):
        """Support arithmetic operations via numpy ufuncs."""
        cls = type(self)

        def _unwrap(x):
            if self._check_representation(x, raise_error=False):
                if self._check_grid(x, raise_error=True):
                    return x.entries
            return x

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
            unwrapped = [_unwrap(inp) for inp in inputs]
            out_arg = kwargs.get("out", None)
            if out_arg is None:
                return cls(
                    self.times,
                    self.frequencies,
                    ufunc(*unwrapped, **kwargs),
                )
            out_unwrapped = [_unwrap(o) for o in out_arg]
            kwargs["out"] = tuple(out_unwrapped)
            ufunc(*unwrapped, **kwargs)
            return out_arg[0]


@dc.dataclass(slots=True, frozen=False)
class WDM(TimeFrequency):
    """
    Wilson-Daubechies-Meyer (WDM) time-frequency representation.

    This represents data using an evenly-spaced 2D grid in the
    time-frequency plane with shape (Nf, Nt). Each "pixel" has size
    ΔF ΔT = 1/2. The times range approximately from 0 to the final
    observation time, while the frequencies range from 0 to the
    Nyquist limit (half the sampling rate).

    Currently, transformations to/from FrequencySeries are allowed,
    but only for full series --- all frequencies and all times.
    See :meth:`.from_freqseries` and :meth:`to_freqseries`.

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
        Array of evenly-spaced times with separation ΔT and size `Nt`.

    frequencies: real 1D array
        Array of evenly-spaced frequencies with separation ΔF and size `Nf`.

    entries: real 2D array
        Array of data entries, with shape `(Nf, Nt)`.
    """

    times: "Linspace"
    frequencies: "Linspace"

    def __init__(
        self,
        times: npt.NDArray[np.floating] | Linspace,
        frequencies: npt.NDArray[np.floating] | Linspace,
        entries: npt.NDArray[np.floating],
    ):
        self.times = Linspace.make(times)
        self.frequencies = Linspace.make(frequencies)
        self.entries = entries

    def is_critically_sampled(self):
        """Return True if :attr:`.dT` * :attr:`.dF` = 1/2."""
        # I don't like how this method is implemented,
        # but I don't see a better way for now.
        return np.isclose(self.dT * self.dF, 1 / 2)

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
        return len(self.times) * self.times.step

    @property
    def sample_interval(self) -> float:
        """
        Time resolution of a TimeSeries corresponding to this WDM.

        Smaller than the wavelet time bin :attr:`.dT`.
        """
        return self.duration / self.ND

    dt = sample_interval
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
        return self.entries.shape

    @property
    def sample_rate(self) -> float:
        """Sampling rate."""
        # We can verify that this is twice (self.frequencies.stop + self.frequencies.step)
        # which is the true maximum frequency in the WDM representation, again due to
        # the special convention in this class that a grid of N points has N bins.
        return 1 / self.sample_interval

    fs = sample_rate
    """Alias for :attr:`.sample_rate`."""

    def to_freqseries(
        self, *, nx: float = 4.0, mask: npt.NDArray[np.bool] | None = None
    ) -> FrequencySeries:
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
        pywfs = _pyw_w2f(pywwv, self.dt, nx) 
        freqs = pywfs.freq
        entries = pywfs.data * pywfs.dt
        # To see if we keep or not
        # https://gitlab.in2p3.fr/lisa-apc/typed-lisa-toolkit/-/merge_requests/4#note_576666
        if mask is not None:
            freqs = np.ma.masked_where(mask, freqs)
            entries = np.ma.masked_where(mask, entries)
        return FrequencySeries(freqs, entries)

    @property
    def nyquist(self) -> float:
        """Nyquist frequency (half the sampling rate)."""
        # I don't like this property name
        return self.sample_rate / 2

    @classmethod
    def from_pywWDM(cls, pywwv: pywWDM, /) -> Self:
        """Convert a pywWDM object to a WDM."""
        entries = pywwv.data
        times = Linspace(pywwv.time[0], pywwv.time[1] - pywwv.time[0], len(pywwv.time))
        frequencies = Linspace(
            pywwv.freq[0], pywwv.freq[1] - pywwv.freq[0], len(pywwv.freq)
        )
        return cls(times=times, frequencies=frequencies, entries=entries)

    def _to_pywWDM(self) -> pywWDM:
        """Convert self to a pywWDM object."""
        return pywWDM(
            data=self.entries,
            time=np.array(self.times),
            freq=np.array(self.frequencies),
        )
