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
the values for the time-frequency matrices (:class:`.TimeFrequency`).
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
.. autoprotocol:: Tapering

Entities
--------
.. autoclass:: Representation
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
   :special-members: __mul__, __rmul__, __add__, __sub__, __truediv__, __rtruediv__, __neg__

.. autoclass:: TimeSeries
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
   :special-members: __add__

.. autoclass:: FrequencySeries
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
   :special-members: __add__

.. autoclass:: Phasor
   :members:
   :member-order: groupwise
   :undoc-members:
   :inherited-members:
   :special-members: __add__

.. autoclass:: TimeFrequency
    :members:
    :member-order: groupwise
    :undoc-members:
    :inherited-members:

"""

from __future__ import annotations
import abc
from collections.abc import Callable
import dataclasses as dc
import logging
from typing import TypeVar, Generic, Self, Protocol, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.signal  # type: ignore[import-untyped]

from .. import utils

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

ArrayFunc = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
Interpolator = Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], ArrayFunc]

_slice = slice  # Alias for slice


class Tapering(Protocol):
    """Protocol for tapering functions."""

    def __call__(self, __array: npt.NDArray[np.number]) -> npt.NDArray[np.floating]:
        """Return the tapering window to apply on the array."""


class Representation(Generic[NPFloatingT, NPNumberT_co]):
    """A format in which detector data are displayed.

    This is the base class for all representations, which
    is abstract and cannot be instantiated directly.
    """

    entries: npt.NDArray[NPNumberT_co]
    """ The data values. """

    @abc.abstractmethod
    def create_like(self, entries: npt.NDArray[np.number]) -> Self:
        """Create a new series with different entries but the same grid."""

    def _guard_binary_op(self, other):
        del other

    def __mul_num__(self, other: Numeric) -> Self:
        """Multiply a representation by a number or a numeric array."""
        return self.create_like(self.entries * other)

    @abc.abstractmethod
    def __mul__(self, other: Self | Numeric) -> Self:
        """Multiply a representation by another representation, a number or a numeric array."""

    def __rmul__(self, other: Numeric):
        """Multiply a number or a numeric array by a representation."""
        return self.__mul_num__(other)

    @abc.abstractmethod
    def __truediv__(self, other: Self | Numeric) -> Self:
        """Divide a representation by another representation, a number or a numeric array."""

    def __rtruediv__(self, other: Numeric):
        """Divide a number or a numeric array by a series."""
        return self.create_like(other / self.entries)

    def __add__(self, other: Self) -> Self:
        """Add two representations."""
        return self.create_like(self.entries + other.entries)

    def __sub__(self, other: Self) -> Self:
        """Subtract two representations."""
        return self + (-other)

    def __neg__(self) -> Self:
        """Negate a representation."""
        return self.create_like(-self.entries)


@dc.dataclass(slots=True, frozen=True)
class _Series(
    Representation[NPFloatingT, NPNumberT_co], Generic[NPFloatingT, NPNumberT_co]
):
    """A series of numbers on a grid."""

    grid: npt.NDArray[NPFloatingT]
    """ The grid of the representation. """

    entries: npt.NDArray[NPNumberT_co]
    """ The values loaded on the grid. """

    @property
    def is_consistent(self) -> bool:
        """Check if the grid and entries have the same shape."""
        return self.grid.shape == self.entries.shape

    def create_like(self, entries: npt.NDArray[np.number]):
        """Create a new series with the same grid as the current one."""
        # We ignore the type check below since we know that the new entries
        # could have a different data type than the current entries.
        return type(self)(grid=self.grid, entries=entries)  # type: ignore[arg-type]

    @property
    def has_even_spacing(self) -> bool:
        """Check if the grid has even spacing."""
        return np.allclose(np.diff(self.grid), self.resolution)

    @property
    def resolution(self) -> NPFloatingT:
        """The resolution of the grid.

        Note
        ----
        No check is performed to ensure that the grid are evenly spaced.
        Just return the difference between the first two grid points.
        """
        return self.grid[1] - self.grid[0]

    def __mul_series__(self, other: _Series[NPFloatingT, np.number]):
        """Multiply two series."""
        self._guard_binary_op(other)
        return self.create_like(self.entries * other.entries)

    def __mul__(self, other: _Series[NPFloatingT, np.number] | Numeric):
        """Multiply a series by another series, a number or a numeric array."""
        self._guard_binary_op(other)
        if isinstance(other, _Series):
            return self.__mul_series__(other)
        return self.__mul_num__(other)

    def __truediv__(self, other: _Series[NPFloatingT, np.number] | Numeric):
        """Divide a series by another series, a number or a numeric array."""
        self._guard_binary_op(other)
        if isinstance(other, _Series):
            return self.create_like(self.entries / other.entries)
        return self.create_like(self.entries / other)

    def __add__(self, other: Self) -> Self:
        """Add two series.

        Note
        ----
        This method allows adding two series with different grids, as long as
        one grid is a subset of the other grid. The resulting series will have
        the grid of the longer series.
        """
        self._guard_binary_op(other)
        if len(self.grid) < len(other.grid):
            return other + self
        _slice = utils.get_subset_slice(self.grid, other.grid[0], other.grid[-1])
        if np.array_equal(self.grid[_slice], other.grid):
            _entries = self.entries.copy()
            _entries[_slice] += other.entries
            return self.create_like(_entries)
        raise ValueError("The grids of the two series are not compatible.")

    def exp(self) -> Self:
        """Return the exponential of the series."""
        return self.create_like(np.exp(self.entries))

    def sqrt(self) -> Self:
        """Return the square root of the series."""
        return self.create_like(np.sqrt(self.entries))

    def _get_subset_slice(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: slice | None = None,
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
            slice = utils.get_subset_slice(self.grid, interval[0], interval[1])
        return slice

    def get_subset(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: slice | None = None,
    ) -> Self:
        """Return the subset as a new instance."""
        slice = self._get_subset_slice(interval=interval, slice=slice)
        return type(self)(grid=self.grid[slice], entries=self.entries[slice])

    def get_embedded(self, embedding_grid: npt.NDArray[NPFloatingT]) -> Self:
        """Return the series embedded in a new grid."""
        entries = utils.extend_to(embedding_grid)(self.grid, self.entries)
        return type(self)(grid=embedding_grid, entries=entries)

    def get_plotter(self):
        """Return the plotter for the series."""
        raise NotImplementedError("This method needs to be implemented in subclasses.")


@dc.dataclass(slots=True, frozen=True)
class FrequencySeries(
    _Series[NPFloatingT, NPNumberT_co], Generic[NPFloatingT, NPNumberT_co]
):
    """A series of numbers on a frequency grid. Subclass of :class:`.Representation`."""

    @staticmethod
    def _abs(
        complex_numbers: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
    ) -> npt.NDArray[np.floating[NPTBitT]]:
        return np.abs(complex_numbers)

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

    @property
    def frequencies(self) -> npt.NDArray[NPFloatingT]:
        """The frequencies of the series. Alias for :attr:`.grid`."""
        return self.grid

    @property
    def df(self) -> NPFloatingT:
        """The frequency spacing. Alias for :attr:`.resolution`."""
        return self.resolution

    def conj(self) -> Self:
        """Return the complex conjugate of the series."""
        return self.create_like(self.entries.conj())

    def angle(self):
        """Return the angle of the series."""
        return self.create_like(self._angle(self.entries))

    def abs(self):
        """Return the absolute value of the series."""
        return self.create_like(self._abs(self.entries))

    @property
    def real(self):
        """Return the real part of the series."""
        return self.create_like(self._real(self.entries))

    @property
    def imag(self):
        """Return the imaginary part of the series."""
        return self.create_like(self._imag(self.entries))

    def irfft(
        self, time_grid: npt.NDArray[np.floating], tapering: Tapering | None = None
    ):
        """Inverse real FFT of the series."""
        tapering_window = (
            tapering(self.frequencies)
            if tapering is not None
            else np.ones_like(self.frequencies)
        )
        return TimeSeries(
            grid=time_grid,
            entries=np.fft.irfft(self.entries * tapering_window, n=len(time_grid)),
        )

    def get_time_shifted(self, shift: float):
        """Shift the series in time."""
        return self.create_like(
            self.entries * np.exp(-2j * np.pi * self.frequencies * shift)
        )

    def get_plotter(self) -> plotters.FSPlotter:
        """Return the plotter for the series."""
        from ..viz import plotters

        return plotters.FSPlotter(self)


@dc.dataclass(slots=True, frozen=True)
class TimeSeries(
    _Series[NPFloatingT, NPNumberT_co], Generic[NPFloatingT, NPNumberT_co]
):
    """A series of numbers on a time grid. Subclass of :class:`.Representation`."""

    @property
    def times(self) -> npt.NDArray[NPFloatingT]:
        """The times of the series. Alias for :attr:`.grid`."""
        return self.grid

    @property
    def dt(self) -> NPFloatingT:
        """The time step. Alias for :attr:`.resolution`."""
        return self.resolution

    def rfft(self, tapering: Tapering | None = None):
        """Fast Fourier transform of the series."""
        tapering_window = (
            tapering(self.times) if tapering is not None else np.ones_like(self.times)
        )
        return FrequencySeries(
            grid=np.fft.rfftfreq(n=len(self.times), d=self.dt),
            entries=np.fft.rfft(self.entries * tapering_window),
        )

    def get_plotter(self) -> plotters.TSPlotter:
        """Return the plotter for the series."""
        from ..viz import plotters

        return plotters.TSPlotter(self)

    def stfft(self, win: npt.NDArray[np.floating], hop: int):
        """Short-time Fourier transform of the series."""
        SFT = scipy.signal.ShortTimeFFT(win=win, hop=hop, fs=1 / self.dt)
        times = SFT.t(len(self.entries)) + self.times[0]
        freqs = SFT.f
        Sx = SFT.stft(self.entries)
        return TimeFrequency(times=times, frequencies=freqs, entries=Sx)


@dc.dataclass(slots=True, frozen=True)
class Phasor(
    FrequencySeries[NPFloatingT, np.complexfloating[NPTBitT, NPTBitT]],
    Generic[NPFloatingT, NPTBitT],
):
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

    phases: npt.NDArray[np.floating[NPTBitT]]

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

    def create_like(self, entries: npt.NDArray[np.number]):
        """Create a new series with the same grid as the current one."""
        # We ignore the type check below since we know that the new entries
        # could have a different data type than the current entries.
        return type(self)(grid=self.grid, entries=entries, phases=self.phases)  # type: ignore[arg-type]

    def get_subset(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: slice | None = None,
    ) -> Self:
        """Return the subset as a new instance."""
        slice = self._get_subset_slice(interval=interval, slice=slice)
        return type(self)(
            grid=self.grid[slice],
            entries=self.entries[slice],
            phases=self.phases[slice],
        )

    def get_embedded(self, embedding_grid: npt.NDArray[NPFloatingT]) -> Self:
        """Return the series embedded in a new grid."""
        entries = utils.extend_to(embedding_grid)(self.grid, self.entries)
        phases = utils.extend_to(embedding_grid)(self.grid, self.phases)
        return type(self)(grid=embedding_grid, entries=entries, phases=phases)

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
        amp_real, amp_imag = self.cplx_to_reim(self.amplitudes)
        amplitudes_real = interpolator(self.frequencies, amp_real)(frequencies)
        amplitudes_imag = interpolator(self.frequencies, amp_imag)(frequencies)
        amplitudes = self.reim_to_cplx(amplitudes_real, amplitudes_imag)
        phases = interpolator(self.frequencies, self.phases)(frequencies)
        return type(self).make(frequencies, amplitudes, phases)

    def to_frequency_series(
        self,
    ) -> FrequencySeries[NPFloatingT, np.complexfloating[NPTBitT, NPTBitT]]:
        """Get the :class:`.FrequencySeries` representation of the waveform."""
        return FrequencySeries(
            self.frequencies,
            self.entries,
        )

    def get_plotter(self) -> plotters.PhasorPlotter:
        """Return the plotter for the phasor."""
        from ..viz import plotters

        return plotters.PhasorPlotter(self)


@dc.dataclass(slots=True, frozen=True)
class TimeFrequency(
    Representation[NPFloatingT, NPNumberT_co], Generic[NPFloatingT, NPNumberT_co]
):
    """Time-frequency representation.

    The entries are stored in a 2D array, where the first axis corresponds to
    the frequencies and the second axis corresponds to the times. The time and
    frequency grids are stored as 1D arrays.

    The time-frequency representation could contain the spectrogram or the
    scalogram of a signal, depending on how the entries are computed.
    """

    times: npt.NDArray[NPFloatingT]
    """ The time grid of the time-frequency representation. """

    frequencies: npt.NDArray[NPFloatingT]
    """ The frequency grid of the time-frequency representation. """

    entries: npt.NDArray[NPNumberT_co]
    """ The data values in the time-frequency representation. """

    @property
    def dt(self) -> NPFloatingT:
        """The time step."""
        return self.times[1] - self.times[0]

    @property
    def df(self) -> NPFloatingT:
        """The frequency spacing."""
        return self.frequencies[1] - self.frequencies[0]

    def get_subset(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
    ) -> Self:
        """Return the subset as a new instance."""
        if time_interval is not None:
            time_slice = utils.get_subset_slice(
                self.times, time_interval[0], time_interval[1]
            )
        else:
            time_slice = _slice(None)
        if freq_interval is not None:
            freq_slice = utils.get_subset_slice(
                self.frequencies, freq_interval[0], freq_interval[1]
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
            entries=entries,  # type: ignore[arg-type]
        )

    def __mul_tf__(self, other: TimeFrequency[NPFloatingT, np.number]):
        """Multiply two time-frequency representations."""
        self._guard_binary_op(other)
        return self.create_like(self.entries * other.entries)

    def __mul__(self, other: TimeFrequency[NPFloatingT, np.number] | Numeric):
        """Multiply a series by another time-frequency representations or a number."""
        self._guard_binary_op(other)
        if isinstance(other, TimeFrequency):
            return self.__mul_tf__(other)
        return self.__mul_num__(other)

    def __truediv__(self, other: TimeFrequency[NPFloatingT, np.number] | Numeric):  # type: ignore[override]
        # We violate the Liskov Substitution Principle on purpose here.
        """Divide a series by another series or a number."""
        self._guard_binary_op(other)
        if isinstance(other, TimeFrequency):
            return self.create_like(self.entries / other.entries)
        return self.create_like(self.entries / other)

    def get_plotter(self) -> plotters.TimeFrequencyPlotter:
        """Return the plotter for the time-frequency representation."""
        from ..viz import plotters

        return plotters.TimeFrequencyPlotter(self)
