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
(time or frequency) or in both domains. In the first case, there is
:class:`.TimeSeries`, :class:`.FrequencySeries` and :class:`.Phasor`.
The second case consists of time-frequency matrices  (spectrograms,
scalograms, etc.). In both cases, we need a grid (1D or 2D) and values
loaded on the grid. We model these representations as immutable data
classes that support arithmetic operations between them and with numeric
values or arrays. We implement generic classes that accept a certain
:class:`numpy.dtype` as type parameter. This allows us to specify and
keep track of the data type of the values in the representation.

.. currentmodule:: typed_lisa_toolkit.containers.representations

Types
-----
.. autoclass:: Numeric
.. autoclass:: NPNumberT_co
.. autoclass:: NPFloatingT
.. autoclass:: NPTBitT
.. autoclass:: ArrayFunc
.. autoclass:: TaperT

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
"""

from __future__ import annotations
import abc
from collections.abc import Callable
import dataclasses as dc
import logging
from typing import TypeVar, Generic, Self

import numpy as np
import numpy.typing as npt

from .. import utils

log = logging.getLogger(__name__)

PyNum = int | float | complex  # Union[int, float, complex]
Numeric = PyNum | np.number | npt.NDArray[np.number]

NPNumberT_co = TypeVar("NPNumberT_co", bound=np.number, covariant=True)
"""Covariant numpy number data type."""

NPFloatingT = TypeVar("NPFloatingT", bound=np.floating)
"""Numpy floating data type."""

NPTBitT = TypeVar("NPTBitT", bound=npt.NBitBase)
"""Numpy bit data type."""

ArrayFunc = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
TaperT = ArrayFunc


@dc.dataclass(slots=True, frozen=True)
class Representation(Generic[NPFloatingT, NPNumberT_co]):
    """A format in which detector data are displayed.

    This is the base class for all representations.
    """

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
        return type(self)(grid=self.grid, entries=entries)  # type: ignore

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
    def __truediv__(
        self, other: Representation[NPFloatingT, np.number] | Numeric
    ) -> Self:
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
        return self.create_like(self.entries * other.entries)

    def __mul__(self, other: _Series[NPFloatingT, np.number] | Numeric):  # type: ignore
        # We violate the Liskov Substitution Principle on purpose here.
        """Multiply a series by another series, a number or a numeric array."""
        if isinstance(other, _Series):
            return self.__mul_series__(other)
        return self.__mul_num__(other)

    def __truediv__(self, other: _Series[NPFloatingT, np.number] | Numeric):  # type: ignore
        # We violate the Liskov Substitution Principle on purpose here.
        """Divide a series by another series, a number or a numeric array."""
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

    def get_subset(self, *, interval: tuple[float, float] | None = None) -> Self:
        """Return the subset as a new instance."""
        if interval is None:
            return self
        mask = utils.get_subset_slice(self.grid, interval[0], interval[1])
        return type(self)(grid=self.grid[mask], entries=self.entries[mask])


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
        return np.angle(complex_numbers)

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
        self, time_grid: npt.NDArray[np.floating], tapering: TaperT | None = None
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

    def to_phasor(self) -> Phasor:
        """Get the :class:`.Phasor` representation of the waveform."""
        return Phasor(self.frequencies, self.entries)


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

    def rfft(self, tapering: TaperT | None = None):
        """Fast Fourier transform of the series."""
        tapering_window = (
            tapering(self.times) if tapering is not None else np.ones_like(self.times)
        )
        return FrequencySeries(
            grid=np.fft.rfftfreq(n=len(self.times), d=self.dt),
            entries=np.fft.rfft(self.entries * tapering_window),
        )


@dc.dataclass(slots=True, frozen=True)
class Phasor(FrequencySeries[NPFloatingT, NPNumberT_co]):
    """Phasor representation. Subclass of :class:`.FrequencySeries`.

    A phasor is a couple of amplitude and phase that represent a complex number.
    This class encapsulates a sequence of phasors at different frequencies, which
    can be used to represent a waveform. This representation is useful for
    interpolating waveforms generated on a sparse grid of frequencies to a dense
    grid of frequencies.
    """

    @classmethod
    def make(
        cls,
        frequencies: npt.NDArray[NPFloatingT],
        amplitudes: npt.NDArray[np.floating],
        phases: npt.NDArray[np.floating],
    ) -> Self:
        """Create a phasor sequence."""
        cplx = cls.phasor_to_cplx(amplitudes, phases)
        return cls(frequencies, cplx)  # type: ignore
        # The returned type depends on the type of the input arrays.

    @staticmethod
    def reim_to_cplx(
        real_parts: npt.NDArray[np.floating[NPTBitT]],
        imag_parts: npt.NDArray[np.floating[NPTBitT]],
    ) -> npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]]:
        """Convert real and imaginary parts to complex numbers."""
        return real_parts + 1j * imag_parts

    @staticmethod
    def cplx_to_phasor(
        complex_numbers: npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]],
    ) -> tuple[npt.NDArray[np.floating[NPTBitT]], npt.NDArray[np.floating[NPTBitT]]]:
        """Convert complex numbers to phasors."""
        return Phasor._abs(complex_numbers), Phasor._angle(complex_numbers)

    @staticmethod
    def phasor_to_cplx(
        amplitudes: npt.NDArray[np.floating[NPTBitT]],
        phases: npt.NDArray[np.floating[NPTBitT]],
    ) -> npt.NDArray[np.complexfloating[NPTBitT, NPTBitT]]:
        """Convert phasors to complex numbers."""
        return amplitudes * np.exp(1j * phases)

    @staticmethod
    def reim_to_phasor(
        real_parts: npt.NDArray[np.floating[NPTBitT]],
        imag_parts: npt.NDArray[np.floating[NPTBitT]],
    ) -> tuple[npt.NDArray[np.floating[NPTBitT]], npt.NDArray[np.floating[NPTBitT]]]:
        """Convert real and imaginary parts to phasors."""
        return Phasor.cplx_to_phasor(Phasor.reim_to_cplx(real_parts, imag_parts))

    @property
    def amplitudes(self):
        """The amplitudes of the phasors."""
        return self._abs(self.entries)

    @property
    def phases(self):
        """The phases of the phasors."""
        return self._angle(self.entries)

    def __repr__(self) -> str:
        """Return the string representation of the phasor."""
        return f"{self.__class__.__name__}(frequencies={self.frequencies}, amplitudes={self.amplitudes}, phases={self.phases})"

    def get_interpolated(
        self,
        frequencies: npt.NDArray[NPFloatingT],
        interpolator: Callable[..., ArrayFunc],
    ) -> Self:
        """Get the phasors interpolated to the given frequencies."""
        amplitudes = interpolator(self.frequencies, self.amplitudes)(frequencies)
        phases = interpolator(self.frequencies, self.phases)(frequencies)
        return self.make(frequencies, amplitudes, phases)

    def to_freq_series(self) -> FrequencySeries[NPFloatingT, np.complexfloating]:
        """Get the :class:`.FrequencySeries` representation of the waveform."""
        return FrequencySeries(
            self.frequencies,
            self.phasor_to_cplx(self.amplitudes, self.phases),
        )
