"""
Module for series.

We model a series as a pair of numpy arrays, one for the grid
and one for the signal loaded on the grid. We implement
:class:`.Series` and its subclasses (:class:`.FrequencySeries` and
:class:`.TimeSeries`) to be immutable and to support arithmetic
operations between two series or between a series and a numeric
value or array.

:class:`.Series` and its subclasses are generic classes that
accept a certain :class:`numpy.dtype` as type parameter. This
allows us to specify and keep track of the data type of the
signal in the series.
"""

from __future__ import annotations
import logging
from collections.abc import Callable
import dataclasses as dc
from typing import TypeVar, Generic, Self


import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

PyNum = int | float | complex  # Union[int, float, complex]
Numeric = PyNum | np.number | npt.NDArray[np.number]

NPDT_co = TypeVar("NPDT_co", bound=np.number, covariant=True)
"""Covariant numpy data type."""

NPDTb_co = TypeVar("NPDTb_co", bound=np.number, covariant=True)  # Numpy number type bis
"""Covariant numpy data type (bis)."""

TaperT = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]


@dc.dataclass(slots=True, frozen=True)
class Series(Generic[NPDT_co]):
    """A series of numbers on a grid."""

    grid: npt.NDArray[np.floating]
    """ The grid of the series. """

    signal: npt.NDArray[NPDT_co]
    """ The signal of the series. """

    @property
    def is_consistent(self) -> bool:
        """Check if the grid and signal have the same length."""
        return len(self.grid) == len(self.signal)

    @property
    def has_even_spacing(self) -> bool:
        """Check if the grid has even spacing."""
        return np.allclose(np.diff(self.grid), self.grid[1] - self.grid[0])

    def create_like(self, signal: npt.NDArray[NPDTb_co]):
        """Create a new series with the same grid as the current one."""
        # We ignore the type check below since we know that the new signal
        # could have a different data type than the current signal.
        return type(self)(grid=self.grid, signal=signal)  # type: ignore

    def __mul_series__(self, other: Series[np.number]):
        """Multiply two series."""
        return self.create_like(self.signal * other.signal)

    def __mul_num__(self, other: Numeric):
        """Multiply a series by a number or a numeric array."""
        return self.create_like(self.signal * other)

    def __mul__(self, other: Series[np.number] | Numeric):
        """Multiply a series by another series, a number or a numeric array."""
        if isinstance(other, Series):
            return self.__mul_series__(other)
        return self.__mul_num__(other)

    def __rmul__(self, other: Numeric):
        """Multiply a number or a numeric array by a series."""
        return self.__mul_num__(other)

    def __truediv__(self, other: Series[np.number] | Numeric):
        """Divide a series by another series, a number or a numeric array."""
        if isinstance(other, Series):
            return self.create_like(self.signal / other.signal)
        return self.create_like(self.signal / other)

    def __rtruediv__(self, other: Numeric):
        """Divide a number or a numeric array by a series."""
        return self.create_like(other / self.signal)

    def __add__(self, other: Self):
        """Add two series."""
        return self.create_like(self.signal + other.signal)

    def __sub__(self, other: Self):
        """Subtract two series."""
        return self.create_like(self.signal - other.signal)

    def __neg__(self):
        """Negate a series."""
        return self.create_like(-self.signal)

    def exp(self):
        """Exponential of the series."""
        return self.create_like(np.exp(self.signal))

    def sqrt(self):
        """Square root of the series."""
        return self.create_like(np.sqrt(self.signal))


@dc.dataclass(slots=True, frozen=True)
class FrequencySeries(Series[NPDT_co], Generic[NPDT_co]):
    """A series of numbers on a frequency grid."""

    @property
    def frequencies(self) -> npt.NDArray[np.floating]:
        """The frequencies of the series."""
        return self.grid

    @property
    def df(self) -> float:
        """The frequency spacing.

        Note
        ----
        No check is performed to ensure that the frequencies are evenly spaced.
        Just return the difference between the first two frequencies.
        """
        return self.grid[1] - self.grid[0]

    def conj(self):
        """Return the complex conjugate of the series."""
        return self.create_like(self.signal.conj())

    def angle(self):
        """Return the angle of the series."""
        return self.create_like(np.angle(self.signal))

    def abs(self):
        """Return the absolute value of the series."""
        return self.create_like(np.abs(self.signal))

    def real(self):
        """Return the real part of the series."""
        return self.create_like(np.real(self.signal))

    def imag(self):
        """Return the imaginary part of the series."""
        return self.create_like(np.imag(self.signal))

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
            signal=np.fft.irfft(self.signal * tapering_window, n=len(time_grid)),
        )

    def get_time_shifted(self, shift: float):
        """Shift the series in time."""
        return self.create_like(
            self.signal * np.exp(-2j * np.pi * self.frequencies * shift)
        )


@dc.dataclass(slots=True, frozen=True)
class TimeSeries(Series[NPDT_co], Generic[NPDT_co]):
    """A series of numbers on a time grid."""

    @property
    def times(self) -> npt.NDArray[np.floating]:
        """The times of the series."""
        return self.grid

    @property
    def dt(self) -> float:
        """The time step.

        Note
        ----
        No check is performed to ensure that the times are evenly spaced.
        Just return the difference between the first two times.
        """
        return self.grid[1] - self.grid[0]

    def rfft(self, tapering: TaperT | None = None):
        """Fast Fourier transform of the series."""
        tapering_window = (
            tapering(self.times) if tapering is not None else np.ones_like(self.times)
        )
        return FrequencySeries(
            grid=np.fft.rfftfreq(n=len(self.times), d=self.dt),
            signal=np.fft.rfft(self.signal * tapering_window),
        )
