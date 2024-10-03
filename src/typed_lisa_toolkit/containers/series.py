"""
Module for series of numbers on a grid.

The classes are designed to be immutable and to support arithmetic operations
between series and numbers or other series.
"""

from __future__ import annotations
import logging
import dataclasses as dc
from typing import Callable, TypeVar, Generic, Self


import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

PyNum = int | float | complex  # Union[int, float, complex]
Numeric = PyNum | np.number | npt.NDArray[np.number]
PyNumT = TypeVar("PyNumT", int, float, complex)  # Python number type
NPNumT_co = TypeVar("NPNumT_co", bound=np.number, covariant=True)  # Numpy number type
NPNumTb_co = TypeVar(
    "NPNumTb_co", bound=np.number, covariant=True
)  # Numpy number type bis
TaperT = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]


@dc.dataclass(slots=True, frozen=True)
class Series(Generic[NPNumT_co]):
    """A series of numbers on a grid."""

    grid: npt.NDArray[np.floating]
    """ The grid of the series. """

    signal: npt.NDArray[NPNumT_co]
    """ The signal of the series. """

    @property
    def is_consistent(self):
        """Check if the grid and signal have the same length."""
        return len(self.grid) == len(self.signal)

    @property
    def has_even_spacing(self):
        """Check if the grid has even spacing."""
        return np.allclose(np.diff(self.grid), self.grid[1] - self.grid[0])

    def create_like(self, signal: npt.NDArray[NPNumTb_co]):
        """Create a new series with the same grid as the current one."""
        # We need to ignore the type check below because we might change the type of the signal.
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
class FrequencySeries(Series[NPNumT_co], Generic[NPNumT_co]):
    """A series of numbers on a frequency grid."""

    @property
    def frequencies(self):
        """The frequencies of the series."""
        return self.grid

    @property
    def df(self):
        """The frequency spacing.

        Note
        ----
        This returns just the difference between the first two frequencies.
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

    def exp(self):
        """Return the exponential of the series."""
        return self.create_like(np.exp(self.signal))

    def sqrt(self):
        """Return the square root of the series."""
        return self.create_like(np.sqrt(self.signal))

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
class TimeSeries(Series[NPNumT_co], Generic[NPNumT_co]):
    """A series of numbers on a time grid."""

    @property
    def times(self):
        """The times of the series."""
        return self.grid

    @property
    def dt(self):
        """The time step.

        Note
        ----
        This returns just the difference between the first two times.
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
