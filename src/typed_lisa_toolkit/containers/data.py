"""Module for data containers."""

from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import TypeVar, Generic, Self, TYPE_CHECKING
import warnings

import numpy as np
import numpy.typing as npt

from . import arithdicts, series

if TYPE_CHECKING:
    from ..viz import data as data_plotter

log = logging.getLogger(__name__)

ValueT = TypeVar("ValueT", bound=series.Series)
# In the future, we can add Time-Frequency matrices to the bound
floatingT = TypeVar("floatingT", bound=np.floating)
numberT = TypeVar("numberT", bound=np.number)


def get_subset_slice(
    increasing_array: npt.NDArray[np.floating], min: float, max: float
):
    """Return the slice for the subset [min, max] of the increasing array."""
    start_idx = np.searchsorted(increasing_array, min, side="left")
    end_idx = np.searchsorted(increasing_array, max, side="right")
    return slice(start_idx, end_idx)


class Data(arithdicts.ChannelDict[ValueT], Generic[ValueT]):
    """Dictionary data container."""

    @property
    def grid(self):
        """Return the grid."""
        return next(iter(self.values())).grid

    def get_subset(self, *, interval: tuple[float, float] | None = None):
        """Return the subset as a new instance."""
        if interval is None:
            return self
        mask = get_subset_slice(self.grid, interval[0], interval[1])
        value_type = type(next(iter(self.values())))
        series_dict = {
            chnname: value_type(grid=self.grid[mask], signal=chn.signal[mask])
            for chnname, chn in self.items()
        }
        return self.create_new(series_dict)

    def _get_plotter(self) -> type[data_plotter.DataPlotter]:
        """Return the plotter class."""
        raise NotImplementedError("The method must be implemented in the subclass.")

    def draw(
        self,
        compare_to: Self | None = None,
        *,
        interval: tuple[float, float] | None = None,
        **kwargs,
    ):
        """Plot the TSData instance in its domain or plot its comparison with another TSData instance."""
        plotter = self._get_plotter()

        if compare_to is None:
            return plotter(self.get_subset(interval=interval)).draw(**kwargs)
        return plotter(self.get_subset(interval=interval)).compare(
            compare_to.get_subset(interval=interval), **kwargs
        )


class TSData(Data[series.TimeSeries[floatingT]]):
    """Dictionary data container of time series data."""

    @property
    def times(self):
        """Return the times."""
        return next(iter(self.values())).times

    @property
    def dt(self):
        """Return the time step."""
        return next(iter(self.values())).dt

    def get_frequencies(self):
        """Return the frequencies grid matching the time grid."""
        return np.fft.rfftfreq(len(self.times), d=self.dt)

    def get_fsdata(
        self, *, keep_times: bool = True, tapering: series.TaperT | None = None
    ):
        """Return the frequency series data."""
        fsdict = {chnname: chn.rfft(tapering=tapering) for chnname, chn in self.items()}
        if keep_times:
            return TimedFSData(fsdict, times=self.times)
        return FSData(fsdict)

    def get_zero_padded(
        self, pad_time: tuple[float, float], tapering: series.TaperT | None = None
    ):
        """Return the zero-padded data."""
        pad_width = tuple(int(np.rint(time / self.dt)) for time in pad_time)
        time_end_values = (
            -self.dt * pad_width[0] + self.times[0],
            self.dt * pad_width[1] + self.times[-1],
        )
        padded_time = np.pad(
            self.times, pad_width, mode="linear_ramp", end_values=time_end_values
        )
        tapering_window = tapering(self.times) if tapering is not None else 1

        def get_padded_ts(chn: series.TimeSeries):
            signal = chn.signal * tapering_window
            padded_signal = np.pad(signal, pad_width, mode="constant")
            return series.TimeSeries(grid=padded_time, signal=padded_signal)

        tsdict = {chnname: get_padded_ts(chn) for chnname, chn in self.items()}
        return self.create_new(tsdict)

    def _get_plotter(self):
        from ..viz import data as data_plotter  # pylint: disable=import-outside-toplevel

        return data_plotter.TimeSeriesPlotter


class FSData(Data[series.FrequencySeries[numberT]]):
    """Dictionary data container of frequency series data."""

    @property
    def frequencies(self):
        """Return the frequencies."""
        return next(iter(self.values())).frequencies

    @property
    def df(self):
        """Return the frequency step."""
        return next(iter(self.values())).df

    def conj(self):
        """Return the conjugate of the data."""
        return self.create_new({chnname: chn.conj() for chnname, chn in self.items()})

    def angle(self):
        """Return the angle of the data."""
        return self.create_new({chnname: chn.angle() for chnname, chn in self.items()})

    def abs(self):
        """Return the absolute value of the data."""
        return self.create_new({chnname: chn.abs() for chnname, chn in self.items()})

    def exp(self):
        """Return the exponential of the data."""
        return self.create_new({chnname: chn.exp() for chnname, chn in self.items()})

    def real(self):
        """Return the real part of the data."""
        return self.create_new({chnname: chn.real() for chnname, chn in self.items()})

    def imag(self):
        """Return the imaginary part of the data."""
        return self.create_new({chnname: chn.imag() for chnname, chn in self.items()})

    def set_times(self, times: npt.NDArray[floatingT]):
        """Set the time grid."""
        return TimedFSData(self, times)

    def get_tsdata(
        self, times: npt.NDArray[floatingT], *, tapering: series.TaperT | None
    ):
        """Return the time series data."""
        dt: floatingT = times[1] - times[0]
        nyquist_dt = 1 / (2 * self.frequencies[-1])
        if dt < nyquist_dt and not np.isclose(dt, nyquist_dt):
            warnings.warn("The time grid is too coarse.")
        tsdict = {
            chnname: chn.irfft(times, tapering=tapering)
            for chnname, chn in self.items()
        }
        return TSData(tsdict)

    def _get_plotter(self):
        from ..viz import data as data_plotter  # pylint: disable=import-outside-toplevel

        return data_plotter.FrequencySeriesPlotter


class TimedFSData(FSData[numberT], Generic[numberT]):
    """Dictionary data container for frequency series data with time information."""

    def __init__(
        self,
        data: Mapping[str, series.FrequencySeries[numberT]],
        times: npt.NDArray[np.floating],
    ):
        super().__init__(data)
        self.times = times

    def set_times(self, times: npt.NDArray[np.floating]):
        """Set the time grid.

        This method returns `self` to allow for fluent method chaining.
        """
        self.times = times
        return self

    def create_new(self, data: Mapping[str, series.FrequencySeries[numberT]]):  # type: ignore # noqa: D102
        # Unless series.FrequencySeries is wrongly implemented so that it does not follow
        # the SupportsArithmetic protocol, there is no reason to think the type hint is wrong.
        # It is unlcear to me why mypy thinks it is wrong.
        new = type(self)(data, self.times)
        return new

    def drop_times(self):
        """Drop the time grid."""
        return FSData(self.data)

    def get_tsdata(self, *, tapering: series.TaperT | None = None):  # type: ignore
        # Indeed the signature of this subclass method is different from the superclass method.
        # This is intentional, as the subclass method is more specific. Ignoring the error.
        """Return the time series data."""
        return super().get_tsdata(self.times, tapering=tapering)
