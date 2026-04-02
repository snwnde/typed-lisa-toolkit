import warnings
from typing import Literal, overload

from .. import _constructors  # pyright: ignore[reportPrivateUsage]
from ..types import data
from ..types import representations as reps
from ..types.misc import Axis, Linspace


@overload
def time2freq(
    tsd: data.TSData,
    /,
    *,
    keep_time: Literal[True] = True,
) -> data.TimedFSData: ...


@overload
def time2freq(
    tsd: data.TSData,
    /,
    *,
    keep_time: Literal[False],
) -> data.FSData: ...


@overload
def time2freq(
    ts: reps.TimeSeries[Linspace],
    /,
    *,
    keep_time: bool = True,
) -> reps.UniformFrequencySeries: ...


def time2freq(
    td: reps.TimeSeries[Linspace] | data.TSData,
    /,
    *,
    keep_time: bool = True,
):
    """Convert time-domain representation or data to frequency-domain representation or data using the real FFT.

    .. note::
        When the input is a :class:`~types.representations.TimeSeries` instance,
        the ``keep_time`` argument is ignored.
    """
    xp = td.xp
    try:
        fft = xp.fft
    except AttributeError:
        raise NotImplementedError(
            f"{xp.__name__} does not support FFT operations, which are required for `time2freq`."
        )
    _freqs = fft.rfftfreq(len(td.times), d=td.times.step)
    freqs = _constructors.linspace(_freqs[0], _freqs[1] - _freqs[0], len(_freqs))
    signal = fft.rfft(td.entries * td.times.step, axis=-1)
    fs = _constructors.frequency_series(
        frequencies=freqs,
        entries=signal,
    )
    if isinstance(td, reps.TimeSeries):
        return fs
    else:
        fsd = data.FSData(fs, td.channel_names, td.name)
        if keep_time:
            fsd = fsd.set_times(td.times)
        return fsd


@overload
def freq2time(fsd: data.FSData, /, *, times: Axis) -> data.TSData: ...


@overload
def freq2time(
    fsd: reps.FrequencySeries[Linspace], /, *, times: Axis
) -> reps.UniformTimeSeries: ...


def freq2time(
    fd: reps.FrequencySeries[Linspace] | data.FSData,
    /,
    *,
    times: Axis,
):
    """Convert frequency-domain representation or data to time-domain representation or data using the inverse real FFT."""
    xp = fd.xp
    try:
        fft = xp.fft
    except AttributeError:
        raise NotImplementedError(
            f"{xp.__name__} does not support FFT operations, which are required for `freq2time`."
        )
    _times = Linspace.make(times)
    is_even = len(fd.frequencies) % 2 == 0
    nyquist_freq = (
        fd.frequencies.stop
        if is_even
        else fd.frequencies.stop + fd.frequencies.step / 2
    )
    nyquist_dt = 1.0 / (2 * nyquist_freq)
    if _times.step < nyquist_dt and not xp.isclose(_times.step, nyquist_dt):
        warnings.warn("The time grid is denser than the Nyquist limit.")

    signal = fft.irfft(fd.entries / _times.step, n=len(_times), axis=-1)
    ts = _constructors.time_series(times=_times, entries=signal)
    if isinstance(fd, reps.FrequencySeries):
        return ts
    else:
        tsd = data.TSData(ts, fd.channel_names, fd.name)
        return tsd
