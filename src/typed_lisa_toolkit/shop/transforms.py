"""Functions for transforming data and waveforms between time-domain, frequency-domain, and time-frequency plane."""  # noqa: E501

import warnings
from types import ModuleType
from typing import Literal, overload

from .. import _constructors  # pyright: ignore[reportPrivateUsage]
from ..types import Array, Grid2DCartesian, Linspace, data
from ..types import representations as reps
from ..types.misc import Axis


def _import_wdm_transform() -> ModuleType:
    try:
        import wdm_transform
    except ImportError:
        msg = (
            "The `wdm_transform` package is required for WDM transformations."
            "Install it with: pip install wdm-transform"
        )
        raise ImportError(
            msg,
        ) from None
    else:
        return wdm_transform


def _conventionaize(ary: "Array") -> "Array":
    return ary[None, None, None, None, ...]


# Meyer window default parameters
DEFAULT_WINDOW_A = 1 / 3
DEFAULT_WINDOW_D = 1.0


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
        When the input is a :class:`~types.TimeSeries` instance,
        the ``keep_time`` argument is ignored.
    """  # noqa: E501
    xp = td.xp
    try:
        fft = xp.fft
    except AttributeError as e:
        msg = (
            f"{xp.__name__} does not support FFT operations, "
            "which are required for `time2freq`."
        )
        raise NotImplementedError(msg) from e
    _freqs = fft.rfftfreq(len(td.times), d=td.times.step)
    freqs = _constructors.linspace(_freqs[0], _freqs[-1], len(_freqs))
    signal = fft.rfft(td.get_kernel() * td.times.step, axis=-1)
    if isinstance(td, reps.TimeSeries):
        return _constructors.frequency_series(
            frequencies=freqs,
            entries=signal,
        )
    fsd = _constructors.fsdata(
        frequencies=freqs,
        entries=signal,
        channels=td.channel_names,
        name=td.name,
    )
    if keep_time:
        fsd = fsd.set_times(td.times)
    return fsd


@overload
def freq2time(fsd: data.FSData, /, *, times: Axis) -> data.TSData: ...


@overload
def freq2time(
    fsd: reps.FrequencySeries[Linspace],
    /,
    *,
    times: Axis,
) -> reps.UniformTimeSeries: ...


def freq2time(
    fd: reps.FrequencySeries[Linspace] | data.FSData,
    /,
    *,
    times: Axis,
):
    """Convert frequency-domain representation or data to time-domain representation or data using the inverse real FFT."""  # noqa: E501
    xp = fd.xp
    try:
        fft = xp.fft
    except AttributeError as e:
        msg = (
            f"{xp.__name__} does not support FFT operations, "
            "which are required for `freq2time`."
        )
        raise NotImplementedError(msg) from e
    _times = Linspace.make(times)
    is_even = len(fd.frequencies) % 2 == 0
    nyquist_freq = (
        fd.frequencies.stop
        if is_even
        else fd.frequencies.stop + fd.frequencies.step / 2
    )
    nyquist_dt = 1.0 / (2 * nyquist_freq)
    if _times.step < nyquist_dt and not xp.isclose(_times.step, nyquist_dt):
        warnings.warn("The time grid is denser than the Nyquist limit.", stacklevel=2)

    signal = fft.irfft(fd.get_kernel() / _times.step, n=len(_times), axis=-1)
    if isinstance(fd, reps.FrequencySeries):
        return _constructors.time_series(times=_times, entries=signal)
    return _constructors.tsdata(
        times=_times,
        entries=signal,
        channels=fd.channel_names,
        name=fd.name,
    )


@overload
def time2wdm(
    tdata: data.TSData,
    /,
    *,
    Nt: int,  # noqa: N803
    Nf: int,  # noqa: N803
) -> data.WDMData[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def time2wdm(
    tseries: reps.TimeSeries[Linspace],
    /,
    *,
    Nt: int,  # noqa: N803
    Nf: int,  # noqa: N803
) -> reps.WDM[Grid2DCartesian[Linspace, Linspace]]: ...


def time2wdm(tthing: data.TSData | reps.TimeSeries[Linspace], /, *, Nt: int, Nf: int):  # noqa: N803
    """Transform a time series to WDM.

    .. note::
        The convention for signal duration in :class:`WDM` is Nt*ΔT = N*Δt. This is not
        equivalent to the convention ``grid[-1] - grid[0]``, more useful for nonuniform
        grids, that you may be assuming elsewhere.

    Parameters
    ----------
    tseries : TimeSeries
        Regularly-sampled time series of length at least Nf*Nt.
    Nt : int
        Length of WDM time grid.
    Nf : int
        Nf+1 is the length of the WDM frequency grid.

    Returns
    -------
    WDM
        Transform of the first Nf*Nt points of the time series.

    Raises
    ------
    ValueError
        If the time series is too small for the chosen values of Nf and Nt.
    """
    _import_wdm_transform()
    from wdm_transform.transforms import (
        forward_wdm as _forward_wdm,
    )

    if isinstance(tthing, data.TSData):
        return _constructors.wdmdata(
            {key: time2wdm(val, Nt=Nt, Nf=Nf) for (key, val) in tthing.items()},
            name=tthing.name,
        )
    assert isinstance(tthing, reps.TimeSeries)  # noqa: S101
    tseries = tthing

    if Nt * Nf > tseries.times.num:
        msg = "Time series too small for given Nf and Nt"
        raise ValueError(msg)

    tseries = tseries[: Nt * Nf]
    _entries = tseries.entries.squeeze()
    if _entries.ndim != 1:
        msg = "Currently only single-channel time series are supported by time2wdm."
        raise ValueError(msg)
    coeffs = _forward_wdm(
        _entries,
        nt=Nt,
        nf=Nf,
        a=DEFAULT_WINDOW_A,
        d=DEFAULT_WINDOW_D,
        dt=tseries.times.step,
    ).T
    if coeffs.shape != (Nf + 1, Nt):
        msg = "Unexpected shape of WDM coefficients."
        raise ValueError(msg)

    dT = Nf * tseries.times.step  # noqa: N806
    dF = 0.5 / dT  # noqa: N806
    tgrid = Linspace(start=tseries.times.start, step=dT, num=Nt)
    fgrid = Linspace(start=0, step=dF, num=Nf + 1)

    return _constructors.wdm(
        times=tgrid,
        frequencies=fgrid,
        entries=_conventionaize(coeffs),
    )


@overload
def wdm2time(
    wdmdata: data.WDMData[Grid2DCartesian[Linspace, Linspace]],
    /,
) -> data.TSData: ...


@overload
def wdm2time(
    wdm: reps.WDM[Grid2DCartesian[Linspace, Linspace]],
    /,
) -> reps.UniformTimeSeries: ...


def wdm2time(
    wdmthing: reps.WDM[Grid2DCartesian[Linspace, Linspace]]
    | data.WDMData[Grid2DCartesian[Linspace, Linspace]],
    /,
):
    """Transform WDM expansion to equivalent time series.

    Parameters
    ----------
    wdm : WDM or WDMData
        WDM expansion with grid parameters Nf and Nt.

    Returns
    -------
    TimeSeries or TSData
        Time series of size Nt*Nf.
    """
    _import_wdm_transform()
    from wdm_transform.transforms import (
        inverse_wdm as _inverse_wdm,
    )

    if isinstance(wdmthing, data.WDMData):
        return _constructors.tsdata(
            {key: wdm2time(val) for (key, val) in wdmthing.items()},
            name=wdmthing.name,
        )
    assert isinstance(wdmthing, reps.WDM)  # noqa: S101
    wdm = wdmthing
    _coeffs = wdm.entries.squeeze()
    if _coeffs.ndim != 2:  # noqa: PLR2004
        msg = "Currently only single-batch WDMs are supported by wdm2time."
        raise ValueError(msg)

    entries = _inverse_wdm(
        coeffs=_coeffs.T,
        dt=wdm.dt,
        a=DEFAULT_WINDOW_A,
        d=DEFAULT_WINDOW_D,
    )
    tgrid = Linspace(wdm.times.start, wdm.dt, wdm.ND)
    return _constructors.time_series(tgrid, _conventionaize(entries))


@overload
def freq2wdm(
    fseries: reps.FrequencySeries[Linspace],
    /,
    *,
    Nt: int,  # noqa: N803
    Nf: int,  # noqa: N803
    t0: float = 0.0,
) -> reps.WDM[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def freq2wdm(
    fdata: data.FSData,
    /,
    *,
    Nt: int,  # noqa: N803
    Nf: int,  # noqa: N803
    t0: float = 0.0,
) -> data.WDMData[Grid2DCartesian[Linspace, Linspace]]: ...


def freq2wdm(
    fthing: reps.FrequencySeries[Linspace] | data.FSData,
    /,
    *,
    Nt: int,  # noqa: N803
    Nf: int,  # noqa: N803
    t0: float = 0.0,
):
    """Transform frequency series to WDM.

    Parameters
    ----------
    fseries : :class:`~types.FrequencySeries` or :class:`~types.FSData`
        frequency series, assumed to be full-grid (from DC to Nyquist).
    Nt : int
        Number of WDM time bins.
    Nf : int
        WDM frequency grid has length Nf+1.
    t0 : float, optional
        Initial time of WDM time grid, by default 0.0
    """
    _import_wdm_transform()
    from wdm_transform.transforms import (
        get_backend as _get_backend,
    )

    if isinstance(fthing, data.FSData):
        return _constructors.wdmdata(
            {key: freq2wdm(val, Nt=Nt, Nf=Nf, t0=t0) for (key, val) in fthing.items()},
            name=fthing.name,
        )
    assert isinstance(fthing, reps.FrequencySeries), (  # noqa: S101
        f"Expected a FrequencySeries input, got {type(fthing)}"
    )
    fseries = fthing
    backend = _get_backend()
    tseries_entries = backend.fft.irfft(fseries.entries, n=Nf * Nt)
    duration = 1 / fseries.frequencies.step
    dt = duration / (Nf * Nt)
    tseries_grid = Linspace(start=t0, step=dt, num=Nf * Nt)
    tseries = _constructors.time_series(tseries_grid, tseries_entries)
    return time2wdm(tseries, Nt=Nt, Nf=Nf)


@overload
def wdm2freq(
    wdmdata: data.WDMData[Grid2DCartesian[Linspace, Linspace]],
    /,
) -> data.FSData: ...


@overload
def wdm2freq(
    wdm: reps.WDM[Grid2DCartesian[Linspace, Linspace]],
    /,
) -> reps.UniformFrequencySeries: ...


def wdm2freq(
    wdmthing: reps.WDM[Grid2DCartesian[Linspace, Linspace]]
    | data.WDMData[Grid2DCartesian[Linspace, Linspace]],
    /,
):
    """Transform WDM expansion to a frequency series.

    .. note::
        The input WDM representation is assumed to
        contain all frequencies from DC to Nyquist.
    """
    _import_wdm_transform()
    from wdm_transform.transforms import (
        frequency_wdm as _frequency_wdm,
    )

    if isinstance(wdmthing, data.WDMData):
        return _constructors.fsdata(
            {key: wdm2freq(val) for (key, val) in wdmthing.items()},
            name=wdmthing.name,
        )
    assert isinstance(wdmthing, reps.WDM)  # noqa: S101
    wdm = wdmthing
    _coeffs = wdm.entries.squeeze()
    if _coeffs.ndim != 2:  # noqa: PLR2004
        msg = "Currently only single-batch WDMs are supported by wdm2freq."
        raise ValueError(msg)

    wtfs = _frequency_wdm(_coeffs.T, dt=wdm.dt, a=DEFAULT_WINDOW_A, d=DEFAULT_WINDOW_D)
    # wtfs is on a grid from fftfreq but we want rfftfreq
    _num = wtfs.n // 2 + 1 if wtfs.n % 2 == 0 else (wtfs.n + 1) // 2
    freqs = _constructors.linspace_from_step(start=0.0, step=wdm.df, num=_num)
    _entries = _conventionaize(wtfs.data[:_num])
    return _constructors.frequency_series(freqs, entries=_entries)
