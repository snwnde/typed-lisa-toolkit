from types import ModuleType

import numpy as np
import numpy.testing as npt
import pytest

from typed_lisa_toolkit import shop
from typed_lisa_toolkit.types import (
    WDM,
    Axis,
    FSData,
    Linspace,
    TimeSeries,
    TSData,
    UniformFrequencySeries,
    WDMData,
)


def _require_wdm_transform():
    try:
        import wdm_transform
    except ImportError:
        msg = (
            "The wdm_transform package is required for WDM conversion tests. "
            "Please install it with `pip install wdm_transform`."
        )
        pytest.skip(msg)
    else:
        return wdm_transform


def test_time2freq_timeseries_returns_representation(
    xp: ModuleType,
    lin_time_series: TimeSeries[Axis[Linspace]],
):
    fs = shop.time2freq(lin_time_series)

    assert isinstance(fs, UniformFrequencySeries)
    expected = xp.fft.rfft(
        lin_time_series.entries * lin_time_series.times.ax.step, axis=-1
    )
    npt.assert_allclose(np.asarray(fs.entries), expected)


def test_time2freq_tsdata_keep_time_switches_type(
    tsdata: TSData,
):
    times = tsdata.times
    fs_no_time = shop.time2freq(tsdata, keep_time=False)
    fs_with_time = shop.time2freq(tsdata, keep_time=True)

    assert isinstance(fs_no_time, FSData)
    assert "times" not in dir(fs_no_time)
    assert isinstance(fs_with_time, FSData)
    npt.assert_allclose(np.asarray(fs_with_time.times.ax), times.ax)
    assert fs_with_time.channel_names == tsdata.channel_names


def test_freq2time_frequencyseries_roundtrip(
    lin_time_series: TimeSeries[Axis[Linspace]],
):
    fs = shop.time2freq(lin_time_series)
    recovered = shop.freq2time(fs, times=np.asarray(lin_time_series.times))

    assert isinstance(recovered, TimeSeries)
    npt.assert_allclose(
        np.asarray(recovered.entries),
        np.asarray(lin_time_series.entries),
        atol=1e-12,
    )


def test_freq2time_fsdata_returns_tsdata(
    tsdata: TSData,
):
    times = tsdata.times
    fsd = shop.time2freq(tsdata, keep_time=False)
    recovered = shop.freq2time(fsd, times=times)

    assert isinstance(recovered, TSData)
    assert recovered.channel_names == tsdata.channel_names
    npt.assert_allclose(np.asarray(recovered.times.ax), np.asarray(times.ax))
    npt.assert_allclose(
        np.asarray(recovered.get_kernel()),
        np.asarray(tsdata.get_kernel()),
        atol=1e-12,
    )


def test_freq2time_warns_for_denser_than_nyquist_times(
    lin_time_series: TimeSeries[Axis[Linspace]],
):
    fs = shop.time2freq(lin_time_series)

    dense_times = np.linspace(
        float(lin_time_series.times.start),
        float(lin_time_series.times.stop),
        2 * len(lin_time_series.times) - 1,
    )
    with pytest.warns(UserWarning, match="denser than the Nyquist limit"):
        _ = shop.freq2time(fs, times=dense_times)


def test_time2wdm_roundtrip_timeseries(
    long_time_series: TimeSeries[Axis[Linspace]],
):
    _require_wdm_transform()

    wdm = shop.time2wdm(long_time_series, Nt=4, Nf=2)
    recovered = shop.wdm2time(wdm)

    assert isinstance(wdm, WDM)
    assert isinstance(recovered, TimeSeries)
    npt.assert_allclose(
        np.asarray(recovered.times.ax), np.asarray(long_time_series.times.ax)
    )
    npt.assert_allclose(
        np.asarray(recovered.entries),
        np.asarray(long_time_series.entries),
        atol=1e-12,
    )


def test_time2wdm_roundtrip_tsdata(
    long_tsdata: TSData,
):
    _require_wdm_transform()

    wdmdata = shop.time2wdm(long_tsdata, Nt=4, Nf=2)
    recovered = shop.wdm2time(wdmdata)

    assert isinstance(wdmdata, WDMData)
    assert isinstance(recovered, TSData)
    assert recovered.channel_names == long_tsdata.channel_names
    npt.assert_allclose(
        np.asarray(recovered.times.ax), np.asarray(long_tsdata.times.ax)
    )
    npt.assert_allclose(
        np.asarray(recovered.get_kernel()),
        np.asarray(long_tsdata.get_kernel()),
        atol=1e-12,
    )


def test_freq2wdm_roundtrip_frequencyseries(
    long_time_series: TimeSeries[Axis[Linspace]],
):
    _require_wdm_transform()

    fs = shop.time2freq(long_time_series)
    wdm = shop.freq2wdm(fs, Nt=4, Nf=2)
    recovered = shop.wdm2freq(wdm)

    assert isinstance(wdm, WDM)
    assert isinstance(recovered, UniformFrequencySeries)
    npt.assert_allclose(
        np.asarray(recovered.frequencies.ax),
        np.asarray(fs.frequencies.ax),
        err_msg="Frequencies do not match after round-trip conversion.",
    )
    npt.assert_allclose(
        np.asarray(recovered.entries),
        np.asarray(fs.entries),
        err_msg="Kernel entries do not match after round-trip conversion.",
    )


def test_freq2wdm_roundtrip_fsdata(
    long_tsdata: TSData,
):
    _require_wdm_transform()

    fsd = shop.time2freq(long_tsdata, keep_time=False)
    wdmdata = shop.freq2wdm(fsd, Nt=4, Nf=2)
    recovered = shop.wdm2freq(wdmdata)

    assert isinstance(wdmdata, WDMData)
    assert isinstance(recovered, FSData)
    assert recovered.channel_names == fsd.channel_names
    npt.assert_allclose(
        np.asarray(recovered.get_kernel()),
        np.asarray(fsd.get_kernel()),
        err_msg="Kernel entries do not match after round-trip conversion.",
    )
    npt.assert_allclose(
        np.asarray(recovered.frequencies.ax),
        np.asarray(fsd.frequencies.ax),
        err_msg="Frequencies do not match after round-trip conversion.",
    )


def test_time2freq_tsdata_deprecation_warning(
    tsdata: TSData,
):
    with pytest.warns(DeprecationWarning, match="to_fsdata"):
        _ = tsdata.to_fsdata()
    with pytest.warns(DeprecationWarning, match="to_fsdata"):
        _ = tsdata.to_fsdata(keep_times=False)
