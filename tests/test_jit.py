from typing import cast

import pytest

try:
    import jax
except ImportError:
    msg = (
        "The jax package is required for JIT compilation tests. "
        "Please install it with `pip install jax`."
    )
    pytest.skip(msg, allow_module_level=True)
else:
    # No need to set float64 unless we want to test with NumPy fixtures
    # automatically converted to JAX. Fixtures in JAX already use float64.
    pass


import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import densify_phasor_hw, shop
from typed_lisa_toolkit.types import (
    WDM,
    Array,
    Axis,
    EvolutionarySpectralDensity,
    FSData,
    Grid2DCartesian,
    Harmonic,
    HarmonicWaveform,
    Interpolator,
    Linspace,
    Phasor,
    SpectralDensity,
    TimedFSData,
    TimeSeries,
    TSData,
    UniformFrequencySeries,
    UniformTimeSeries,
)


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_xyz2aet_fsdata_jit(fsdata: FSData):
    """Test xyz2aet is jittable with FSData input."""
    xyz_fsdata = fsdata
    expected = shop.xyz2aet(xyz_fsdata)
    xyz2aet_jit = jax.jit(shop.xyz2aet)  # pyright: ignore[reportUnknownMemberType]
    aet_jit = cast("FSData", xyz2aet_jit(xyz_fsdata))

    # Verify results match
    assert aet_jit.channel_names == expected.channel_names
    npt.assert_allclose(
        np.asarray(aet_jit.get_kernel()),
        np.asarray(expected.get_kernel()),
        atol=1e-12,
    )


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_xyz2aet_sdm_jit(sdm: SpectralDensity):
    """Test xyz2aet is jittable with SpectralDensity input."""
    xyz_sdm = sdm
    expected = shop.xyz2aet(xyz_sdm)
    xyz2aet_jit = jax.jit(shop.xyz2aet)  # pyright: ignore[reportUnknownMemberType]
    aet_jit = cast("SpectralDensity", xyz2aet_jit(xyz_sdm))

    # Verify results match
    assert aet_jit.channel_order == expected.channel_order
    npt.assert_allclose(
        np.asarray(aet_jit.get_kernel()),
        np.asarray(expected.get_kernel()),
        atol=1e-12,
    )


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_xyz2aet_esdm_jit(esdm: EvolutionarySpectralDensity):
    """Test xyz2aet is jittable with EvolutionarySpectralDensity input."""
    xyz_sdm = esdm
    expected = shop.xyz2aet(xyz_sdm)
    xyz2aet_jit = jax.jit(shop.xyz2aet)  # pyright: ignore[reportUnknownMemberType]
    aet_jit = cast("EvolutionarySpectralDensity", xyz2aet_jit(xyz_sdm))

    # Verify results match
    assert aet_jit.channel_order == expected.channel_order
    npt.assert_allclose(
        np.asarray(aet_jit.get_kernel()),
        np.asarray(expected.get_kernel()),
        atol=1e-12,
    )


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_densify_phasor_hw_jit(
    hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[Array]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[Array],
):
    expected = densify_phasor_hw(
        hw_phasor, linear_interpolator, dense_ary_freq_axis, embed=True
    )
    densify_phasor_hw_jit = jax.jit(  # pyright: ignore[reportUnknownMemberType]
        densify_phasor_hw, static_argnums=(1,), static_argnames=("embed",)
    )
    # Verify results match
    hhw_jit = cast(
        "HarmonicWaveform[Harmonic, Phasor[Axis[Array]]]",
        densify_phasor_hw_jit(
            hw_phasor, linear_interpolator, dense_ary_freq_axis, embed=True
        ),
    )
    assert isinstance(hhw_jit, type(expected))
    assert (
        next(iter(hhw_jit.values())).entries == next(iter(expected.values())).entries
    ).all()


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_time2freq_timeseries_jit(
    lin_time_series: UniformTimeSeries,
):
    expected = shop.time2freq(lin_time_series)
    time2freq_jit = jax.jit(shop.time2freq)  # pyright: ignore[reportUnknownMemberType]
    fs_jit = cast("UniformFrequencySeries", time2freq_jit(lin_time_series))
    assert isinstance(fs_jit, UniformFrequencySeries)
    assert (fs_jit.entries == expected.entries).all()


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_time2freq_tsdata_jit(
    tsdata: TSData,
):
    expected = shop.time2freq(tsdata, keep_time=True)
    time2freq_jit = jax.jit(shop.time2freq, static_argnames=("keep_time",))  # pyright: ignore[reportUnknownMemberType]
    fs_jit = cast("TimedFSData", time2freq_jit(tsdata, keep_time=True))
    assert isinstance(fs_jit, TimedFSData)
    assert fs_jit.times == expected.times
    assert fs_jit.frequencies == expected.frequencies
    assert (fs_jit.entries == expected.entries).all()


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_freq2time_frequencyseries_jit(
    lin_time_series: UniformTimeSeries,
):
    fs = shop.time2freq(lin_time_series)
    expected = shop.freq2time(fs, times=lin_time_series.times)
    freq2time_jit = jax.jit(shop.freq2time)  # pyright: ignore[reportUnknownMemberType]
    recovered_jit = cast(
        "UniformTimeSeries", freq2time_jit(fs, times=lin_time_series.times)
    )
    assert isinstance(recovered_jit, UniformTimeSeries)
    npt.assert_allclose(
        np.asarray(recovered_jit.entries),
        np.asarray(expected.entries),
        atol=1e-12,
    )


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_freq2time_fsdata_jit(
    tsdata: TSData,
):
    times = tsdata.times
    fsd = shop.time2freq(tsdata, keep_time=False)
    expected = shop.freq2time(fsd, times=times)
    freq2time_jit = jax.jit(shop.freq2time)  # pyright: ignore[reportUnknownMemberType]
    recovered_jit = cast("TSData", freq2time_jit(fsd, times=times))
    assert isinstance(recovered_jit, TSData)
    assert recovered_jit.channel_names == expected.channel_names
    npt.assert_allclose(np.asarray(recovered_jit.times.ax), np.asarray(times.ax))
    npt.assert_allclose(
        np.asarray(recovered_jit.get_kernel()),
        np.asarray(expected.get_kernel()),
        atol=1e-12,
    )


@pytest.mark.parametrize("xp", ["jax"], indirect=True)
def test_time2wdm_timeseries_jit(
    long_time_series: TimeSeries[Axis[Linspace]],
):
    expected = shop.time2wdm(long_time_series, Nt=4, Nf=2)
    time2wdm_jit = jax.jit(shop.time2wdm, static_argnames=("Nt", "Nf"))  # pyright: ignore[reportUnknownMemberType]
    wdm_jit = cast(
        "WDM[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]",
        time2wdm_jit(long_time_series, Nt=4, Nf=2),
    )
    assert isinstance(wdm_jit, WDM)
    assert (wdm_jit.entries == expected.entries).all()
