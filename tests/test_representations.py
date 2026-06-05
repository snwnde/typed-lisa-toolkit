from types import ModuleType
from typing import cast

import pytest

from typed_lisa_toolkit import (
    densify_phasor,
    frequency_phasor,
    frequency_series,
    phasor,
    stft,
    time_series,
    wdm,
)
from typed_lisa_toolkit.types import (
    STFT,
    WDM,
    AnyArray,
    Axis,
    FrequencyPhasor,
    FrequencySeries,
    Grid2DCartesian,
    Grid2DSparse,
    Interpolator,
    Linspace,
    TimePhasor,
    TimeSeries,
    UniformFrequencySeries,
    UniformTimeSeries,
)

## Representations


def test_uni_freq_series(
    lin_freq_series: UniformFrequencySeries, long_freq_grid1d: tuple[Axis[AnyArray]]
):
    with pytest.raises(ValueError, match="num must be"):
        lin_freq_series.get_subset(interval=(0.1, 0.5))
    subset = lin_freq_series.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    subset = lin_freq_series.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    embed = lin_freq_series.get_embedded(embedding_grid=long_freq_grid1d)
    assert embed.grid == long_freq_grid1d
    embed = lin_freq_series.get_embedded(
        embedding_grid=long_freq_grid1d, known_slices=(slice(0, 3),)
    )
    assert embed.grid == long_freq_grid1d
    # Reject both interval and slice arguments together
    with pytest.raises(ValueError, match=r".+"):
        lin_freq_series.get_subset(interval=(0.2, 0.6), slice=slice(2, 7))
    # Reject tuple of slices
    with pytest.raises(TypeError, match=r".+"):
        lin_freq_series.get_subset(slice=(slice(2, 6),))  # pyright: ignore[reportArgumentType]
    assert lin_freq_series.f_min == lin_freq_series.grid[0][0]
    assert lin_freq_series.f_max == lin_freq_series.grid[0][-1]
    with pytest.warns(DeprecationWarning, match="irfft"):
        _ = lin_freq_series.irfft(
            lin_freq_series.xp.linspace(0, 10, 2 * len(lin_freq_series.grid[0]))
        )


def test_uni_ary_freq_series(
    uni_ary_freq_series: FrequencySeries[Axis[AnyArray]],
):
    subset = uni_ary_freq_series.get_subset(interval=(0.1, 0.5))
    assert len(subset.grid[0]) == 0
    subset = uni_ary_freq_series.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    subset = uni_ary_freq_series.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    embed = uni_ary_freq_series.get_embedded(embedding_grid=uni_ary_freq_series.grid)
    assert embed.grid == uni_ary_freq_series.grid
    embed = uni_ary_freq_series.get_embedded(
        embedding_grid=uni_ary_freq_series.grid, known_slices=(slice(0, 3),)
    )
    assert embed.grid == uni_ary_freq_series.grid
    assert uni_ary_freq_series.f_min == uni_ary_freq_series.grid[0][0]
    assert uni_ary_freq_series.f_max == uni_ary_freq_series.grid[0][-1]


def test_ary_freq_series(
    ary_freq_series: FrequencySeries[Axis[AnyArray]],
):
    subset = ary_freq_series.get_subset(interval=(0.1, 0.5))
    assert len(subset.grid[0]) == 0
    subset = ary_freq_series.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    subset = ary_freq_series.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    embed = ary_freq_series.get_embedded(embedding_grid=ary_freq_series.grid)
    assert embed.grid == ary_freq_series.grid
    embed = ary_freq_series.get_embedded(
        embedding_grid=ary_freq_series.grid, known_slices=(slice(0, 3),)
    )
    assert embed.grid == ary_freq_series.grid


def test_uni_time_series(
    lin_time_series: UniformTimeSeries, long_time_grid1d: tuple[Axis[AnyArray]]
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_time_series.get_subset(interval=(0.1, 0.5))
    subset = lin_time_series.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    subset = lin_time_series.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    embed = lin_time_series.get_embedded(embedding_grid=long_time_grid1d)
    assert embed.grid == long_time_grid1d
    embed = lin_time_series.get_embedded(
        embedding_grid=long_time_grid1d, known_slices=(slice(0, 6),)
    )
    assert embed.grid == long_time_grid1d
    with pytest.warns(DeprecationWarning, match="rfft"):
        _ = lin_time_series.rfft()


def test_uni_ary_time_series(
    uni_ary_time_series: TimeSeries[Axis[AnyArray]],
):
    subset = uni_ary_time_series.get_subset(interval=(0.1, 0.5))
    assert len(subset.grid[0]) == 0
    subset = uni_ary_time_series.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    subset = uni_ary_time_series.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    embed = uni_ary_time_series.get_embedded(embedding_grid=uni_ary_time_series.grid)
    assert embed.grid == uni_ary_time_series.grid
    embed = uni_ary_time_series.get_embedded(
        embedding_grid=uni_ary_time_series.grid, known_slices=(slice(0, 6),)
    )
    assert embed.grid == uni_ary_time_series.grid


def test_ary_time_series(
    ary_time_series: TimeSeries[Axis[AnyArray]],
):
    subset = ary_time_series.get_subset(interval=(0.1, 0.5))
    assert len(subset.grid[0]) == 0
    subset = ary_time_series.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    subset = ary_time_series.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2)
    embed = ary_time_series.get_embedded(embedding_grid=ary_time_series.grid)
    assert embed.grid == ary_time_series.grid
    embed = ary_time_series.get_embedded(
        embedding_grid=ary_time_series.grid, known_slices=(slice(0, 6),)
    )
    assert embed.grid == ary_time_series.grid


def test_lin_freq_phasor(lin_freq_phasor: FrequencyPhasor[Axis[Linspace]]):
    from scipy.interpolate import interp1d

    with pytest.raises(ValueError, match="num must be"):
        _ = lin_freq_phasor.get_subset(interval=(0.1, 0.5))
    subset = lin_freq_phasor.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    subset = lin_freq_phasor.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    embed = lin_freq_phasor.get_embedded(embedding_grid=lin_freq_phasor.grid)
    assert embed.grid == lin_freq_phasor.grid
    embed = lin_freq_phasor.get_embedded(
        embedding_grid=lin_freq_phasor.grid, known_slices=(slice(0, 3),)
    )
    assert embed.grid == lin_freq_phasor.grid
    new_freqs = lin_freq_phasor.xp.linspace(
        lin_freq_phasor.frequencies[2], lin_freq_phasor.frequencies[-3], 8
    )
    interpolated = lin_freq_phasor.get_interpolated(new_freqs, interp1d)
    assert isinstance(interpolated, FrequencyPhasor)
    assert len(interpolated.frequencies) == 8
    fs = lin_freq_phasor.to_frequency_series()
    assert isinstance(fs, FrequencySeries)


def test_lin_time_phasor(lin_time_phasor: TimePhasor[Axis[Linspace]]):
    from scipy.interpolate import interp1d

    with pytest.raises(ValueError, match="num must be"):
        _ = lin_time_phasor.get_subset(interval=(0.1, 0.5))
    subset = lin_time_phasor.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    subset = lin_time_phasor.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    embed = lin_time_phasor.get_embedded(embedding_grid=lin_time_phasor.grid)
    assert embed.grid == lin_time_phasor.grid
    embed = lin_time_phasor.get_embedded(
        embedding_grid=lin_time_phasor.grid, known_slices=(slice(0, 6),)
    )
    assert embed.grid == lin_time_phasor.grid
    new_times = lin_time_phasor.xp.linspace(
        lin_time_phasor.times[2], lin_time_phasor.times[-3], 8
    )
    interpolated = lin_time_phasor.get_interpolated(new_times, interp1d)
    assert isinstance(interpolated, TimePhasor)
    assert len(interpolated.times) == 8
    ts = lin_time_phasor.to_time_series()
    assert isinstance(ts, TimeSeries)


def test_deprecated_phasor_function(lin_freq_axis: Axis[Linspace], xp: ModuleType):
    n_freqs = len(lin_freq_axis)
    amplitudes = xp.asarray([1.0] * n_freqs, dtype=xp.float64) * (1 + 1j)
    phases = xp.asarray([0.0] * n_freqs, dtype=xp.float64)

    with pytest.warns(DeprecationWarning, match="phasor.*frequency_phasor"):
        phasor_obj = phasor(
            frequencies=lin_freq_axis,
            amplitudes=amplitudes[None, None, None, None, :],
            phases=phases[None, None, None, None, :],
        )

    assert isinstance(phasor_obj, FrequencyPhasor)
    assert phasor_obj.domain == "frequency"


def test_frequency_series_factory_validation(
    lin_freq_axis: Axis[Linspace], xp: ModuleType
):
    bad_entries = xp.ones((1, 2, 1, 1, len(lin_freq_axis)), dtype=xp.float64)
    with pytest.raises(ValueError, match="Invalid shape"):
        frequency_series(lin_freq_axis, bad_entries)


def test_time_series_factory_validation(lin_time_axis: Axis[Linspace], xp: ModuleType):
    bad_entries = xp.ones((1, 1, 2, 1, len(lin_time_axis)), dtype=xp.float64)
    with pytest.raises(ValueError, match="Invalid shape"):
        time_series(lin_time_axis, bad_entries)


def test_phasor_factory_validation(lin_freq_axis: Axis[Linspace], xp: ModuleType):
    amplitudes = xp.ones((1, 1, 1, 1, len(lin_freq_axis)), dtype=xp.float64)
    phases = xp.ones((1, 1, 1, 1, len(lin_freq_axis) - 1), dtype=xp.float64)
    with pytest.raises(ValueError, match="must have the same shape"):
        frequency_phasor(
            frequencies=lin_freq_axis, amplitudes=amplitudes, phases=phases
        )

    bad_feature_shape = xp.ones((1, 1, 1, 2, len(lin_freq_axis)), dtype=xp.float64)
    with pytest.raises(ValueError, match="Invalid shape"):
        frequency_phasor(
            frequencies=lin_freq_axis,
            amplitudes=bad_feature_shape,
            phases=bad_feature_shape,
        )


def test_stft_factory_validation(
    lin_freq_axis: Axis[Linspace], lin_time_axis: Axis[Linspace], xp: ModuleType
):
    bad_dense_entries = xp.ones(
        (1, 1, 1, 1, len(lin_freq_axis), len(lin_time_axis) - 1),
        dtype=xp.float64,
    )
    with pytest.raises(ValueError, match="Invalid shape"):
        stft(lin_freq_axis, lin_time_axis, bad_dense_entries)


def test_wdm_factory_validation(
    ary_freq_axis: Axis[AnyArray], lin_time_axis: Axis[Linspace], xp: ModuleType
):
    entries = xp.ones(
        (1, 1, 1, 1, len(ary_freq_axis), len(lin_time_axis)), dtype=xp.float64
    )
    with pytest.raises(ValueError, match="Frequencies axis must be uniformly spaced"):
        wdm(ary_freq_axis, lin_time_axis, entries)  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_phasor_interpolation_rejects_noncanonical_shape(
    ary_phasor: FrequencyPhasor[Axis[AnyArray]],
):
    from scipy.interpolate import interp1d

    bad = type(ary_phasor)(
        grid=ary_phasor.grid,
        entries=ary_phasor.xp.ones(
            (1, 2, 1, 2, len(ary_phasor.frequencies)), dtype=ary_phasor.xp.float64
        ),
    )
    new_freqs = ary_phasor.xp.linspace(
        ary_phasor.frequencies[0], ary_phasor.frequencies[-1], 5
    )
    with pytest.raises(ValueError, match="Only 1D phasors"):
        bad.get_interpolated(new_freqs, interp1d)


def test_densify_phasor_embed_true(
    ary_phasor: FrequencyPhasor[Axis[AnyArray]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[AnyArray],
):
    dense = densify_phasor(
        ary_phasor, linear_interpolator, dense_ary_freq_axis, embed=True
    )
    assert isinstance(dense, FrequencyPhasor)
    assert len(dense.frequencies) == len(dense_ary_freq_axis)


def test_array_input_factory_branches(xp: ModuleType):
    freqs = cast("AnyArray", xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64))
    times = cast("AnyArray", xp.asarray([0.0, 1.0, 2.0], dtype=xp.float64))
    fs_entries = cast("AnyArray", xp.ones((1, 1, 1, 1, len(freqs)), dtype=xp.float64))
    ts_entries = cast("AnyArray", xp.ones((1, 1, 1, 1, len(times)), dtype=xp.float64))

    fs = frequency_series(freqs, fs_entries)
    ts = time_series(times, ts_entries)
    assert isinstance(fs, FrequencySeries)
    assert isinstance(ts, TimeSeries)


def test_phasor_1d_and_make_error_branch(lin_freq_axis: Axis[Linspace], xp: ModuleType):
    amps = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    phases = xp.asarray([0.0, 0.25, 0.5], dtype=xp.float64)
    with pytest.warns(DeprecationWarning, match="phasor.*frequency_phasor"):
        ph = phasor(frequencies=lin_freq_axis, amplitudes=amps, phases=phases)
    assert isinstance(ph, FrequencyPhasor)
    assert ph.entries.shape[3] == 2

    with pytest.raises(ValueError, match="either 1D arrays"):
        FrequencyPhasor.make(
            axis=lin_freq_axis,
            amplitudes=xp.ones((2, 2), dtype=xp.float64),
            phases=xp.ones((2, 2), dtype=xp.float64),
        )


def test_wdm_nonuniform_array_branch(xp: ModuleType):
    freqs = xp.asarray([1.0, 2.5, 3.0], dtype=xp.float64)
    times = xp.asarray([0.0, 1.0, 3.0], dtype=xp.float64)
    entries = xp.ones((1, 1, 1, 1, len(freqs), len(times)), dtype=xp.float64)

    with pytest.raises(ValueError, match="Frequencies axis must be uniformly spaced"):
        wdm(freqs, xp.asarray([0.0, 1.0, 2.0], dtype=xp.float64), entries)
    with pytest.raises(ValueError, match="Times axis must be uniformly spaced"):
        wdm(xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64), times, entries)


def test_series_and_tf_properties(
    lin_freq_series: UniformFrequencySeries,
    lin_time_series: UniformTimeSeries,
    lin_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    shifted = lin_freq_series.get_time_shifted(0.1)
    assert shifted.entries.shape == lin_freq_series.entries.shape

    angle = lin_freq_series.angle()
    assert isinstance(angle, FrequencySeries)

    assert lin_freq_series.df == lin_freq_series.resolution
    assert lin_time_series.dt == lin_time_series.resolution

    assert lin_time_series.t_start == lin_time_series.grid[0][0]
    assert lin_time_series.t_end == lin_time_series.grid[0][-1]

    assert lin_lin_cartesian_stft.kind == "stft"
    assert lin_lin_cartesian_stft.t_start == lin_lin_cartesian_stft.grid[1][0]
    assert lin_lin_cartesian_stft.t_end == lin_lin_cartesian_stft.grid[1][-1]
    assert lin_lin_cartesian_stft.f_min == lin_lin_cartesian_stft.grid[0][0]
    assert lin_lin_cartesian_stft.f_max == lin_lin_cartesian_stft.grid[0][-1]


def test_uni_ary_phasor(uni_ary_phasor: FrequencyPhasor[Axis[AnyArray]]):
    from scipy.interpolate import interp1d

    subset = uni_ary_phasor.get_subset(interval=(0.1, 0.5))
    assert len(subset.grid[0]) == 0
    subset = uni_ary_phasor.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    subset = uni_ary_phasor.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    embed = uni_ary_phasor.get_embedded(embedding_grid=uni_ary_phasor.grid)
    assert embed.grid == uni_ary_phasor.grid
    embed = uni_ary_phasor.get_embedded(
        embedding_grid=uni_ary_phasor.grid, known_slices=(slice(0, 3),)
    )
    assert embed.grid == uni_ary_phasor.grid
    new_freqs = uni_ary_phasor.xp.linspace(
        uni_ary_phasor.frequencies[2], uni_ary_phasor.frequencies[-3], 8
    )
    interpolated = uni_ary_phasor.get_interpolated(new_freqs, interp1d)
    assert isinstance(interpolated, FrequencyPhasor)
    assert len(interpolated.frequencies) == 8


def test_ary_phasor(ary_phasor: FrequencyPhasor[Axis[AnyArray]]):
    from scipy.interpolate import interp1d

    subset = ary_phasor.get_subset(interval=(0.1, 0.5))
    assert len(subset.grid[0]) == 0
    subset = ary_phasor.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    subset = ary_phasor.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.entries.shape == (1, 1, 1, 2, 2)
    embed = ary_phasor.get_embedded(embedding_grid=ary_phasor.grid)
    assert embed.grid == ary_phasor.grid
    embed = ary_phasor.get_embedded(
        embedding_grid=ary_phasor.grid, known_slices=(slice(0, 3),)
    )
    assert embed.grid == ary_phasor.grid
    new_freqs = ary_phasor.xp.linspace(
        ary_phasor.frequencies[2], ary_phasor.frequencies[-3], 8
    )
    interpolated = ary_phasor.get_interpolated(new_freqs, interp1d)
    assert isinstance(interpolated, FrequencyPhasor)
    assert len(interpolated.frequencies) == 8


def test_lin_lin_cartesian_stft(
    lin_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_lin_cartesian_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_lin_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = lin_lin_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = lin_lin_cartesian_stft.get_embedded(
        embedding_grid=lin_lin_cartesian_stft.grid
    )
    assert embed.grid == lin_lin_cartesian_stft.grid
    embed = lin_lin_cartesian_stft.get_embedded(
        embedding_grid=lin_lin_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_lin_cartesian_stft.grid


def test_lin_uni_cartesian_stft(
    lin_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[AnyArray]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_uni_cartesian_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_uni_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = lin_uni_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = lin_uni_cartesian_stft.get_embedded(
        embedding_grid=lin_uni_cartesian_stft.grid
    )
    assert embed.grid == lin_uni_cartesian_stft.grid
    embed = lin_uni_cartesian_stft.get_embedded(
        embedding_grid=lin_uni_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_uni_cartesian_stft.grid


def test_lin_ary_cartesian_stft(
    lin_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[AnyArray]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_ary_cartesian_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_ary_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = lin_ary_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = lin_ary_cartesian_stft.get_embedded(
        embedding_grid=lin_ary_cartesian_stft.grid
    )
    assert embed.grid == lin_ary_cartesian_stft.grid
    embed = lin_ary_cartesian_stft.get_embedded(
        embedding_grid=lin_ary_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_ary_cartesian_stft.grid


def test_ary_lin_cartesian_stft(
    ary_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = ary_lin_cartesian_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = ary_lin_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = ary_lin_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = ary_lin_cartesian_stft.get_embedded(
        embedding_grid=ary_lin_cartesian_stft.grid
    )
    assert embed.grid == ary_lin_cartesian_stft.grid
    embed = ary_lin_cartesian_stft.get_embedded(
        embedding_grid=ary_lin_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == ary_lin_cartesian_stft.grid


def test_ary_uni_cartesian_stft(
    ary_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = ary_uni_cartesian_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = ary_uni_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = ary_uni_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = ary_uni_cartesian_stft.get_embedded(
        embedding_grid=ary_uni_cartesian_stft.grid
    )
    assert embed.grid == ary_uni_cartesian_stft.grid
    embed = ary_uni_cartesian_stft.get_embedded(
        embedding_grid=ary_uni_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == ary_uni_cartesian_stft.grid


def test_uni_lin_cartesian_stft(
    uni_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = uni_lin_cartesian_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = uni_lin_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = uni_lin_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = uni_lin_cartesian_stft.get_embedded(
        embedding_grid=uni_lin_cartesian_stft.grid
    )
    assert embed.grid == uni_lin_cartesian_stft.grid
    embed = uni_lin_cartesian_stft.get_embedded(
        embedding_grid=uni_lin_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == uni_lin_cartesian_stft.grid


def test_uni_uni_cartesian_stft(
    uni_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = uni_uni_cartesian_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = uni_uni_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = uni_uni_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = uni_uni_cartesian_stft.get_embedded(
        embedding_grid=uni_uni_cartesian_stft.grid
    )
    assert embed.grid == uni_uni_cartesian_stft.grid
    embed = uni_uni_cartesian_stft.get_embedded(
        embedding_grid=uni_uni_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == uni_uni_cartesian_stft.grid


def test_uni_ary_cartesian_stft(
    uni_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = uni_ary_cartesian_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = uni_ary_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = uni_ary_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = uni_ary_cartesian_stft.get_embedded(
        embedding_grid=uni_ary_cartesian_stft.grid
    )
    assert embed.grid == uni_ary_cartesian_stft.grid
    embed = uni_ary_cartesian_stft.get_embedded(
        embedding_grid=uni_ary_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == uni_ary_cartesian_stft.grid


def test_ary_ary_cartesian_stft(
    ary_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = ary_ary_cartesian_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = ary_ary_cartesian_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = ary_ary_cartesian_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = ary_ary_cartesian_stft.get_embedded(
        embedding_grid=ary_ary_cartesian_stft.grid
    )
    assert embed.grid == ary_ary_cartesian_stft.grid
    embed = ary_ary_cartesian_stft.get_embedded(
        embedding_grid=ary_ary_cartesian_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == ary_ary_cartesian_stft.grid


def test_lin_lin_sparse_stft(
    lin_lin_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_lin_sparse_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_lin_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = lin_lin_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = lin_lin_sparse_stft.get_embedded(embedding_grid=lin_lin_sparse_stft.grid)
    assert embed.grid == lin_lin_sparse_stft.grid
    embed = lin_lin_sparse_stft.get_embedded(
        embedding_grid=lin_lin_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_lin_sparse_stft.grid


def test_lin_uni_sparse_stft(
    lin_uni_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[AnyArray]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_uni_sparse_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_uni_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = lin_uni_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = lin_uni_sparse_stft.get_embedded(embedding_grid=lin_uni_sparse_stft.grid)
    assert embed.grid == lin_uni_sparse_stft.grid
    embed = lin_uni_sparse_stft.get_embedded(
        embedding_grid=lin_uni_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_uni_sparse_stft.grid


def test_lin_ary_sparse_stft(
    lin_ary_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[AnyArray]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_ary_sparse_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_ary_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = lin_ary_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = lin_ary_sparse_stft.get_embedded(embedding_grid=lin_ary_sparse_stft.grid)
    assert embed.grid == lin_ary_sparse_stft.grid
    embed = lin_ary_sparse_stft.get_embedded(
        embedding_grid=lin_ary_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_ary_sparse_stft.grid


def test_uni_lin_sparse_stft(
    uni_lin_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = uni_lin_sparse_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = uni_lin_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = uni_lin_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = uni_lin_sparse_stft.get_embedded(embedding_grid=uni_lin_sparse_stft.grid)
    assert embed.grid == uni_lin_sparse_stft.grid
    embed = uni_lin_sparse_stft.get_embedded(
        embedding_grid=uni_lin_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == uni_lin_sparse_stft.grid


def test_uni_uni_sparse_stft(
    uni_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = uni_uni_sparse_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = uni_uni_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = uni_uni_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = uni_uni_sparse_stft.get_embedded(embedding_grid=uni_uni_sparse_stft.grid)
    assert embed.grid == uni_uni_sparse_stft.grid
    embed = uni_uni_sparse_stft.get_embedded(
        embedding_grid=uni_uni_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == uni_uni_sparse_stft.grid


def test_uni_ary_sparse_stft(
    uni_ary_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = uni_ary_sparse_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = uni_ary_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = uni_ary_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = uni_ary_sparse_stft.get_embedded(embedding_grid=uni_ary_sparse_stft.grid)
    assert embed.grid == uni_ary_sparse_stft.grid
    embed = uni_ary_sparse_stft.get_embedded(
        embedding_grid=uni_ary_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == uni_ary_sparse_stft.grid


def test_ary_lin_sparse_stft(
    ary_lin_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = ary_lin_sparse_stft.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = ary_lin_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = ary_lin_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = ary_lin_sparse_stft.get_embedded(embedding_grid=ary_lin_sparse_stft.grid)
    assert embed.grid == ary_lin_sparse_stft.grid
    embed = ary_lin_sparse_stft.get_embedded(
        embedding_grid=ary_lin_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == ary_lin_sparse_stft.grid


def test_ary_uni_sparse_stft(
    ary_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = ary_uni_sparse_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = ary_uni_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = ary_uni_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = ary_uni_sparse_stft.get_embedded(embedding_grid=ary_uni_sparse_stft.grid)
    assert embed.grid == ary_uni_sparse_stft.grid
    embed = ary_uni_sparse_stft.get_embedded(
        embedding_grid=ary_uni_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == ary_uni_sparse_stft.grid


def test_ary_ary_sparse_stft(
    ary_ary_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    subset = ary_ary_sparse_stft.get_subset(
        time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
    )
    assert len(subset.grid[0]) == 0
    assert len(subset.grid[1]) == 0
    subset = ary_ary_sparse_stft.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = ary_ary_sparse_stft.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = ary_ary_sparse_stft.get_embedded(embedding_grid=ary_ary_sparse_stft.grid)
    assert embed.grid == ary_ary_sparse_stft.grid
    embed = ary_ary_sparse_stft.get_embedded(
        embedding_grid=ary_ary_sparse_stft.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == ary_ary_sparse_stft.grid


def test_lin_lin_cartesian_wdm(
    lin_lin_cartesian_wdm: WDM[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_lin_cartesian_wdm.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_lin_cartesian_wdm.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    subset = lin_lin_cartesian_wdm.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 2, 2)
    embed = lin_lin_cartesian_wdm.get_embedded(
        embedding_grid=lin_lin_cartesian_wdm.grid
    )
    assert embed.grid == lin_lin_cartesian_wdm.grid
    embed = lin_lin_cartesian_wdm.get_embedded(
        embedding_grid=lin_lin_cartesian_wdm.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_lin_cartesian_wdm.grid


def test_lin_lin_sparse_wdm(
    lin_lin_sparse_wdm: WDM[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        _ = lin_lin_sparse_wdm.get_subset(
            time_interval=(0.1, 0.5), freq_interval=(0.1, 0.5)
        )
    subset = lin_lin_sparse_wdm.get_subset(
        time_interval=(1.0, 2.5), freq_interval=(1.0, 2.5)
    )
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert len(subset.grid.indices) == 1
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    subset = lin_lin_sparse_wdm.get_subset(slices=(slice(0, 2), slice(0, 2)))
    assert len(subset.grid[0]) == 2
    assert len(subset.grid[1]) == 2
    assert subset.entries.shape == (1, 1, 1, 1, 1)
    embed = lin_lin_sparse_wdm.get_embedded(embedding_grid=lin_lin_sparse_wdm.grid)
    assert embed.grid == lin_lin_sparse_wdm.grid
    embed = lin_lin_sparse_wdm.get_embedded(
        embedding_grid=lin_lin_sparse_wdm.grid,
        known_slices=(slice(0, 3), slice(0, 6)),
    )
    assert embed.grid == lin_lin_sparse_wdm.grid


# Functions


def test_densify_phasor(
    uni_ary_phasor: FrequencyPhasor[Axis[AnyArray]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[AnyArray],
):
    densify_phasor(
        uni_ary_phasor,
        linear_interpolator,
        dense_ary_freq_axis,
        embed=False,
    )
