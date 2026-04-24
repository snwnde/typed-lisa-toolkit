# pyright: reportPrivateUsage=false

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Any, TypedDict, cast

import numpy as np
import numpy.testing as npt
import pytest

from typed_lisa_toolkit import (
    axis,
    cast_mode,
    frequency_series,
    fsdata,
    harmonic_projected_waveform,
    harmonic_waveform,
    homogeneous_harmonic_projected_waveform,
    linspace_from_array,
    make_sdm,
    projected_waveform,
    stft,
    time_series,
    wdm,
    wdmdata,
)
from typed_lisa_toolkit.types import (
    STFT,
    WDM,
    AnyAxis,
    Array,
    Axis,
    EvolutionarySpectralDensity,
    FSData,
    Grid2D,
    Grid2DCartesian,
    Harmonic,
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    HomogeneousHarmonicProjectedWaveform,
    Linspace,
    ProjectedWaveform,
    SpectralDensity,
    TimeSeries,
    UniformFrequencySeries,
    UniformTimeSeries,
    WDMData,
    data,
)
from typed_lisa_toolkit.types import (
    representations as reps,
)
from typed_lisa_toolkit.types.representations import (
    _check_entry_grid_compatibility,
    _take_subset,
)

SEED = 11324214
rng = np.random.default_rng(SEED)


type EntryTransform = Callable[[Array], Array]
type ChannelSpec = tuple[Array, EntryTransform]
type HarmonicPhasorHandles = dict[Harmonic, dict[str, reps.Phasor[Axis[Array]]]]


class CanonicalRepresentationsResult[AT: AnyAxis](TypedDict):
    freqs: AT
    times: AT
    entries_fs: Array
    entries_ts: Array
    entries_tf: Array
    fs: reps.FrequencySeries[AT] | UniformFrequencySeries
    ts: TimeSeries[AT] | UniformTimeSeries
    tf: STFT[Grid2DCartesian[AT, AT]]


class FrequencySeriesBuildResult[AT: AnyAxis](TypedDict):
    fs: reps.FrequencySeries[AT] | UniformFrequencySeries
    frequencies: AT
    entries: Array


class StftCase(TypedDict):
    frequencies: Array
    times: Array
    entries: Array
    tf: STFT[Grid2DCartesian[AnyAxis, AnyAxis]]


class TsDataCaseActualResult(TypedDict):
    data: data.TSData


class TsDataCaseExpectedResult[AT: AnyAxis](TypedDict):
    times: AT
    x: Array
    y: Array
    z: Array


class TsDataCaseResult[AT: AnyAxis](TypedDict):
    actual: TsDataCaseActualResult
    expected: TsDataCaseExpectedResult[AT]


class FsDataCaseActualResult(TypedDict):
    data: FSData


class FsDataCaseExpectedResult[AT: AnyAxis](TypedDict):
    frequencies: AT
    x: Array
    y: Array
    z: Array


class FsDataCaseResult[AT: AnyAxis](TypedDict):
    actual: FsDataCaseActualResult
    expected: FsDataCaseExpectedResult[AT]


class FdPairActualResult(TypedDict):
    left: FSData
    right: FSData


class FdPairExpectedResult[AT: AnyAxis](TypedDict):
    frequencies: AT
    left_x: Array
    left_y: Array
    left_z: Array
    right_x: Array
    right_y: Array
    right_z: Array


class FdPairResult[AT: AnyAxis](TypedDict):
    actual: FdPairActualResult
    expected: FdPairExpectedResult[AT]


class WdmPairActualResult(TypedDict):
    left: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]
    right: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]


class WdmPairExpectedResult[AT: AnyAxis](TypedDict):
    times: AT
    frequencies: AT
    left_x: Array
    left_y: Array
    left_z: Array
    right_x: Array
    right_y: Array
    right_z: Array


class WdmPairResult[AT: AnyAxis](TypedDict):
    actual: WdmPairActualResult
    expected: WdmPairExpectedResult[AT]


class HarmonicWaveformFrequencySeriesActualResult(TypedDict):
    wf: HarmonicWaveform[Harmonic, reps.FrequencySeries[AnyAxis]]


class HarmonicWaveformFrequencySeriesExpectedResult(TypedDict):
    frequencies: Axis[Array]
    modes: tuple[Harmonic, Harmonic]
    mode_22: Harmonic
    mode_33: Harmonic
    wf_22: reps.FrequencySeries[AnyAxis]
    wf_33: reps.FrequencySeries[AnyAxis]


class HarmonicWaveformFrequencySeriesResult(TypedDict):
    actual: HarmonicWaveformFrequencySeriesActualResult
    expected: HarmonicWaveformFrequencySeriesExpectedResult


class HarmonicProjectedFrequencyWaveformActualResult[AT: AnyAxis](TypedDict):
    wf: HomogeneousHarmonicProjectedWaveform[Harmonic, reps.FrequencySeries[AT]]


class HarmonicProjectedFrequencyWaveformExpectedResult[AT: AnyAxis](TypedDict):
    frequencies: Axis[Array]
    mode_22: Harmonic
    mode_33: Harmonic
    resp_22: ProjectedWaveform[reps.FrequencySeries[AT]]
    resp_33: ProjectedWaveform[reps.FrequencySeries[AT]]
    resp_22_map: dict[str, reps.FrequencySeries[AT]]
    resp_33_map: dict[str, reps.FrequencySeries[AT]]


class HarmonicProjectedFrequencyWaveformResult[AT: AnyAxis](TypedDict):
    actual: HarmonicProjectedFrequencyWaveformActualResult[AT]
    expected: HarmonicProjectedFrequencyWaveformExpectedResult[AT]


build_harmonic_waveform_from_mapping = harmonic_waveform
build_projected_waveform_from_mapping = projected_waveform
build_harmonic_projected_waveform_from_mapping = harmonic_projected_waveform
build_homogeneous_harmonic_projected_waveform_from_mapping = (
    homogeneous_harmonic_projected_waveform
)


def _randn_array(xp: ModuleType, shape: tuple[int, ...]) -> Array:
    return xp.asarray(rng.standard_normal(shape))


def _canonicalize_1d_entries(values: Array) -> Array:
    """Convert 1D values to canonical entries shape: (B, C, H, F, N)."""
    return values[None, None, None, None, :]


def _canonicalize_2d_entries(values: Array) -> Array:
    """Convert 2D values to canonical entries shape: (B, C, H, F, Nf, Nt)."""
    return values[None, None, None, None, :, :]


# ============================================================================
# Axis builders
# ============================================================================


def _build_uniform_frequencies(xp: ModuleType):
    return axis(linspace_from_array(xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)))


def _build_frequencies(xp: ModuleType) -> Axis[Array]:
    return axis(xp.asarray([1.0, 2.0, 4.0], dtype=xp.float64))


# ============================================================================
# Grid builders
# ============================================================================


def _build_wdm_axes(xp: ModuleType):
    nt, nf = 20, 16
    dt = 0.12891289
    df = 1.0 / (2.0 * dt)
    times = axis(linspace_from_array(xp.asarray(dt * np.arange(nt), dtype=xp.float64)))
    frequencies = axis(
        linspace_from_array(xp.asarray(df * np.arange(nf), dtype=xp.float64))
    )
    return times, frequencies


# ============================================================================
# Representation builders
# ============================================================================


def _build_complex_entries(
    xp: ModuleType,
    values: Sequence[float | complex],
    *,
    random_scale: bool = False,
) -> Array:
    entries = _canonicalize_1d_entries(xp.asarray(values, dtype=xp.complex128))
    if random_scale:
        entries = entries * rng.standard_normal(entries.shape)
    return entries


# ============================================================================
# Waveform/Data builders
# ============================================================================


def _build_fsdata(
    frequencies: Axis[Linspace], channel_entries: Mapping[str, Array]
) -> FSData:
    return fsdata(
        {
            name: frequency_series(frequencies, entries)
            for name, entries in channel_entries.items()
        },
    )


def _build_wdmdata(
    times: Axis[Linspace],
    frequencies: Axis[Linspace],
    channel_entries: Mapping[str, Array],
) -> WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]:
    return wdmdata(
        {
            name: wdm(frequencies=frequencies, times=times, entries=entries)
            for name, entries in channel_entries.items()
        },
    )


def _stack_batched_entries(
    xp: ModuleType,
    base_entries: Array,
    variant_entries: Array,
) -> Array:
    return xp.concatenate([base_entries, variant_entries], axis=0)


def _build_3ch_kernel(
    xp: ModuleType,
    values_x: Array,
    values_y: Array,
    values_z: Array,
    offdiag_xy: Array,
    offdiag_xz: Array,
    offdiag_yz: Array,
) -> Array:
    row0 = xp.stack([values_x, offdiag_xy, offdiag_xz], axis=-1)
    row1 = xp.stack([offdiag_xy, values_y, offdiag_yz], axis=-1)
    row2 = xp.stack([offdiag_xz, offdiag_yz, values_z], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


def _build_batched_channel_entries(
    xp: ModuleType,
    channel_specs: Mapping[str, ChannelSpec],
) -> dict[str, Array]:
    return {
        name: _stack_batched_entries(xp, entries, transform(entries))
        for name, (entries, transform) in channel_specs.items()
    }


def build_canonical_representations(
    xp: ModuleType,
    *,
    n_batches: int,
    n_channels: int,
    n_harmonics: int,
    n_features: int,
    len_time: int,
    len_freq: int,
) -> CanonicalRepresentationsResult[Axis[Linspace]]:
    freqs = axis(linspace_from_array(xp.linspace(0, 1, len_freq)))
    times = axis(linspace_from_array(xp.linspace(0, 10, len_time)))

    entries_fs = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_freq),
    )
    fs = frequency_series(freqs, entries_fs)

    entries_ts = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_time),
    )
    ts = time_series(times, entries_ts)
    entries_tf = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_freq, len_time),
    )
    tf = stft(freqs, times, entries=entries_tf)

    return {
        "freqs": freqs,
        "times": times,
        "entries_fs": entries_fs,
        "entries_ts": entries_ts,
        "entries_tf": entries_tf,
        "fs": fs,
        "ts": ts,
        "tf": tf,
    }


def _build_stft_case(
    xp: ModuleType,
    *,
    frequencies: Array,
    times: Array,
    entries_shape: tuple[int, ...],
) -> StftCase:
    entries = _randn_array(xp, entries_shape)
    tf = stft(frequencies=frequencies, times=times, entries=entries)
    return {
        "frequencies": frequencies,
        "times": times,
        "entries": entries,
        "tf": tf,
    }


def build_stft_make_classmethod_case(xp: ModuleType) -> StftCase:
    times = xp.asarray(np.linspace(0.0, 10.0, 100), dtype=xp.float64)
    frequencies = xp.asarray(np.linspace(0.0, 1.0, 50), dtype=xp.float64)
    return _build_stft_case(
        xp,
        frequencies=frequencies,
        times=times,
        entries_shape=(1, 1, 1, 1, 50, 100),
    )


def build_stft_times_and_frequencies_case(xp: ModuleType) -> StftCase:
    frequencies = xp.asarray(np.linspace(0.0, 1.0, 50), dtype=xp.float64)
    times = xp.asarray(np.linspace(0.0, 10.0, 100), dtype=xp.float64)
    return _build_stft_case(
        xp,
        frequencies=frequencies,
        times=times,
        entries_shape=(1, 1, 1, 1, 50, 100),
    )


def build_freq_series(
    xp: ModuleType,
    *,
    uniform: bool = True,
) -> FrequencySeriesBuildResult[AnyAxis]:
    frequencies = _build_uniform_frequencies(xp) if uniform else _build_frequencies(xp)
    _entries = _build_complex_entries(
        xp,
        [1.0 + 0.5j, -1.0j, 2.0 + 0.0j],
        random_scale=True,
    )
    return {
        "fs": frequency_series(frequencies, _entries),
        "frequencies": frequencies,
        "entries": _entries,
    }


def build_fdata(xp: ModuleType) -> FSData:
    frequencies = _build_uniform_frequencies(xp)
    x_entries = _build_complex_entries(xp, [1.0 + 0.5j, -1.0j, 2.0 + 0.0j])
    y_entries = _build_complex_entries(xp, [0.5 - 0.25j, -1.0 + 0.25j, 2.0 + 0.5j])
    z_entries = _build_complex_entries(xp, [0.2 + 0.75j, -0.5 - 0.25j, 1.25 + 0.1j])
    return _build_fsdata(frequencies, {"X": x_entries, "Y": y_entries, "Z": z_entries})


def build_tsdata_case(xp: ModuleType) -> TsDataCaseResult[Axis[Linspace]]:
    times = axis(
        linspace_from_array(
            xp.asarray(np.linspace(0.0, 3.0, 8), dtype=xp.float64),
        ),
    )
    x = xp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25, 0.75, 0.0], dtype=xp.float64)
    y = xp.asarray([1.0, 0.0, -0.5, 0.25, 0.5, -0.75, 0.0, 1.0], dtype=xp.float64)
    z = xp.asarray([0.5, -0.25, 0.25, -0.75, 0.0, 0.5, -0.5, 0.25], dtype=xp.float64)

    built = data.tsdata(
        {
            "X": time_series(times, x[None, None, None, None, :]),
            "Y": time_series(times, y[None, None, None, None, :]),
            "Z": time_series(times, z[None, None, None, None, :]),
        },
    )
    return {
        "actual": {"data": built},
        "expected": {"times": times, "x": x, "y": y, "z": z},
    }


def build_xyz_tsdata_case(
    xp: ModuleType,
    *,
    n: int = 8,
) -> TsDataCaseResult[Axis[Linspace]]:
    times = axis(
        linspace_from_array(
            xp.asarray(np.linspace(0.0, 3.5, n), dtype=xp.float64),
        ),
    )
    x = xp.asarray([0.0, 1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.1], dtype=xp.float64)
    y = xp.asarray([1.0, -0.5, 0.25, 0.0, 0.4, -0.2, 0.6, -0.8], dtype=xp.float64)
    z = xp.asarray([-0.2, 0.3, -0.1, 0.5, -0.7, 0.9, -0.4, 0.2], dtype=xp.float64)

    built = data.tsdata(
        {
            "X": time_series(times, x[None, None, None, None, :]),
            "Y": time_series(times, y[None, None, None, None, :]),
            "Z": time_series(times, z[None, None, None, None, :]),
        },
    )
    return {
        "actual": {"data": built},
        "expected": {"times": times, "x": x, "y": y, "z": z},
    }


def build_xyz_spectral_density(xp: ModuleType) -> SpectralDensity:
    frequencies = xp.asarray([0.25, 0.5, 0.75], dtype=xp.float64)
    inverse_sdm = xp.broadcast_to(
        xp.asarray(
            [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]],
            dtype=xp.float64,
        ),
        (len(frequencies), 3, 3),
    )
    return make_sdm(inverse_sdm, frequencies=frequencies, channel_names=("X", "Y", "Z"))


def build_xyz_evolutionary_spectral_density(
    xp: ModuleType,
) -> EvolutionarySpectralDensity:
    frequencies = xp.asarray([0.25, 0.5], dtype=xp.float64)
    times = cast("Array", xp.asarray([0.0, 1.0], dtype=xp.float64))
    base = xp.asarray(
        [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]],
        dtype=xp.float64,
    )
    fi = xp.arange(len(frequencies), dtype=xp.float64)[:, None, None, None]
    ti = xp.arange(len(times), dtype=xp.float64)[None, :, None, None]
    inverse_esdm = base[None, None, :, :] * (1.0 + 0.1 * fi + 0.05 * ti)
    return make_sdm(
        inverse_esdm,
        frequencies=frequencies,
        times=times,
        channel_names=("X", "Y", "Z"),
    )


def build_fd_template_band_case(xp: ModuleType) -> FsDataCaseResult[Axis[Linspace]]:
    frequencies = axis(
        linspace_from_array(xp.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=xp.float64)),
    )
    x = xp.asarray(
        [1.0 + 0.0j, 0.5 + 0.1j, 2.0 - 0.2j, 1.0 + 0.5j, 0.1 + 0.0j],
        dtype=xp.complex128,
    )
    y = xp.asarray(
        [0.25 + 0.0j, -0.5 + 0.25j, 1.0 + 0.0j, 0.5 - 0.25j, -0.1 + 0.0j],
        dtype=xp.complex128,
    )
    z = xp.asarray(
        [0.1 + 0.2j, -0.3 + 0.1j, 0.7 - 0.4j, 0.2 + 0.0j, -0.05 + 0.05j],
        dtype=xp.complex128,
    )
    built = _build_fsdata(
        frequencies,
        {
            "X": x[None, None, None, None, :],
            "Y": y[None, None, None, None, :],
            "Z": z[None, None, None, None, :],
        },
    )
    return {
        "actual": {"data": built},
        "expected": {"frequencies": frequencies, "x": x, "y": y, "z": z},
    }


def build_fd_linspace_noise_case(xp: ModuleType) -> FsDataCaseResult[Axis[Linspace]]:
    times = linspace_from_array(xp.asarray(np.linspace(0.0, 7.0, 8), dtype=xp.float64))
    frequencies = axis(
        linspace_from_array(
            xp.asarray(np.fft.rfftfreq(len(times), d=times.step), dtype=xp.float64),
        ),
    )
    x = xp.asarray(
        [1.0 + 0.0j, 0.5 + 0.25j, -0.25 + 0.5j, 0.1 - 0.2j, 0.05 + 0.0j],
        dtype=xp.complex128,
    )
    y = xp.asarray(
        [0.5 + 0.0j, -0.2 + 0.1j, 0.3 - 0.4j, -0.1 + 0.2j, 0.01 + 0.0j],
        dtype=xp.complex128,
    )
    z = xp.asarray(
        [0.2 + 0.1j, -0.1 + 0.0j, 0.4 - 0.2j, 0.05 + 0.15j, -0.02 + 0.0j],
        dtype=xp.complex128,
    )
    built = _build_fsdata(
        frequencies,
        {
            "X": x[None, None, None, None, :],
            "Y": y[None, None, None, None, :],
            "Z": z[None, None, None, None, :],
        },
    ).set_times(times)
    return {
        "actual": {"data": built},
        "expected": {"frequencies": frequencies, "x": x, "y": y, "z": z},
    }


def build_wdm(xp: ModuleType) -> WDM[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]:
    times, frequencies = _build_wdm_axes(xp)
    nt, nf = len(times), len(frequencies)
    entries = _randn_array(xp, (1, 1, 1, 1, nf, nt))
    return wdm(frequencies=frequencies, times=times, entries=entries)


def build_fd_pair(xp: ModuleType) -> FdPairResult[Axis[Linspace]]:
    frequencies = _build_uniform_frequencies(xp)
    left_x = _build_complex_entries(xp, [1.0 + 1.0j, 2.0 - 1.0j, -1.0 + 0.5j])
    left_y = _build_complex_entries(xp, [0.5 - 0.25j, -1.0j, 2.0 + 0.0j])
    left_z = _build_complex_entries(xp, [-0.75 + 0.3j, 0.2 - 1.2j, 1.1 + 0.4j])
    right_x = _build_complex_entries(xp, [2.0 - 1.0j, -1.0 + 2.0j, 0.5 + 0.25j])
    right_y = _build_complex_entries(xp, [1.0 + 0.0j, 0.25 + 1.0j, -0.5 + 2.0j])
    right_z = _build_complex_entries(xp, [0.3 + 0.4j, -1.5 + 0.8j, 0.75 - 0.6j])

    left = _build_fsdata(frequencies, {"X": left_x, "Y": left_y, "Z": left_z})
    right = _build_fsdata(frequencies, {"X": right_x, "Y": right_y, "Z": right_z})

    return {
        "actual": {"left": left, "right": right},
        "expected": {
            "frequencies": frequencies,
            "left_x": left_x,
            "left_y": left_y,
            "left_z": left_z,
            "right_x": right_x,
            "right_y": right_y,
            "right_z": right_z,
        },
    }


def build_wdm_pair(xp: ModuleType) -> WdmPairResult[Axis[Linspace]]:
    times, frequencies = _build_wdm_axes(xp)
    nt, nf = len(times), len(frequencies)

    left_x = _canonicalize_2d_entries(
        xp.outer(xp.cos(frequencies.asarray(xp)), xp.sin(times.asarray(xp)))
    )
    left_y = _canonicalize_2d_entries(xp.ones((nf, nt), dtype=xp.float64))
    left_z = _canonicalize_2d_entries(
        xp.outer(xp.sin(frequencies.asarray(xp)), xp.cos(times.asarray(xp)))
    )
    right_x = left_x
    right_y = left_y
    right_z = left_z

    left = _build_wdmdata(times, frequencies, {"X": left_x, "Y": left_y, "Z": left_z})
    right = _build_wdmdata(
        times,
        frequencies,
        {"X": right_x, "Y": right_y, "Z": right_z},
    )

    return {
        "actual": {"left": left, "right": right},
        "expected": {
            "times": times,
            "frequencies": frequencies,
            "left_x": left_x,
            "left_y": left_y,
            "left_z": left_z,
            "right_x": right_x,
            "right_y": right_y,
            "right_z": right_z,
        },
    }


def build_fd_pair_batched(xp: ModuleType) -> FdPairResult[Axis[Linspace]]:
    base = build_fd_pair(xp)
    base_expected = base["expected"]
    frequencies = base_expected["frequencies"]

    left_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (
                base_expected["left_x"],
                lambda entries: 0.75 * entries + (0.1 - 0.2j),
            ),
            "Y": (
                base_expected["left_y"],
                lambda entries: 1.25 * entries + (-0.05 + 0.1j),
            ),
            "Z": (
                base_expected["left_z"],
                lambda entries: -0.85 * entries + (0.15 - 0.25j),
            ),
        },
    )
    right_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (
                base_expected["right_x"],
                lambda entries: -0.5 * entries + (0.2 + 0.05j),
            ),
            "Y": (
                base_expected["right_y"],
                lambda entries: 0.6 * entries + (-0.1 + 0.15j),
            ),
            "Z": (
                base_expected["right_z"],
                lambda entries: 1.1 * entries + (-0.2 - 0.05j),
            ),
        },
    )

    left = _build_fsdata(
        frequencies,
        {name: entries for name, entries in left_entries.items()},
    )
    right = _build_fsdata(
        frequencies,
        {name: entries for name, entries in right_entries.items()},
    )

    return {
        "actual": {"left": left, "right": right},
        "expected": {
            "frequencies": frequencies,
            "left_x": left_entries["X"],
            "left_y": left_entries["Y"],
            "left_z": left_entries["Z"],
            "right_x": right_entries["X"],
            "right_y": right_entries["Y"],
            "right_z": right_entries["Z"],
        },
    }


def build_wdm_pair_batched(xp: ModuleType) -> WdmPairResult[Axis[Linspace]]:
    base = build_wdm_pair(xp)
    base_expected = base["expected"]
    times = base_expected["times"]
    frequencies = base_expected["frequencies"]

    left_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (base_expected["left_x"], lambda entries: 0.8 * entries + 0.3),
            "Y": (base_expected["left_y"], lambda entries: 1.1 * entries - 0.2),
            "Z": (base_expected["left_z"], lambda entries: -0.6 * entries + 0.4),
        },
    )
    right_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (base_expected["right_x"], lambda entries: -0.4 * entries + 0.5),
            "Y": (base_expected["right_y"], lambda entries: 0.7 * entries + 0.25),
            "Z": (base_expected["right_z"], lambda entries: 1.2 * entries - 0.35),
        },
    )

    left = _build_wdmdata(
        times,
        frequencies,
        {name: entries for name, entries in left_entries.items()},
    )
    right = _build_wdmdata(
        times,
        frequencies,
        {name: entries for name, entries in right_entries.items()},
    )

    return {
        "actual": {"left": left, "right": right},
        "expected": {
            "times": times,
            "frequencies": frequencies,
            "left_x": left_entries["X"],
            "left_y": left_entries["Y"],
            "left_z": left_entries["Z"],
            "right_x": right_entries["X"],
            "right_y": right_entries["Y"],
            "right_z": right_entries["Z"],
        },
    }


# ============================================================================
# Noisemodel builders
# ============================================================================


def diagonal_kernel_3ch(xp: ModuleType) -> Array:
    values_x = xp.asarray([2.0, 4.0, 8.0], dtype=xp.float64)
    values_y = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    values_z = xp.asarray([0.75, 1.25, 2.0], dtype=xp.float64)
    offdiag_xy = xp.zeros_like(values_x)
    offdiag_xz = xp.zeros_like(values_x)
    offdiag_yz = xp.zeros_like(values_x)
    return _build_3ch_kernel(
        xp,
        values_x,
        values_y,
        values_z,
        offdiag_xy,
        offdiag_xz,
        offdiag_yz,
    )


def dense_kernel_3ch(xp: ModuleType) -> Array:
    values_x = xp.asarray([2.0, 4.0, 8.0], dtype=xp.float64)
    values_y = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    values_z = xp.asarray([1.1, 0.9, 1.4], dtype=xp.float64)
    offdiag_xy = xp.asarray([0.1, 0.2, -0.3], dtype=xp.float64)
    offdiag_xz = xp.asarray([-0.05, 0.15, 0.2], dtype=xp.float64)
    offdiag_yz = xp.asarray([0.08, -0.04, 0.12], dtype=xp.float64)
    return _build_3ch_kernel(
        xp,
        values_x,
        values_y,
        values_z,
        offdiag_xy,
        offdiag_xz,
        offdiag_yz,
    )


def dense_esdm_3ch(xp: ModuleType) -> Array:
    times, frequencies = _build_wdm_axes(xp)
    n_freq = len(frequencies)
    n_time = len(times)
    fi = xp.arange(n_freq, dtype=xp.float64)[:, None]
    ti = xp.arange(n_time, dtype=xp.float64)[None, :]

    a = 1.8 + 0.05 * xp.cos(0.3 * fi) + 0.03 * xp.sin(0.2 * ti)
    d = 1.2 + 0.04 * xp.sin(0.25 * fi) + 0.02 * xp.cos(0.35 * ti)
    g = 1.5 + 0.03 * xp.cos(0.18 * fi) + 0.04 * xp.sin(0.12 * ti)
    b = 0.08 * xp.cos(0.15 * fi + 0.1 * ti)
    c = 0.06 * xp.sin(0.21 * fi - 0.07 * ti)
    e = 0.05 * xp.cos(0.11 * fi + 0.09 * ti)

    row0 = xp.stack([a, b, c], axis=-1)
    row1 = xp.stack([b, d, e], axis=-1)
    row2 = xp.stack([c, e, g], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


# ============================================================================
# Representation behavior test helpers
# ============================================================================


type HelperNoArgs = Callable[[], None]
type BackendHelper = Callable[[ModuleType], None]
type WdmHelper = Callable[[WDM[Grid2D[Axis[Linspace], Axis[Linspace]]]], None]
type StftHelper = Callable[[StftCase], None]


def _test_linspace_shape_property() -> None:
    ls = Linspace(1.0, 0.5, 4)
    assert ls.shape == (4,)


def _test_linspace_stop_property() -> None:
    ls = Linspace(1.0, 0.5, 4)
    assert ls.stop == pytest.approx(2.5)  # pyright: ignore[reportUnknownMemberType]


def _test_linspace_eq_raises_for_non_linspacelike() -> None:
    ls = Linspace(0.0, 1.0, 5)
    with pytest.raises(TypeError):
        ls.__eq__(42)


def _test_linspace_eq_returns_false_for_step_mismatch() -> None:
    ls1 = Linspace(0.0, 1.0, 5)
    ls2 = Linspace(0.0, 2.0, 5)
    assert ls1 != ls2


def _test_linspace_array_with_copy_false() -> None:
    ls = Linspace(0.0, 1.0, 5)
    arr = np.array(ls, copy=False)
    npt.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0, 4.0])


def _test_linspace_getitem_invalid_type_raises() -> None:
    ls = Linspace(0.0, 1.0, 10)
    with pytest.raises(TypeError):
        ls["bad"]  # pyright: ignore[reportCallIssue, reportArgumentType]


def _test_check_entry_grid_compatibility_raises_on_mismatch(xp: ModuleType) -> None:
    grid = (xp.asarray(np.linspace(0, 1, 10)),)
    entries = xp.asarray(np.ones((1, 1, 1, 1, 20)))
    with pytest.raises(ValueError, match=r".+"):
        _check_entry_grid_compatibility(grid, entries)


def _test_take_subset_slice_dimension_mismatch_raises(xp: ModuleType) -> None:
    grid = (xp.asarray(np.linspace(0, 1, 10)),)
    entries = xp.asarray(np.ones((1, 1, 1, 1, 10)))
    with pytest.raises(ValueError, match=r".+"):
        _take_subset(grid, entries, (slice(0, 5), slice(0, 3)))


def _test_take_subset_with_array_grid(xp: ModuleType) -> None:
    grid = (xp.asarray(np.linspace(0.0, 1.0, 20)),)
    entries = xp.asarray(np.arange(20, dtype=float))[None, None, None, None, :]
    new_grid, new_entries = _take_subset(grid, entries, (slice(5, 10),))
    npt.assert_allclose(np.asarray(new_grid[0]), np.asarray(grid[0][5:10]))
    npt.assert_allclose(
        np.asarray(new_entries[0, 0, 0, 0, :]),
        np.asarray(entries[0, 0, 0, 0, 5:10]),
    )


def _test_non_uniform_grid_stays_array(xp: ModuleType) -> None:
    non_uniform = cast("Array", xp.asarray(np.array([0.0, 1.0, 3.0, 7.0])))
    ts = time_series(times=non_uniform, entries=xp.ones((1, 1, 1, 1, 4)))
    assert not isinstance(ts.grid[0], Linspace)


def _test_axis_onset_and_end_from_plain_arrays(xp: ModuleType) -> None:
    freqs = cast("Array", xp.asarray(np.array([0.01, 0.02, 0.03, 0.04])))
    fs = frequency_series(freqs, entries=xp.asarray(xp.ones((1, 1, 1, 1, 4))))
    assert fs.f_min == pytest.approx(0.01)  # pyright: ignore[reportUnknownMemberType]
    assert fs.f_max == pytest.approx(0.04)  # pyright: ignore[reportUnknownMemberType]


def _test_frequency_series_get_time_shifted(xp: ModuleType) -> None:
    n, dt = 32, 1.0 / 128
    freqs = axis(linspace_from_array(xp.asarray(np.fft.rfftfreq(n, d=dt))))
    entries_fs = xp.asarray(np.fft.rfft(np.sin(2 * np.pi * np.arange(n) * dt)))[
        None,
        None,
        None,
        None,
        :,
    ]
    fs = frequency_series(freqs, entries_fs)
    shifted = fs.get_time_shifted(2 * dt)
    assert isinstance(shifted, UniformFrequencySeries)
    assert shifted.entries.shape == fs.entries.shape


def _test_frequency_series_angle(xp: ModuleType) -> None:
    freqs = axis(linspace_from_array(xp.asarray(np.linspace(1e-4, 1e-2, 10))))
    z = xp.asarray(np.exp(1j * np.linspace(0, 4 * np.pi, 10)))[
        None,
        None,
        None,
        None,
        :,
    ]
    fs = frequency_series(freqs, z)
    angles = fs.angle()
    assert isinstance(angles, UniformFrequencySeries)
    assert angles.entries.shape == fs.entries.shape


def _test_stft_make_classmethod(stft_case: StftCase) -> None:
    tf = stft_case["tf"]
    assert isinstance(tf, STFT)
    npt.assert_allclose(np.array(tf.grid[1]), stft_case["times"], rtol=1e-10)
    npt.assert_allclose(np.array(tf.grid[0]), stft_case["frequencies"], rtol=1e-10)


def _test_stft_times_and_frequencies_properties(stft_case: StftCase) -> None:
    tf = stft_case["tf"]
    npt.assert_allclose(np.array(tf.times), stft_case["times"])
    npt.assert_allclose(np.array(tf.frequencies), stft_case["frequencies"])


def _test_series_repr_and_grid_shape(xp: ModuleType) -> None:
    freqs = Linspace(0.0, 1e-3, 20)
    entries = xp.asarray(np.ones((1, 1, 1, 1, 20)))
    fs = frequency_series(freqs, entries)
    r = repr(fs)
    assert "UniformFrequencySeries" in r


def _test_wdm_nd_duration_sample_interval(
    wdm: WDM[Grid2D[Axis[Linspace], Axis[Linspace]]],
) -> None:
    assert wdm.Nf * wdm.Nt == wdm.ND
    assert wdm.dt == pytest.approx(wdm.sample_interval)  # pyright: ignore[reportUnknownMemberType]


def _test_wdm_df_shape_sample_rate_nyquist(
    wdm: WDM[Grid2D[Axis[Linspace], Axis[Linspace]]],
) -> None:
    assert wdm.shape == (wdm.Nf, wdm.Nt)


def _test_wdm_is_critically_sampled(
    wdm: WDM[Grid2D[Axis[Linspace], Axis[Linspace]]],
) -> None:
    result = wdm.is_critically_sampled()
    expected = bool(np.isclose(wdm.dT * wdm.dF, 0.5))
    assert bool(result) == expected


def _test_wdm_get_subset_time(
    wdm: WDM[Grid2D[Axis[Linspace], Axis[Linspace]]],
) -> None:
    times_arr = np.asarray(wdm.times)
    t_mid = float(times_arr[len(times_arr) // 2])
    sub = wdm.get_subset(time_interval=(float(times_arr[0]), t_mid))
    assert isinstance(sub, WDM)
    assert sub.Nt < wdm.Nt


def _test_wdm_get_subset_freq(
    wdm: WDM[Grid2D[Axis[Linspace], Axis[Linspace]]],
) -> None:
    freqs_arr = np.asarray(wdm.frequencies)
    f_mid = float(freqs_arr[len(freqs_arr) // 2])
    sub = wdm.get_subset(freq_interval=(float(freqs_arr[0]), f_mid))
    assert isinstance(sub, WDM)
    assert sub.Nf < wdm.Nf


def _test_data_base_get_plotter_notimplemented() -> None:
    class Dummy(data.Data[TimeSeries[AnyAxis]]):
        _REP_TYPE = TimeSeries[AnyAxis]  # pyright: ignore[reportUnannotatedClassAttribute]

        @property
        def kind(self):
            return None

    times = np.linspace(0.0, 1.0, 4)
    representation = time_series(times=times, entries=np.ones((1, 1, 1, 1, 4)))
    dummy = Dummy.from_dict({"X": representation})

    with pytest.raises(AttributeError):
        dummy._get_plotter()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]


_LINSPACE_HELPERS: dict[str, HelperNoArgs] = {
    "test_shape_property": _test_linspace_shape_property,
    "test_stop_property": _test_linspace_stop_property,
    "test_eq_raises_for_non_linspacelike": (
        _test_linspace_eq_raises_for_non_linspacelike
    ),
    "test_eq_returns_false_for_step_mismatch": (
        _test_linspace_eq_returns_false_for_step_mismatch
    ),
    "test_array_with_copy_false": _test_linspace_array_with_copy_false,
    "test_getitem_invalid_type_raises": _test_linspace_getitem_invalid_type_raises,
}

_REPRESENTATION_HELPERS: dict[str, BackendHelper] = {
    "test_check_entry_grid_compatibility_raises_on_mismatch": (
        _test_check_entry_grid_compatibility_raises_on_mismatch
    ),
    "test_take_subset_slice_dimension_mismatch_raises": (
        _test_take_subset_slice_dimension_mismatch_raises
    ),
    "test_take_subset_with_array_grid": _test_take_subset_with_array_grid,
    "test_non_uniform_grid_stays_array": _test_non_uniform_grid_stays_array,
    "test_axis_onset_and_end_from_plain_arrays": (
        _test_axis_onset_and_end_from_plain_arrays
    ),
}

_ADVANCED_REPRESENTATION_HELPERS: dict[str, BackendHelper | StftHelper] = {
    "test_frequency_series_get_time_shifted": _test_frequency_series_get_time_shifted,
    "test_frequency_series_angle": _test_frequency_series_angle,
    "test_stft_make_classmethod": _test_stft_make_classmethod,
    "test_stft_times_and_frequencies_properties": (
        _test_stft_times_and_frequencies_properties
    ),
    "test_series_repr_and_grid_shape": _test_series_repr_and_grid_shape,
}

_WDM_HELPERS: dict[str, WdmHelper] = {
    "test_nd_duration_sample_interval": _test_wdm_nd_duration_sample_interval,
    "test_df_shape_sample_rate_nyquist": _test_wdm_df_shape_sample_rate_nyquist,
    "test_is_critically_sampled": _test_wdm_is_critically_sampled,
    "test_get_subset_time": _test_wdm_get_subset_time,
    "test_get_subset_freq": _test_wdm_get_subset_freq,
}


@pytest.fixture
def data_abstract_branch_helpers() -> dict[str, HelperNoArgs]:
    return {
        "test_data_base_get_plotter_notimplemented": (
            _test_data_base_get_plotter_notimplemented
        ),
    }


@pytest.fixture
def linspace_helpers() -> dict[str, HelperNoArgs]:
    return _LINSPACE_HELPERS


@pytest.fixture
def representation_helpers() -> dict[str, BackendHelper]:
    return _REPRESENTATION_HELPERS


@pytest.fixture
def advanced_representation_helpers() -> dict[str, BackendHelper | StftHelper]:
    return _ADVANCED_REPRESENTATION_HELPERS


@pytest.fixture
def wdm_helpers() -> dict[str, WdmHelper]:
    return _WDM_HELPERS


# ============================================================================
# Likelihood-adjacent waveform helpers
# ============================================================================


# ============================================================================
# Waveform Helper Functions (real phasor objects)
# ============================================================================


def _as_frequency_axis(frequencies: Any = None) -> Axis[Array]:
    if frequencies is None:
        return axis(np.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=np.float64))
    if isinstance(frequencies, Axis):
        return cast("Axis[Array]", frequencies)
    return axis(np.asarray(frequencies, dtype=np.float64))


def build_test_phasor(
    *,
    f_min: float,
    f_max: float,
    frequencies: Any = None,
    amplitude_scale: float = 1.0,
    phase_shift: float = 0.0,
) -> reps.Phasor[Axis[Array]]:
    """Build a compact 1D phasor with support restricted to [f_min, f_max]."""
    freq_axis = _as_frequency_axis(frequencies)
    freq_values = np.asarray(freq_axis.asarray(np), dtype=np.float64)
    mask = np.logical_and(freq_values >= f_min, freq_values <= f_max)
    support = freq_values[mask]
    if support.size == 0:
        msg = "No frequency support in requested [f_min, f_max] interval."
        raise ValueError(msg)

    amplitudes = amplitude_scale * (
        1.0 + 0.1 * np.arange(support.size, dtype=np.float64)
    )
    phases = phase_shift + 0.2 * np.arange(support.size, dtype=np.float64)
    return reps.phasor(
        axis(support),
        amplitudes=amplitudes.astype(np.complex128),
        phases=phases,
    )


def build_harmonic_projected_phasor_waveform(
    *,
    frequencies: Any = None,
) -> tuple[HarmonicProjectedWaveform[Harmonic, Any], HarmonicPhasorHandles]:
    """Build a two-mode HarmonicProjectedWaveform with real phasor leaves."""
    frequency_axis = _as_frequency_axis(frequencies)
    mode_22 = cast_mode((2, 2))
    mode_33 = cast_mode((3, 3))

    p22x = build_test_phasor(
        f_min=1.0,
        f_max=3.0,
        frequencies=frequency_axis,
        amplitude_scale=1.0,
        phase_shift=0.0,
    )
    p22y = build_test_phasor(
        f_min=1.0,
        f_max=3.0,
        frequencies=frequency_axis,
        amplitude_scale=0.9,
        phase_shift=0.1,
    )
    p22z = build_test_phasor(
        f_min=1.0,
        f_max=3.0,
        frequencies=frequency_axis,
        amplitude_scale=1.1,
        phase_shift=-0.05,
    )
    p33x = build_test_phasor(
        f_min=0.5,
        f_max=2.0,
        frequencies=frequency_axis,
        amplitude_scale=0.8,
        phase_shift=0.2,
    )
    p33y = build_test_phasor(
        f_min=0.5,
        f_max=2.0,
        frequencies=frequency_axis,
        amplitude_scale=1.2,
        phase_shift=-0.15,
    )
    p33z = build_test_phasor(
        f_min=0.5,
        f_max=2.0,
        frequencies=frequency_axis,
        amplitude_scale=1.05,
        phase_shift=0.05,
    )

    wf = harmonic_projected_waveform(
        {
            mode_22: projected_waveform({"X": p22x, "Y": p22y, "Z": p22z}),
            mode_33: projected_waveform({"X": p33x, "Y": p33y, "Z": p33z}),
        },
    )

    handles = {
        mode_22: {
            "X": p22x,
            "Y": p22y,
            "Z": p22z,
        },
        mode_33: {
            "X": p33x,
            "Y": p33y,
            "Z": p33z,
        },
    }
    return wf, handles


def build_fake_harmonic_projected_waveform() -> tuple[
    HarmonicProjectedWaveform[Harmonic, Any], HarmonicPhasorHandles
]:
    """Backward-compatible alias for tests using the old helper name."""
    return build_harmonic_projected_phasor_waveform()


def _make_fs[AT: AnyAxis](
    xp: ModuleType,
    frequencies: AT,
    values: Sequence[float | complex],
) -> reps.FrequencySeries[AT]:
    entries = xp.asarray(values, dtype=xp.complex128)[None, None, None, None, :]
    return frequency_series(frequencies, entries)


def build_harmonic_waveform_frequency_series(
    xp: ModuleType,
) -> HarmonicWaveformFrequencySeriesResult:
    frequencies = axis(xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64))

    mode_22 = cast_mode((2, 2))
    mode_33 = cast_mode((3, 3))

    wf_22 = _make_fs(xp, frequencies, [1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j])
    wf_33 = _make_fs(xp, frequencies, [-0.5 + 1.0j, 0.25 + 0.0j, 1.5 - 0.25j])

    wf = harmonic_waveform({mode_22: wf_22, mode_33: wf_33})

    return {
        "actual": {"wf": wf},
        "expected": {
            "frequencies": frequencies,
            "modes": (mode_22, mode_33),
            "mode_22": mode_22,
            "mode_33": mode_33,
            "wf_22": wf_22,
            "wf_33": wf_33,
        },
    }


def build_harmonic_projected_frequency_waveform(
    xp: ModuleType,
) -> HarmonicProjectedFrequencyWaveformResult[Axis[Linspace]]:
    frequencies = axis(xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64))
    _freqs = axis(linspace_from_array(frequencies.ax))

    mode_22 = cast_mode((2, 2))
    mode_33 = cast_mode((3, 3))

    resp_22_map = {
        "X": _make_fs(xp, _freqs, [1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j]),
        "Y": _make_fs(xp, _freqs, [0.5 + 0.25j, -1.0 + 0.0j, 0.25 - 0.25j]),
        "Z": _make_fs(xp, _freqs, [0.3 - 0.1j, 0.8 + 0.4j, -0.2 + 0.9j]),
    }
    resp_33_map = {
        "X": _make_fs(xp, _freqs, [0.2 + 0.0j, -0.5 + 1.0j, 0.1 - 0.2j]),
        "Y": _make_fs(xp, _freqs, [1.0 + 0.0j, 1.5 + 0.0j, 2.0 + 0.0j]),
        "Z": _make_fs(xp, _freqs, [0.4 + 0.2j, -0.7 + 0.1j, 1.1 - 0.3j]),
    }

    resp_22 = projected_waveform(resp_22_map)
    resp_33 = projected_waveform(resp_33_map)

    wf = homogeneous_harmonic_projected_waveform({mode_22: resp_22, mode_33: resp_33})

    return {
        "actual": {"wf": wf},
        "expected": {
            "frequencies": frequencies,
            "mode_22": mode_22,
            "mode_33": mode_33,
            "resp_22": resp_22,
            "resp_33": resp_33,
            "resp_22_map": resp_22_map,
            "resp_33_map": resp_33_map,
        },
    }


def build_harmonic_waveform_constructor(
    xp: ModuleType,
) -> HarmonicWaveform[Harmonic, reps.FrequencySeries[AnyAxis]]:
    case = build_harmonic_waveform_frequency_series(xp)
    expected = case["expected"]
    return harmonic_waveform(
        {
            expected["mode_22"]: expected["wf_22"],
            expected["mode_33"]: expected["wf_33"],
        },
    )


def build_projected_waveform_constructor(
    xp: ModuleType,
) -> ProjectedWaveform[reps.FrequencySeries[Axis[Linspace]]]:
    case = build_harmonic_projected_frequency_waveform(xp)
    return projected_waveform(case["expected"]["resp_22_map"])


def build_harmonic_projected_waveform_constructor(
    xp: ModuleType,
) -> HarmonicProjectedWaveform[Harmonic, reps.FrequencySeries[Axis[Linspace]]]:
    case = build_harmonic_projected_frequency_waveform(xp)
    expected = case["expected"]
    return harmonic_projected_waveform(
        {
            expected["mode_22"]: projected_waveform(expected["resp_22_map"]),
            expected["mode_33"]: projected_waveform(expected["resp_33_map"]),
        },
    )


def build_homogeneous_harmonic_projected_waveform_constructor(
    xp: ModuleType,
) -> HomogeneousHarmonicProjectedWaveform[
    Harmonic, reps.FrequencySeries[Axis[Linspace]]
]:
    case = build_harmonic_projected_frequency_waveform(xp)
    expected = case["expected"]
    return homogeneous_harmonic_projected_waveform(
        {
            expected["mode_22"]: projected_waveform(expected["resp_22_map"]),
            expected["mode_33"]: projected_waveform(expected["resp_33_map"]),
        },
    )


@pytest.fixture(autouse=True)
def _inject_builder_globals(
    request: pytest.FixtureRequest,
):
    module_dict = cast("dict[str, Any]", request.module.__dict__)  # pyright: ignore[reportUnknownMemberType]
    for name, value in {
        "build_canonical_representations": build_canonical_representations,
        "build_stft_make_classmethod_case": build_stft_make_classmethod_case,
        "build_stft_times_and_frequencies_case": build_stft_times_and_frequencies_case,
        "build_fdata": build_fdata,
        "build_tsdata_case": build_tsdata_case,
        "build_xyz_tsdata_case": build_xyz_tsdata_case,
        "build_xyz_spectral_density": build_xyz_spectral_density,
        "build_xyz_evolutionary_spectral_density": (
            build_xyz_evolutionary_spectral_density
        ),
        "build_fd_template_band_case": build_fd_template_band_case,
        "build_fd_linspace_noise_case": build_fd_linspace_noise_case,
        "build_wdm": build_wdm,
        "build_fd_pair": build_fd_pair,
        "build_wdm_pair": build_wdm_pair,
        "build_fd_pair_batched": build_fd_pair_batched,
        "build_wdm_pair_batched": build_wdm_pair_batched,
        "diagonal_kernel_3ch": diagonal_kernel_3ch,
        "dense_kernel_3ch": dense_kernel_3ch,
        "dense_esdm_3ch": dense_esdm_3ch,
        "build_harmonic_waveform_frequency_series": (
            build_harmonic_waveform_frequency_series
        ),
        "build_harmonic_projected_phasor_waveform": (
            build_harmonic_projected_phasor_waveform
        ),
        "build_fake_harmonic_projected_waveform": (
            build_fake_harmonic_projected_waveform
        ),
        "build_test_phasor": build_test_phasor,
        "make_mock_phasor": build_test_phasor,
        "build_harmonic_projected_frequency_waveform": (
            build_harmonic_projected_frequency_waveform
        ),
        "build_harmonic_waveform_constructor": build_harmonic_waveform_constructor,
        "build_projected_waveform_constructor": build_projected_waveform_constructor,
        "build_harmonic_projected_waveform_constructor": (
            build_harmonic_projected_waveform_constructor
        ),
        "build_homogeneous_harmonic_projected_waveform_constructor": (
            build_homogeneous_harmonic_projected_waveform_constructor
        ),
        "build_harmonic_waveform_from_mapping": build_harmonic_waveform_from_mapping,
        "build_projected_waveform_from_mapping": build_projected_waveform_from_mapping,
        "build_harmonic_projected_waveform_from_mapping": (
            build_harmonic_projected_waveform_from_mapping
        ),
        "build_homogeneous_harmonic_projected_waveform_from_mapping": (
            build_homogeneous_harmonic_projected_waveform_from_mapping
        ),
    }.items():
        module_dict[name] = value
