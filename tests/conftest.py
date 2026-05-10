# pyright: reportPrivateUsage=false

from types import ModuleType
from typing import cast

import numpy as np
import pytest

from typed_lisa_toolkit import (
    axis,
    build_grid2d,
    cast_mode,
    frequency_phasor,
    frequency_series,
    fsdata,
    harmonic_projected_waveform,
    harmonic_waveform,
    homogeneous_harmonic_projected_waveform,
    linspace_from_array,
    make_sdm,
    projected_waveform,
    stft,
    stftdata,
    time_phasor,
    time_series,
    tsdata,
    wdm,
    wdmdata,
)
from typed_lisa_toolkit.types import (
    STFT,
    WDM,
    Array,
    Axis,
    FrequencySeries,
    FSData,
    Grid2DCartesian,
    Grid2DSparse,
    Harmonic,
    HarmonicProjectedWaveform,
    Linspace,
    Phasor,
    STFTData,
    TimeSeries,
    TSData,
    UniformFrequencySeries,
    UniformTimeSeries,
    WDMData,
)

SEED = 11324214
rng = np.random.default_rng(SEED)


@pytest.fixture(scope="session", name="xp", params=["numpy", "jax"])
def xp_fixture(request) -> ModuleType:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    """Fixture to parametrize tests over different array libraries."""
    xp_name = request.param  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if xp_name == "numpy":
        import numpy as np

        return np
    if xp_name == "jax":
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", val=True)  # pyright: ignore[reportUnknownMemberType]

        return jnp
    msg = f"Unsupported array library: {xp_name}"
    raise ValueError(msg)


### Axis fixtures


@pytest.fixture(scope="session", name="lin_time_axis")
def lin_time_axis_fixture(xp: ModuleType) -> Axis[Linspace]:
    return axis(
        linspace_from_array(
            xp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=xp.float64)
        )
    )


@pytest.fixture(scope="session", name="uni_ary_time_axis")
def uni_ary_time_axis_fixture(xp: ModuleType) -> Axis[Array]:
    return axis(xp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=xp.float64))


@pytest.fixture(scope="session", name="ary_time_axis")
def ary_time_axis_fixture(xp: ModuleType) -> Axis[Array]:
    return axis(xp.asarray([0.0, 1.0, 2.0, 4.0, 7.0, 11.0], dtype=xp.float64))


@pytest.fixture(scope="session", name="long_time_axis")
def long_time_axis_fixture(xp: ModuleType) -> Axis[Linspace]:
    return axis(
        linspace_from_array(
            xp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=xp.float64)
        )
    )


@pytest.fixture(scope="session", name="short_time_axis")
def short_time_axis_fixture(xp: ModuleType) -> Axis[Array]:
    return axis(xp.asarray([0.0, 1.0], dtype=xp.float64))


@pytest.fixture(scope="session", name="lin_freq_axis")
def lin_freq_axis_fixture(xp: ModuleType) -> Axis[Linspace]:
    return axis(linspace_from_array(xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)))


@pytest.fixture(scope="session", name="uni_ary_freq_axis")
def uni_ary_freq_axis_fixture(xp: ModuleType) -> Axis[Array]:
    return axis(xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64))


@pytest.fixture(scope="session", name="dense_ary_freq_axis")
def dense_ary_freq_axis_fixture(xp: ModuleType) -> Axis[Array]:
    return axis(xp.linspace(1.0, 3.0, num=10))


@pytest.fixture(scope="session", name="ary_freq_axis")
def ary_freq_axis_fixture(xp: ModuleType) -> Axis[Array]:
    return axis(xp.asarray([1.0, 2.0, 4.0], dtype=xp.float64))


@pytest.fixture(scope="session", name="long_freq_axis")
def long_freq_axis_fixture(xp: ModuleType) -> Axis[Array]:
    return axis(xp.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=xp.float64))


### Grid fixtures


@pytest.fixture(scope="session", name="lin_time_grid1d")
def lin_time_grid1d_fixture(lin_time_axis: Axis[Linspace]):
    return (lin_time_axis,)


@pytest.fixture(scope="session", name="uni_ary_time_grid1d")
def uni_ary_time_grid1d_fixture(uni_ary_time_axis: Axis[Array]):
    return (uni_ary_time_axis,)


@pytest.fixture(scope="session", name="ary_time_grid1d")
def ary_time_grid1d_fixture(ary_time_axis: Axis[Array]):
    return (ary_time_axis,)


@pytest.fixture(scope="session", name="long_time_grid1d")
def long_time_grid1d_fixture(long_time_axis: Axis[Array]):
    return (long_time_axis,)


@pytest.fixture(scope="session", name="lin_freq_grid1d")
def lin_freq_grid1d_fixture(lin_freq_axis: Axis[Linspace]):
    return (lin_freq_axis,)


@pytest.fixture(scope="session", name="long_freq_grid1d")
def long_freq_grid1d_fixture(long_freq_axis: Axis[Array]):
    return (long_freq_axis,)


@pytest.fixture(scope="session", name="uni_ary_freq_grid1d")
def uni_ary_freq_grid1d_fixture(uni_ary_freq_axis: Axis[Array]):
    return (uni_ary_freq_axis,)


@pytest.fixture(scope="session", name="ary_freq_grid1d")
def ary_freq_grid1d_fixture(ary_freq_axis: Axis[Array]):
    return (ary_freq_axis,)


@pytest.fixture(scope="session", name="lin_lin_cartesian")
def lin_lin_cartesian_fixture(
    lin_freq_axis: Axis[Linspace], lin_time_axis: Axis[Linspace]
):
    return build_grid2d(lin_freq_axis, lin_time_axis)


@pytest.fixture(scope="session", name="lin_uni_cartesian")
def lin_uni_cartesian_fixture(
    lin_freq_axis: Axis[Linspace], uni_ary_time_axis: Axis[Array]
):
    return build_grid2d(lin_freq_axis, uni_ary_time_axis)


@pytest.fixture(scope="session", name="lin_ary_cartesian")
def lin_ary_cartesian_fixture(
    lin_freq_axis: Axis[Linspace], ary_time_axis: Axis[Array]
):
    return build_grid2d(lin_freq_axis, ary_time_axis)


@pytest.fixture(scope="session", name="uni_lin_cartesian")
def uni_lin_cartesian_fixture(
    uni_ary_freq_axis: Axis[Array], lin_time_axis: Axis[Linspace]
):
    return build_grid2d(uni_ary_freq_axis, lin_time_axis)


@pytest.fixture(scope="session", name="uni_uni_cartesian")
def uni_uni_cartesian_fixture(
    uni_ary_freq_axis: Axis[Array], uni_ary_time_axis: Axis[Array]
):
    return build_grid2d(uni_ary_freq_axis, uni_ary_time_axis)


@pytest.fixture(scope="session", name="uni_ary_cartesian")
def uni_ary_cartesian_fixture(
    uni_ary_freq_axis: Axis[Array], ary_time_axis: Axis[Array]
):
    return build_grid2d(uni_ary_freq_axis, ary_time_axis)


@pytest.fixture(scope="session", name="ary_lin_cartesian")
def ary_lin_cartesian_fixture(
    ary_freq_axis: Axis[Array], lin_time_axis: Axis[Linspace]
):
    return build_grid2d(ary_freq_axis, lin_time_axis)


@pytest.fixture(scope="session", name="ary_uni_cartesian")
def ary_uni_cartesian_fixture(
    ary_freq_axis: Axis[Array], uni_ary_time_axis: Axis[Array]
):
    return build_grid2d(ary_freq_axis, uni_ary_time_axis)


@pytest.fixture(scope="session", name="ary_ary_cartesian")
def ary_ary_cartesian_fixture(ary_freq_axis: Axis[Array], ary_time_axis: Axis[Array]):
    return build_grid2d(ary_freq_axis, ary_time_axis)


@pytest.fixture(scope="session", name="lin_lin_sparse")
def lin_lin_sparse_fixture(
    xp: ModuleType, lin_freq_axis: Axis[Linspace], lin_time_axis: Axis[Linspace]
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(lin_freq_axis, lin_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="lin_uni_sparse")
def lin_uni_sparse_fixture(
    xp: ModuleType,
    lin_freq_axis: Axis[Linspace],
    uni_ary_time_axis: Axis[Array],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(lin_freq_axis, uni_ary_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="lin_ary_sparse")
def lin_ary_sparse_fixture(
    xp: ModuleType,
    lin_freq_axis: Axis[Linspace],
    ary_time_axis: Axis[Array],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(lin_freq_axis, ary_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="uni_lin_sparse")
def uni_lin_sparse_fixture(
    xp: ModuleType,
    uni_ary_freq_axis: Axis[Array],
    lin_time_axis: Axis[Linspace],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(uni_ary_freq_axis, lin_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="uni_uni_sparse")
def uni_uni_sparse_fixture(
    xp: ModuleType,
    uni_ary_freq_axis: Axis[Array],
    uni_ary_time_axis: Axis[Array],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(
        uni_ary_freq_axis, uni_ary_time_axis, sparse_indices=sparse_indices
    )


@pytest.fixture(scope="session", name="uni_ary_sparse")
def uni_ary_sparse_fixture(
    xp: ModuleType,
    uni_ary_freq_axis: Axis[Array],
    ary_time_axis: Axis[Array],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(uni_ary_freq_axis, ary_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="ary_lin_sparse")
def ary_lin_sparse_fixture(
    xp: ModuleType,
    ary_freq_axis: Axis[Array],
    lin_time_axis: Axis[Linspace],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(ary_freq_axis, lin_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="ary_uni_sparse")
def ary_uni_sparse_fixture(
    xp: ModuleType,
    ary_freq_axis: Axis[Array],
    uni_ary_time_axis: Axis[Array],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(ary_freq_axis, uni_ary_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="ary_ary_sparse")
def ary_ary_sparse_fixture(
    xp: ModuleType,
    ary_freq_axis: Axis[Array],
    ary_time_axis: Axis[Array],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(ary_freq_axis, ary_time_axis, sparse_indices=sparse_indices)


@pytest.fixture(scope="session", name="lin_lin_sparse_cartesian")
def lin_lin_sparse_cartesian_fixture(
    xp: ModuleType,
    lin_freq_axis: Axis[Linspace],
    lin_time_axis: Axis[Linspace],
):
    sparse_indices = cast("Array", xp.asarray([[0, 0], [1, 2], [2, 4]]))
    return build_grid2d(lin_freq_axis, lin_time_axis, sparse_indices=sparse_indices)


### Representation fixtures


@pytest.fixture(scope="session", name="lin_freq_series")
def lin_freq_series_fixture(
    xp: ModuleType, lin_freq_axis: Axis[Linspace]
) -> UniformFrequencySeries:
    entries = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    return frequency_series(lin_freq_axis, entries[None, None, None, None, :])


@pytest.fixture(scope="session", name="uni_ary_freq_series")
def uni_ary_freq_series_fixture(
    xp: ModuleType,
    uni_ary_freq_axis: Axis[Array],
) -> FrequencySeries[Axis[Array]]:
    entries = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    return frequency_series(uni_ary_freq_axis, entries[None, None, None, None, :])


@pytest.fixture(scope="session", name="ary_freq_series")
def ary_freq_series_fixture(
    xp: ModuleType, ary_freq_axis: Axis[Array]
) -> FrequencySeries[Axis[Array]]:
    entries = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    return frequency_series(ary_freq_axis, entries[None, None, None, None, :])


@pytest.fixture(scope="session", name="lin_time_series")
def lin_time_series_fixture(
    xp: ModuleType, lin_time_axis: Axis[Linspace]
) -> UniformTimeSeries:
    entries = xp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25], dtype=xp.float64)
    return time_series(lin_time_axis, entries[None, None, None, None, :])


@pytest.fixture(scope="session", name="long_time_series")
def long_time_series_fixture(
    xp: ModuleType, long_time_axis: Axis[Linspace]
) -> UniformTimeSeries:
    entries = xp.asarray(
        [0.0, 1.0, 0.5, -0.5, -1.0, -0.25, 0.75, -0.75], dtype=xp.float64
    )
    return time_series(long_time_axis, entries[None, None, None, None, :])


@pytest.fixture(scope="session", name="uni_ary_time_series")
def uni_ary_time_series_fixture(
    xp: ModuleType,
    uni_ary_time_axis: Axis[Array],
) -> TimeSeries[Axis[Array]]:
    entries = xp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25], dtype=xp.float64)
    return time_series(uni_ary_time_axis, entries[None, None, None, None, :])


@pytest.fixture(scope="session", name="ary_time_series")
def ary_time_series_fixture(
    xp: ModuleType, ary_time_axis: Axis[Array]
) -> TimeSeries[Axis[Array]]:
    entries = xp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25], dtype=xp.float64)
    return time_series(ary_time_axis, entries[None, None, None, None, :])


@pytest.fixture(scope="session", name="lin_freq_phasor")
def lin_freq_phasor_fixture(xp: ModuleType, lin_freq_axis: Axis[Linspace]):
    amplitudes = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64) * (1 + 1j)
    phases = xp.asarray([0.0, 0.5, 1.0], dtype=xp.float64) * np.pi
    return frequency_phasor(
        frequencies=lin_freq_axis,
        amplitudes=amplitudes[None, None, None, None, :],
        phases=phases[None, None, None, None, :],
    )


@pytest.fixture(scope="session", name="lin_time_phasor")
def lin_time_phasor_fixture(xp: ModuleType, lin_time_axis: Axis[Linspace]):
    amplitudes = xp.asarray([1.0, 0.5, 0.25, -0.25, -0.5, -0.75], dtype=xp.float64) * (
        1 + 1j
    )
    phases = xp.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=xp.float64) * np.pi
    return time_phasor(
        times=lin_time_axis,
        amplitudes=amplitudes[None, None, None, None, :],
        phases=phases[None, None, None, None, :],
    )


@pytest.fixture(scope="session", name="uni_ary_phasor")
def uni_ary_phasor_fixture(xp: ModuleType, uni_ary_freq_axis: Axis[Array]):
    amplitudes = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64) * (1 + 1j)
    phases = xp.asarray([0.0, 0.5, 1.0], dtype=xp.float64) * np.pi
    return frequency_phasor(
        frequencies=uni_ary_freq_axis,
        amplitudes=amplitudes[None, None, None, None, :],
        phases=phases[None, None, None, None, :],
    )


@pytest.fixture(scope="session", name="ary_phasor")
def ary_phasor_fixture(xp: ModuleType, ary_freq_axis: Axis[Array]):
    amplitudes = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64) * (1 + 1j)
    phases = xp.asarray([0.0, 0.5, 1.0], dtype=xp.float64) * np.pi
    return frequency_phasor(
        frequencies=ary_freq_axis,
        amplitudes=amplitudes[None, None, None, None, :],
        phases=phases[None, None, None, None, :],
    )


@pytest.fixture(scope="session", name="lin_lin_cartesian_stft")
def lin_lin_cartesian_stft_fixture(
    xp: ModuleType,
    lin_lin_cartesian: Grid2DCartesian[Axis[Linspace], Axis[Linspace]],
) -> STFT[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(lin_lin_cartesian[0]), len(lin_lin_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(lin_lin_cartesian[0], lin_lin_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="lin_uni_cartesian_stft")
def lin_uni_cartesian_stft_fixture(
    xp: ModuleType,
    lin_uni_cartesian: Grid2DCartesian[Axis[Linspace], Axis[Array]],
) -> STFT[Grid2DCartesian[Axis[Linspace], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(lin_uni_cartesian[0]), len(lin_uni_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(lin_uni_cartesian[0], lin_uni_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="lin_ary_cartesian_stft")
def lin_ary_cartesian_stft_fixture(
    xp: ModuleType,
    lin_ary_cartesian: Grid2DCartesian[Axis[Linspace], Axis[Array]],
) -> STFT[Grid2DCartesian[Axis[Linspace], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(lin_ary_cartesian[0]), len(lin_ary_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(lin_ary_cartesian[0], lin_ary_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="uni_lin_cartesian_stft")
def uni_lin_cartesian_stft_fixture(
    xp: ModuleType,
    uni_lin_cartesian: Grid2DCartesian[Axis[Array], Axis[Linspace]],
) -> STFT[Grid2DCartesian[Axis[Array], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(uni_lin_cartesian[0]), len(uni_lin_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(uni_lin_cartesian[0], uni_lin_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="uni_uni_cartesian_stft")
def uni_uni_cartesian_stft_fixture(
    xp: ModuleType,
    uni_uni_cartesian: Grid2DCartesian[Axis[Array], Axis[Array]],
) -> STFT[Grid2DCartesian[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(uni_uni_cartesian[0]), len(uni_uni_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(uni_uni_cartesian[0], uni_uni_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="uni_ary_cartesian_stft")
def uni_ary_cartesian_stft_fixture(
    xp: ModuleType,
    uni_ary_cartesian: Grid2DCartesian[Axis[Array], Axis[Array]],
) -> STFT[Grid2DCartesian[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(uni_ary_cartesian[0]), len(uni_ary_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(uni_ary_cartesian[0], uni_ary_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="ary_lin_cartesian_stft")
def ary_lin_cartesian_stft_fixture(
    xp: ModuleType,
    ary_lin_cartesian: Grid2DCartesian[Axis[Array], Axis[Linspace]],
) -> STFT[Grid2DCartesian[Axis[Array], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(ary_lin_cartesian[0]), len(ary_lin_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(ary_lin_cartesian[0], ary_lin_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="ary_uni_cartesian_stft")
def ary_uni_cartesian_stft_fixture(
    xp: ModuleType,
    ary_uni_cartesian: Grid2DCartesian[Axis[Array], Axis[Array]],
) -> STFT[Grid2DCartesian[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(ary_uni_cartesian[0]), len(ary_uni_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(ary_uni_cartesian[0], ary_uni_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="ary_ary_cartesian_stft")
def ary_ary_cartesian_stft_fixture(
    xp: ModuleType,
    ary_ary_cartesian: Grid2DCartesian[Axis[Array], Axis[Array]],
) -> STFT[Grid2DCartesian[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(ary_ary_cartesian[0]), len(ary_ary_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return stft(ary_ary_cartesian[0], ary_ary_cartesian[1], entries=entries)


@pytest.fixture(scope="session", name="lin_lin_sparse_stft")
def lin_lin_sparse_stft_fixture(
    xp: ModuleType,
    lin_lin_sparse: Grid2DSparse[Axis[Linspace], Axis[Linspace]],
) -> STFT[Grid2DSparse[Axis[Linspace], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        lin_lin_sparse[0],
        lin_lin_sparse[1],
        entries=entries,
        sparse_indices=lin_lin_sparse.indices,
    )


@pytest.fixture(scope="session", name="lin_uni_sparse_stft")
def lin_uni_sparse_stft_fixture(
    xp: ModuleType,
    lin_uni_sparse: Grid2DSparse[Axis[Linspace], Axis[Array]],
) -> STFT[Grid2DSparse[Axis[Linspace], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        lin_uni_sparse[0],
        lin_uni_sparse[1],
        entries=entries,
        sparse_indices=lin_uni_sparse.indices,
    )


@pytest.fixture(scope="session", name="lin_ary_sparse_stft")
def lin_ary_sparse_stft_fixture(
    xp: ModuleType,
    lin_ary_sparse: Grid2DSparse[Axis[Linspace], Axis[Array]],
) -> STFT[Grid2DSparse[Axis[Linspace], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        lin_ary_sparse[0],
        lin_ary_sparse[1],
        entries=entries,
        sparse_indices=lin_ary_sparse.indices,
    )


@pytest.fixture(scope="session", name="uni_lin_sparse_stft")
def uni_lin_sparse_stft_fixture(
    xp: ModuleType,
    uni_lin_sparse: Grid2DSparse[Axis[Array], Axis[Linspace]],
) -> STFT[Grid2DSparse[Axis[Array], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        uni_lin_sparse[0],
        uni_lin_sparse[1],
        entries=entries,
        sparse_indices=uni_lin_sparse.indices,
    )


@pytest.fixture(scope="session", name="uni_uni_sparse_stft")
def uni_uni_sparse_stft_fixture(
    xp: ModuleType,
    uni_uni_sparse: Grid2DSparse[Axis[Array], Axis[Array]],
) -> STFT[Grid2DSparse[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        uni_uni_sparse[0],
        uni_uni_sparse[1],
        entries=entries,
        sparse_indices=uni_uni_sparse.indices,
    )


@pytest.fixture(scope="session", name="uni_ary_sparse_stft")
def uni_ary_sparse_stft_fixture(
    xp: ModuleType,
    uni_ary_sparse: Grid2DSparse[Axis[Array], Axis[Array]],
) -> STFT[Grid2DSparse[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        uni_ary_sparse[0],
        uni_ary_sparse[1],
        entries=entries,
        sparse_indices=uni_ary_sparse.indices,
    )


@pytest.fixture(scope="session", name="ary_lin_sparse_stft")
def ary_lin_sparse_stft_fixture(
    xp: ModuleType,
    ary_lin_sparse: Grid2DSparse[Axis[Array], Axis[Linspace]],
) -> STFT[Grid2DSparse[Axis[Array], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        ary_lin_sparse[0],
        ary_lin_sparse[1],
        entries=entries,
        sparse_indices=ary_lin_sparse.indices,
    )


@pytest.fixture(scope="session", name="ary_uni_sparse_stft")
def ary_uni_sparse_stft_fixture(
    xp: ModuleType,
    ary_uni_sparse: Grid2DSparse[Axis[Array], Axis[Array]],
) -> STFT[Grid2DSparse[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        ary_uni_sparse[0],
        ary_uni_sparse[1],
        entries=entries,
        sparse_indices=ary_uni_sparse.indices,
    )


@pytest.fixture(scope="session", name="ary_ary_sparse_stft")
def ary_ary_sparse_stft_fixture(
    xp: ModuleType,
    ary_ary_sparse: Grid2DSparse[Axis[Array], Axis[Array]],
) -> STFT[Grid2DSparse[Axis[Array], Axis[Array]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return stft(
        ary_ary_sparse[0],
        ary_ary_sparse[1],
        entries=entries,
        sparse_indices=ary_ary_sparse.indices,
    )


@pytest.fixture(scope="session", name="lin_lin_cartesian_wdm")
def lin_lin_cartesian_wdm_fixture(
    xp: ModuleType,
    lin_lin_cartesian: Grid2DCartesian[Axis[Linspace], Axis[Linspace]],
) -> WDM[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal(
            (1, 1, 1, 1, len(lin_lin_cartesian[0]), len(lin_lin_cartesian[1]))
        ),
        dtype=xp.float64,
    )
    return wdm(
        frequencies=lin_lin_cartesian[0],
        times=lin_lin_cartesian[1],
        entries=entries,
    )


@pytest.fixture(scope="session", name="lin_lin_sparse_wdm")
def lin_lin_sparse_wdm_fixture(
    xp: ModuleType,
    lin_lin_sparse: Grid2DSparse[Axis[Linspace], Axis[Linspace]],
) -> WDM[Grid2DSparse[Axis[Linspace], Axis[Linspace]]]:
    entries = xp.asarray(
        np.random.default_rng(SEED).standard_normal((1, 1, 1, 1, 3)),
        dtype=xp.float64,
    )
    return wdm(
        frequencies=lin_lin_sparse[0],
        times=lin_lin_sparse[1],
        entries=entries,
        sparse_indices=lin_lin_sparse.indices,
    )


## Data fixtures


@pytest.fixture(scope="session", name="tsdata")
def tsdata_fixture(xp: ModuleType, lin_time_axis: Axis[Linspace]) -> TSData:
    x = xp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25], dtype=xp.float64)
    y = xp.asarray([1.0, 0.0, -0.5, 0.25, 0.5, -0.75], dtype=xp.float64)
    z = xp.asarray([0.5, -0.25, 0.25, -0.75, 0.0, 0.5], dtype=xp.float64)
    return tsdata(
        {
            "X": time_series(lin_time_axis, x[None, None, None, None, :]),
            "Y": time_series(lin_time_axis, y[None, None, None, None, :]),
            "Z": time_series(lin_time_axis, z[None, None, None, None, :]),
        },
    )


@pytest.fixture(scope="session", name="long_tsdata")
def long_tsdata_fixture(xp: ModuleType, long_time_axis: Axis[Linspace]) -> TSData:
    x = xp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25, 0.25, 0.75], dtype=xp.float64)
    y = xp.asarray([1.0, 0.0, -0.5, 0.25, 0.5, -0.75, 0.0, 0.5], dtype=xp.float64)
    z = xp.asarray([0.5, -0.25, 0.25, -0.75, 0.0, 0.5, 0.0, 0.25], dtype=xp.float64)
    return tsdata(
        {
            "X": time_series(long_time_axis, x[None, None, None, None, :]),
            "Y": time_series(long_time_axis, y[None, None, None, None, :]),
            "Z": time_series(long_time_axis, z[None, None, None, None, :]),
        },
    )


@pytest.fixture(scope="session", name="fsdata")
def fsdata_fixture(xp: ModuleType, lin_freq_axis: Axis[Linspace]) -> FSData:
    x = xp.asarray([1.0 + 0.5j, -1.0j, 2.0 + 0.0j], dtype=xp.complex128)
    y = xp.asarray([0.5 - 0.25j, -1.0 + 0.25j, 2.0 + 0.5j], dtype=xp.complex128)
    z = xp.asarray([0.2 + 0.75j, -0.5 - 0.25j, 1.25 + 0.1j], dtype=xp.complex128)
    return fsdata(
        {
            "X": frequency_series(lin_freq_axis, x[None, None, None, None, :]),
            "Y": frequency_series(lin_freq_axis, y[None, None, None, None, :]),
            "Z": frequency_series(lin_freq_axis, z[None, None, None, None, :]),
        },
    )


@pytest.fixture(scope="session", name="stftdata_cartesian")
def stftdata_cartesian_fixture(
    xp: ModuleType,
    lin_lin_cartesian: Grid2DCartesian[Axis[Linspace], Axis[Linspace]],
) -> STFTData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]:
    n_freq, n_time = len(lin_lin_cartesian[0]), len(lin_lin_cartesian[1])
    rng = np.random.default_rng(SEED)
    x = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_freq, n_time)),
        dtype=xp.float64,
    )
    y = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_freq, n_time)),
        dtype=xp.float64,
    )
    z = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_freq, n_time)),
        dtype=xp.float64,
    )
    return stftdata(
        {
            "X": stft(lin_lin_cartesian[0], lin_lin_cartesian[1], entries=x),
            "Y": stft(lin_lin_cartesian[0], lin_lin_cartesian[1], entries=y),
            "Z": stft(lin_lin_cartesian[0], lin_lin_cartesian[1], entries=z),
        },
    )


@pytest.fixture(scope="session", name="stftdata_sparse")
def stftdata_sparse_fixture(
    xp: ModuleType,
    lin_lin_sparse: Grid2DSparse[Axis[Linspace], Axis[Linspace]],
) -> STFTData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]]:
    n_entries = len(lin_lin_sparse.indices)
    rng = np.random.default_rng(SEED)
    x = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_entries)),
        dtype=xp.float64,
    )
    y = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_entries)),
        dtype=xp.float64,
    )
    z = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_entries)),
        dtype=xp.float64,
    )
    return stftdata(
        {
            "X": stft(
                lin_lin_sparse[0],
                lin_lin_sparse[1],
                entries=x,
                sparse_indices=lin_lin_sparse.indices,
            ),
            "Y": stft(
                lin_lin_sparse[0],
                lin_lin_sparse[1],
                entries=y,
                sparse_indices=lin_lin_sparse.indices,
            ),
            "Z": stft(
                lin_lin_sparse[0],
                lin_lin_sparse[1],
                entries=z,
                sparse_indices=lin_lin_sparse.indices,
            ),
        },
    )


@pytest.fixture(scope="session", name="wdmdata_cartesian")
def wdmdata_cartesian_fixture(
    xp: ModuleType,
    lin_lin_cartesian: Grid2DCartesian[Axis[Linspace], Axis[Linspace]],
) -> WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]]:
    n_freq, n_time = len(lin_lin_cartesian[0]), len(lin_lin_cartesian[1])
    rng = np.random.default_rng(SEED)
    x = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_freq, n_time)),
        dtype=xp.float64,
    )
    y = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_freq, n_time)),
        dtype=xp.float64,
    )
    z = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_freq, n_time)),
        dtype=xp.float64,
    )
    return wdmdata(
        {
            "X": wdm(lin_lin_cartesian[0], lin_lin_cartesian[1], entries=x),
            "Y": wdm(lin_lin_cartesian[0], lin_lin_cartesian[1], entries=y),
            "Z": wdm(lin_lin_cartesian[0], lin_lin_cartesian[1], entries=z),
        },
    )


@pytest.fixture(scope="session", name="wdmdata_sparse")
def wdmdata_sparse_fixture(
    xp: ModuleType,
    lin_lin_sparse: Grid2DSparse[Axis[Linspace], Axis[Linspace]],
) -> WDMData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]]:
    n_entries = len(lin_lin_sparse.indices)
    rng = np.random.default_rng(SEED)
    x = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_entries)),
        dtype=xp.float64,
    )
    y = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_entries)),
        dtype=xp.float64,
    )
    z = xp.asarray(
        rng.standard_normal((1, 1, 1, 1, n_entries)),
        dtype=xp.float64,
    )
    print(
        wdm(
            lin_lin_sparse[0],
            lin_lin_sparse[1],
            entries=x,
            sparse_indices=lin_lin_sparse.indices,
        )
    )
    return wdmdata(
        {
            "X": wdm(
                lin_lin_sparse[0],
                lin_lin_sparse[1],
                entries=x,
                sparse_indices=lin_lin_sparse.indices,
            ),
            "Y": wdm(
                lin_lin_sparse[0],
                lin_lin_sparse[1],
                entries=y,
                sparse_indices=lin_lin_sparse.indices,
            ),
            "Z": wdm(
                lin_lin_sparse[0],
                lin_lin_sparse[1],
                entries=z,
                sparse_indices=lin_lin_sparse.indices,
            ),
        },
    )


## Waveform fixtures


@pytest.fixture(scope="session", name="harmonic_waveform")
def harmonic_waveform_fixture(xp: ModuleType, lin_freq_axis: Axis[Linspace]):
    mode_22 = cast_mode((2, 2))
    mode_33 = cast_mode((3, 3))

    wf_22 = frequency_series(
        lin_freq_axis,
        xp.asarray([1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j], dtype=xp.complex128)[
            None, None, None, None, :
        ],
    )
    wf_33 = frequency_series(
        lin_freq_axis,
        xp.asarray([-0.5 + 1.0j, 0.25 + 0.0j, 1.5 - 0.25j], dtype=xp.complex128)[
            None, None, None, None, :
        ],
    )

    return harmonic_waveform({mode_22: wf_22, mode_33: wf_33})


@pytest.fixture(scope="session", name="hw_phasor")
def hw_phasor_fixture(xp: ModuleType, ary_freq_axis: Axis[Array]):
    mode_22 = cast_mode((2, 2))
    mode_33 = cast_mode((3, 3))

    wf_22 = frequency_phasor(
        frequencies=ary_freq_axis,
        amplitudes=cast(
            "Array",
            (xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64) * (1 + 1j))[
                None, None, None, None, :
            ],
        ),
        phases=cast(
            "Array",
            (xp.asarray([0.0, 0.5, 1.0], dtype=xp.float64) * xp.pi)[
                None, None, None, None, :
            ],
        ),
    )
    wf_33 = frequency_phasor(
        frequencies=ary_freq_axis,
        amplitudes=cast(
            "Array",
            (xp.asarray([0.5, 0.25, 0.75], dtype=xp.float64) * (1 + 1j))[
                None, None, None, None, :
            ],
        ),
        phases=cast(
            "Array",
            (xp.asarray([0.25, 0.75, 0.5], dtype=xp.float64) * xp.pi)[
                None, None, None, None, :
            ],
        ),
    )

    return harmonic_waveform({mode_22: wf_22, mode_33: wf_33})


@pytest.fixture(scope="session", name="projected_waveform")
def projected_waveform_fixture(xp: ModuleType, lin_freq_axis: Axis[Linspace]):
    x = frequency_series(
        lin_freq_axis,
        xp.asarray([1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j], dtype=xp.complex128)[
            None, None, None, None, :
        ],
    )
    y = frequency_series(
        lin_freq_axis,
        xp.asarray([0.5 + 0.25j, -1.0 + 0.0j, 0.25 - 0.25j], dtype=xp.complex128)[
            None, None, None, None, :
        ],
    )
    z = frequency_series(
        lin_freq_axis,
        xp.asarray([0.3 - 0.1j, 0.8 + 0.4j, -0.2 + 0.9j], dtype=xp.complex128)[
            None, None, None, None, :
        ],
    )
    return projected_waveform({"X": x, "Y": y, "Z": z})


@pytest.fixture(scope="session", name="harmonic_projected_waveform")
def harmonic_projected_waveform_fixture(xp: ModuleType, lin_freq_axis: Axis[Linspace]):
    mode_22 = cast_mode((2, 2))
    mode_33 = cast_mode((3, 3))

    resp_22 = projected_waveform(
        {
            "X": frequency_series(
                lin_freq_axis,
                xp.asarray([1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j], dtype=xp.complex128)[
                    None, None, None, None, :
                ],
            ),
            "Y": frequency_series(
                lin_freq_axis,
                xp.asarray(
                    [0.5 + 0.25j, -1.0 + 0.0j, 0.25 - 0.25j],
                    dtype=xp.complex128,
                )[None, None, None, None, :],
            ),
            "Z": frequency_series(
                lin_freq_axis,
                xp.asarray([0.3 - 0.1j, 0.8 + 0.4j, -0.2 + 0.9j], dtype=xp.complex128)[
                    None, None, None, None, :
                ],
            ),
        },
    )
    resp_33 = projected_waveform(
        {
            "X": frequency_series(
                lin_freq_axis,
                xp.asarray([0.2 + 0.0j, -0.5 + 1.0j, 0.1 - 0.2j], dtype=xp.complex128)[
                    None, None, None, None, :
                ],
            ),
            "Y": frequency_series(
                lin_freq_axis,
                xp.asarray([1.0 + 0.0j, 1.5 + 0.0j, 2.0 + 0.0j], dtype=xp.complex128)[
                    None, None, None, None, :
                ],
            ),
            "Z": frequency_series(
                lin_freq_axis,
                xp.asarray([0.4 + 0.2j, -0.7 + 0.1j, 1.1 - 0.3j], dtype=xp.complex128)[
                    None, None, None, None, :
                ],
            ),
        },
    )

    return harmonic_projected_waveform({mode_22: resp_22, mode_33: resp_33})


@pytest.fixture(scope="session", name="hpw_freq_phasor")
def hpw_freq_phasor_fixture(xp: ModuleType, ary_freq_axis: Axis[Array]):
    mode_22 = cast_mode((2, 2))
    mode_33 = cast_mode((3, 3))

    resp_22 = projected_waveform(
        {
            "X": frequency_phasor(
                frequencies=ary_freq_axis,
                amplitudes=cast(
                    "Array",
                    (xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                phases=cast(
                    "Array",
                    (xp.asarray([0.0, 0.5, 1.0], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
            "Y": frequency_phasor(
                frequencies=ary_freq_axis,
                amplitudes=cast(
                    "Array",
                    (xp.asarray([0.5, 0.25, 0.75], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                phases=cast(
                    "Array",
                    (xp.asarray([0.25, 0.75, 0.5], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
            "Z": frequency_phasor(
                frequencies=ary_freq_axis,
                amplitudes=cast(
                    "Array",
                    (xp.asarray([0.3, 0.8, -0.2], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                phases=cast(
                    "Array",
                    (xp.asarray([-0.1, 0.4, -0.9], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
        },
    )
    resp_33 = projected_waveform(
        {
            "X": frequency_phasor(
                frequencies=ary_freq_axis,
                amplitudes=cast(
                    "Array",
                    (xp.asarray([0.2, -0.5, 0.1], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                phases=cast(
                    "Array",
                    (xp.asarray([0.0, 1.0, -0.2], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
            "Y": frequency_phasor(
                frequencies=ary_freq_axis,
                amplitudes=cast(
                    "Array",
                    (xp.asarray([1.0, 1.5, 2.0], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                phases=cast(
                    "Array",
                    (xp.asarray([0.0, 0.0, 0.0], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
            "Z": frequency_phasor(
                frequencies=ary_freq_axis,
                amplitudes=cast(
                    "Array",
                    (xp.asarray([0.4, -0.7, 1.1], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                phases=cast(
                    "Array",
                    (xp.asarray([0.2, -0.1, -0.3], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
        },
    )
    return harmonic_projected_waveform({mode_22: resp_22, mode_33: resp_33})


@pytest.fixture(scope="session", name="homogeneous_harmonic_projected_waveform")
def homogeneous_harmonic_projected_waveform_fixture(
    harmonic_projected_waveform: HarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
):
    return homogeneous_harmonic_projected_waveform(
        {
            mode: projected_waveform_obj
            for mode, projected_waveform_obj in harmonic_projected_waveform.items()
        },
    )


@pytest.fixture(scope="session", name="pw_freq_phasor")
def pw_freq_phasor_fixture(xp: ModuleType, ary_freq_axis: Axis[Array]):
    resp = projected_waveform(
        {
            "X": frequency_phasor(
                ary_freq_axis,
                cast(
                    "Array",
                    (xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                cast(
                    "Array",
                    (xp.asarray([0.0, 0.5, 1.0], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
            "Y": frequency_phasor(
                ary_freq_axis,
                cast(
                    "Array",
                    (xp.asarray([0.5, 0.25, 0.75], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                cast(
                    "Array",
                    (xp.asarray([0.25, 0.75, 0.5], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
            "Z": frequency_phasor(
                ary_freq_axis,
                cast(
                    "Array",
                    (xp.asarray([0.3, 0.8, -0.2], dtype=xp.float64) * (1 + 1j))[
                        None, None, None, None, :
                    ],
                ),
                cast(
                    "Array",
                    (xp.asarray([-0.1, 0.4, -0.9], dtype=xp.float64) * xp.pi)[
                        None, None, None, None, :
                    ],
                ),
            ),
        },
    )
    return projected_waveform({"X": resp["X"], "Y": resp["Y"], "Z": resp["Z"]})


@pytest.fixture(scope="session", name="hhpw_freq_phasor")
def hhpw_freq_phasor_fixture(
    hpw_freq_phasor: HarmonicProjectedWaveform[Harmonic, Phasor[Axis[Array]]],
):
    return homogeneous_harmonic_projected_waveform(
        {
            mode: projected_waveform_obj
            for mode, projected_waveform_obj in hpw_freq_phasor.items()
        },
    )


### Helper fixtures


@pytest.fixture(scope="session", name="linear_interpolator")
def linear_interpolator_fixture(xp: ModuleType):  # pyright: ignore[reportUnknownParameterType]

    def _linear_interpolator(x, y):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        x_arr = xp.asarray(x, dtype=xp.float64)
        y_arr = xp.asarray(y)

        def _interp(x_new):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
            return xp.interp(xp.asarray(x_new, dtype=xp.float64), x_arr, y_arr)

        return _interp  # pyright: ignore[reportUnknownVariableType]

    return _linear_interpolator  # pyright: ignore[reportUnknownVariableType]


### Noise model fixtures


@pytest.fixture(scope="session", name="sdm")
def sdm_fixture(xp: ModuleType, lin_freq_axis: Axis[Linspace]):
    # rng = np.random.default_rng(SEED)
    values_x = xp.asarray([2.0, 4.0, 8.0], dtype=xp.float64)
    values_y = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    values_z = xp.asarray([1.1, 0.9, 1.4], dtype=xp.float64)
    offdiag_xy = xp.asarray([0.1, 0.2, -0.3], dtype=xp.float64)
    offdiag_xz = xp.asarray([-0.05, 0.15, 0.2], dtype=xp.float64)
    offdiag_yz = xp.asarray([0.08, -0.04, 0.12], dtype=xp.float64)
    row0 = xp.stack([values_x, offdiag_xy, offdiag_xz], axis=-1)
    row1 = xp.stack([offdiag_xy, values_y, offdiag_yz], axis=-1)
    row2 = xp.stack([offdiag_xz, offdiag_yz, values_z], axis=-1)
    inverse_sdm = xp.stack([row0, row1, row2], axis=-2)
    return make_sdm(
        inverse_sdm, frequencies=lin_freq_axis, channel_names=("X", "Y", "Z")
    )


@pytest.fixture(scope="session", name="esdm")
def esdm_fixture(
    xp: ModuleType, lin_freq_axis: Axis[Linspace], short_time_axis: Axis[Linspace]
):
    invevsdm = xp.asarray(
        [
            [
                [[2.0, 0.3, 0.1], [0.3, 1.2, -0.05], [0.1, -0.05, 1.4]],
                [[1.5, 0.2, -0.1], [0.2, 0.9, 0.04], [-0.1, 0.04, 1.1]],
            ],
            [
                [[2.5, 0.1, 0.08], [0.1, 1.3, 0.02], [0.08, 0.02, 1.6]],
                [[1.8, -0.2, 0.03], [-0.2, 1.1, -0.04], [0.03, -0.04, 1.2]],
            ],
            [
                [[1.9, 0.05, -0.02], [0.05, 1.0, 0.01], [-0.02, 0.01, 1.3]],
                [[2.2, 0.12, 0.06], [0.12, 1.4, -0.03], [0.06, -0.03, 1.7]],
            ],
        ],
    )
    return make_sdm(
        invevsdm,
        frequencies=lin_freq_axis,
        times=short_time_axis,
        channel_names=("X", "Y", "Z"),
    )
