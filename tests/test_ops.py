from types import ModuleType

import pytest

from typed_lisa_toolkit.types import (
    STFT,
    WDM,
    AnyArray,
    Axis,
    FrequencySeries,
    FSData,
    Grid2DCartesian,
    Grid2DSparse,
    Harmonic,
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    HomogeneousHarmonicProjectedWaveform,
    Linspace,
    Phasor,
    ProjectedWaveform,
    STFTData,
    TimeSeries,
    TSData,
    UniformFrequencySeries,
    UniformTimeSeries,
    WDMData,
)

## Representations


def test_lin_freq_series(xp: ModuleType, lin_freq_series: UniformFrequencySeries):
    # Scalar
    assert xp.all((lin_freq_series + 1).entries == lin_freq_series.entries + 1)
    assert xp.all((lin_freq_series * 2).entries == lin_freq_series.entries * 2)
    # Non-scalar
    ary = lin_freq_series.entries * 0.1
    assert xp.all(
        (lin_freq_series + ary).entries
        == xp.asarray(lin_freq_series.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_freq_series * ary).entries
        == xp.asarray(lin_freq_series.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_freq_series.entries) + xp.asarray(ary)
    lin_freq_series += ary
    assert xp.all(lin_freq_series.entries == expected)


def test_uni_ary_freq_series(
    xp: ModuleType,
    uni_ary_freq_series: FrequencySeries[Axis[AnyArray]],
):
    # Scalar
    assert xp.all((uni_ary_freq_series + 1).entries == uni_ary_freq_series.entries + 1)
    assert xp.all((uni_ary_freq_series * 2).entries == uni_ary_freq_series.entries * 2)
    # Non-scalar
    ary = uni_ary_freq_series.entries * 0.1
    assert xp.all(
        (uni_ary_freq_series + ary).entries
        == xp.asarray(uni_ary_freq_series.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_ary_freq_series * ary).entries
        == xp.asarray(uni_ary_freq_series.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_ary_freq_series.entries) + xp.asarray(ary)
    uni_ary_freq_series += ary
    assert xp.all(uni_ary_freq_series.entries == expected)


def test_ary_freq_series(
    xp: ModuleType,
    ary_freq_series: FrequencySeries[Axis[AnyArray]],
):
    # Scalar
    assert xp.all((ary_freq_series + 1).entries == ary_freq_series.entries + 1)
    assert xp.all((ary_freq_series * 2).entries == ary_freq_series.entries * 2)
    # Non-scalar
    ary = ary_freq_series.entries * 0.1
    assert xp.all(
        (ary_freq_series + ary).entries
        == xp.asarray(ary_freq_series.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_freq_series * ary).entries
        == xp.asarray(ary_freq_series.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_freq_series.entries) + xp.asarray(ary)
    ary_freq_series += ary
    assert xp.all(ary_freq_series.entries == expected)


def test_lin_time_series(xp: ModuleType, lin_time_series: UniformTimeSeries):
    # Scalar
    assert xp.all((lin_time_series + 1).entries == lin_time_series.entries + 1)
    assert xp.all((lin_time_series * 2).entries == lin_time_series.entries * 2)
    # Non-scalar
    ary = lin_time_series.entries * 0.1
    assert xp.all(
        (lin_time_series + ary).entries
        == xp.asarray(lin_time_series.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_time_series * ary).entries
        == xp.asarray(lin_time_series.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_time_series.entries) + xp.asarray(ary)
    lin_time_series += ary
    assert xp.all(lin_time_series.entries == expected)


def test_uni_ary_time_series(
    xp: ModuleType,
    uni_ary_time_series: TimeSeries[Axis[AnyArray]],
):
    # Scalar
    assert xp.all((uni_ary_time_series + 1).entries == uni_ary_time_series.entries + 1)
    assert xp.all((uni_ary_time_series * 2).entries == uni_ary_time_series.entries * 2)
    # Non-scalar
    ary = uni_ary_time_series.entries * 0.1
    assert xp.all(
        (uni_ary_time_series + ary).entries
        == xp.asarray(uni_ary_time_series.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_ary_time_series * ary).entries
        == xp.asarray(uni_ary_time_series.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_ary_time_series.entries) + xp.asarray(ary)
    uni_ary_time_series += ary
    assert xp.all(uni_ary_time_series.entries == expected)


def test_ary_time_series(
    xp: ModuleType,
    ary_time_series: TimeSeries[Axis[AnyArray]],
):
    # Scalar
    assert xp.all((ary_time_series + 1).entries == ary_time_series.entries + 1)
    assert xp.all((ary_time_series * 2).entries == ary_time_series.entries * 2)
    # Non-scalar
    ary = ary_time_series.entries * 0.1
    assert xp.all(
        (ary_time_series + ary).entries
        == xp.asarray(ary_time_series.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_time_series * ary).entries
        == xp.asarray(ary_time_series.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_time_series.entries) + xp.asarray(ary)
    ary_time_series += ary
    assert xp.all(ary_time_series.entries == expected)


def test_lin_phasor(lin_freq_phasor: Phasor[Axis[Linspace]]):
    # Catch error
    with pytest.raises(TypeError):
        _ = lin_freq_phasor + 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_uni_ary_phasor(uni_ary_phasor: Phasor[Axis[AnyArray]]):
    # Catch error
    with pytest.raises(TypeError):
        _ = uni_ary_phasor + 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_ary_phasor(ary_phasor: Phasor[Axis[AnyArray]]):
    # Catch error
    with pytest.raises(TypeError):
        _ = ary_phasor + 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_lin_lin_cartesian_stft(
    xp: ModuleType,
    lin_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all(
        (lin_lin_cartesian_stft + 1).entries == lin_lin_cartesian_stft.entries + 1
    )
    assert xp.all(
        (lin_lin_cartesian_stft * 2).entries == lin_lin_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = lin_lin_cartesian_stft.entries * 0.1
    assert xp.all(
        (lin_lin_cartesian_stft + ary).entries
        == xp.asarray(lin_lin_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_lin_cartesian_stft * ary).entries
        == xp.asarray(lin_lin_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_lin_cartesian_stft.entries) + xp.asarray(ary)
    lin_lin_cartesian_stft += ary
    assert xp.all(lin_lin_cartesian_stft.entries == expected)


def test_lin_uni_cartesian_stft(
    xp: ModuleType,
    lin_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all(
        (lin_uni_cartesian_stft + 1).entries == lin_uni_cartesian_stft.entries + 1
    )
    assert xp.all(
        (lin_uni_cartesian_stft * 2).entries == lin_uni_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = lin_uni_cartesian_stft.entries * 0.1
    assert xp.all(
        (lin_uni_cartesian_stft + ary).entries
        == xp.asarray(lin_uni_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_uni_cartesian_stft * ary).entries
        == xp.asarray(lin_uni_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_uni_cartesian_stft.entries) + xp.asarray(ary)
    lin_uni_cartesian_stft += ary
    assert xp.all(lin_uni_cartesian_stft.entries == expected)


def test_lin_ary_cartesian_stft(
    xp: ModuleType,
    lin_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all(
        (lin_ary_cartesian_stft + 1).entries == lin_ary_cartesian_stft.entries + 1
    )
    assert xp.all(
        (lin_ary_cartesian_stft * 2).entries == lin_ary_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = lin_ary_cartesian_stft.entries * 0.1
    assert xp.all(
        (lin_ary_cartesian_stft + ary).entries
        == xp.asarray(lin_ary_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_ary_cartesian_stft * ary).entries
        == xp.asarray(lin_ary_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_ary_cartesian_stft.entries) + xp.asarray(ary)
    lin_ary_cartesian_stft += ary
    assert xp.all(lin_ary_cartesian_stft.entries == expected)


def test_ary_lin_cartesian_stft(
    xp: ModuleType,
    ary_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[Linspace]]],
):
    # Scalar
    assert xp.all(
        (ary_lin_cartesian_stft + 1).entries == ary_lin_cartesian_stft.entries + 1
    )
    assert xp.all(
        (ary_lin_cartesian_stft * 2).entries == ary_lin_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = ary_lin_cartesian_stft.entries * 0.1
    assert xp.all(
        (ary_lin_cartesian_stft + ary).entries
        == xp.asarray(ary_lin_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_lin_cartesian_stft * ary).entries
        == xp.asarray(ary_lin_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_lin_cartesian_stft.entries) + xp.asarray(ary)
    ary_lin_cartesian_stft += ary
    assert xp.all(ary_lin_cartesian_stft.entries == expected)


def test_ary_uni_cartesian_stft(
    xp: ModuleType,
    ary_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all(
        (ary_uni_cartesian_stft + 1).entries == ary_uni_cartesian_stft.entries + 1
    )
    assert xp.all(
        (ary_uni_cartesian_stft * 2).entries == ary_uni_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = ary_uni_cartesian_stft.entries * 0.1
    assert xp.all(
        (ary_uni_cartesian_stft + ary).entries
        == xp.asarray(ary_uni_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_uni_cartesian_stft * ary).entries
        == xp.asarray(ary_uni_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_uni_cartesian_stft.entries) + xp.asarray(ary)
    ary_uni_cartesian_stft += ary
    assert xp.all(ary_uni_cartesian_stft.entries == expected)


def test_uni_lin_cartesian_stft(
    xp: ModuleType,
    uni_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[Linspace]]],
):
    # Scalar
    assert xp.all(
        (uni_lin_cartesian_stft + 1).entries == uni_lin_cartesian_stft.entries + 1
    )
    assert xp.all(
        (uni_lin_cartesian_stft * 2).entries == uni_lin_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = uni_lin_cartesian_stft.entries * 0.1
    assert xp.all(
        (uni_lin_cartesian_stft + ary).entries
        == xp.asarray(uni_lin_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_lin_cartesian_stft * ary).entries
        == xp.asarray(uni_lin_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_lin_cartesian_stft.entries) + xp.asarray(ary)
    uni_lin_cartesian_stft += ary
    assert xp.all(uni_lin_cartesian_stft.entries == expected)


def test_uni_uni_cartesian_stft(
    xp: ModuleType,
    uni_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all(
        (uni_uni_cartesian_stft + 1).entries == uni_uni_cartesian_stft.entries + 1
    )
    assert xp.all(
        (uni_uni_cartesian_stft * 2).entries == uni_uni_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = uni_uni_cartesian_stft.entries * 0.1
    assert xp.all(
        (uni_uni_cartesian_stft + ary).entries
        == xp.asarray(uni_uni_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_uni_cartesian_stft * ary).entries
        == xp.asarray(uni_uni_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_uni_cartesian_stft.entries) + xp.asarray(ary)
    uni_uni_cartesian_stft += ary
    assert xp.all(uni_uni_cartesian_stft.entries == expected)


def test_uni_ary_cartesian_stft(
    xp: ModuleType,
    uni_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all(
        (uni_ary_cartesian_stft + 1).entries == uni_ary_cartesian_stft.entries + 1
    )
    assert xp.all(
        (uni_ary_cartesian_stft * 2).entries == uni_ary_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = uni_ary_cartesian_stft.entries * 0.1
    assert xp.all(
        (uni_ary_cartesian_stft + ary).entries
        == xp.asarray(uni_ary_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_ary_cartesian_stft * ary).entries
        == xp.asarray(uni_ary_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_ary_cartesian_stft.entries) + xp.asarray(ary)
    uni_ary_cartesian_stft += ary
    assert xp.all(uni_ary_cartesian_stft.entries == expected)


def test_ary_ary_cartesian_stft(
    xp: ModuleType,
    ary_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all(
        (ary_ary_cartesian_stft + 1).entries == ary_ary_cartesian_stft.entries + 1
    )
    assert xp.all(
        (ary_ary_cartesian_stft * 2).entries == ary_ary_cartesian_stft.entries * 2
    )
    # Non-scalar
    ary = ary_ary_cartesian_stft.entries * 0.1
    assert xp.all(
        (ary_ary_cartesian_stft + ary).entries
        == xp.asarray(ary_ary_cartesian_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_ary_cartesian_stft * ary).entries
        == xp.asarray(ary_ary_cartesian_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_ary_cartesian_stft.entries) + xp.asarray(ary)
    ary_ary_cartesian_stft += ary
    assert xp.all(ary_ary_cartesian_stft.entries == expected)


def test_lin_lin_sparse_stft(
    xp: ModuleType,
    lin_lin_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((lin_lin_sparse_stft + 1).entries == lin_lin_sparse_stft.entries + 1)
    assert xp.all((lin_lin_sparse_stft * 2).entries == lin_lin_sparse_stft.entries * 2)
    # Non-scalar
    ary = lin_lin_sparse_stft.entries * 0.1
    assert xp.all(
        (lin_lin_sparse_stft + ary).entries
        == xp.asarray(lin_lin_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_lin_sparse_stft * ary).entries
        == xp.asarray(lin_lin_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_lin_sparse_stft.entries) + xp.asarray(ary)
    lin_lin_sparse_stft += ary
    assert xp.all(lin_lin_sparse_stft.entries == expected)


def test_lin_uni_sparse_stft(
    xp: ModuleType,
    lin_uni_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all((lin_uni_sparse_stft + 1).entries == lin_uni_sparse_stft.entries + 1)
    assert xp.all((lin_uni_sparse_stft * 2).entries == lin_uni_sparse_stft.entries * 2)
    # Non-scalar
    ary = lin_uni_sparse_stft.entries * 0.1
    assert xp.all(
        (lin_uni_sparse_stft + ary).entries
        == xp.asarray(lin_uni_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_uni_sparse_stft * ary).entries
        == xp.asarray(lin_uni_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_uni_sparse_stft.entries) + xp.asarray(ary)
    lin_uni_sparse_stft += ary
    assert xp.all(lin_uni_sparse_stft.entries == expected)


def test_lin_ary_sparse_stft(
    xp: ModuleType,
    lin_ary_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all((lin_ary_sparse_stft + 1).entries == lin_ary_sparse_stft.entries + 1)
    assert xp.all((lin_ary_sparse_stft * 2).entries == lin_ary_sparse_stft.entries * 2)
    # Non-scalar
    ary = lin_ary_sparse_stft.entries * 0.1
    assert xp.all(
        (lin_ary_sparse_stft + ary).entries
        == xp.asarray(lin_ary_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_ary_sparse_stft * ary).entries
        == xp.asarray(lin_ary_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_ary_sparse_stft.entries) + xp.asarray(ary)
    lin_ary_sparse_stft += ary
    assert xp.all(lin_ary_sparse_stft.entries == expected)


def test_uni_lin_sparse_stft(
    xp: ModuleType,
    uni_lin_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((uni_lin_sparse_stft + 1).entries == uni_lin_sparse_stft.entries + 1)
    assert xp.all((uni_lin_sparse_stft * 2).entries == uni_lin_sparse_stft.entries * 2)
    # Non-scalar
    ary = uni_lin_sparse_stft.entries * 0.1
    assert xp.all(
        (uni_lin_sparse_stft + ary).entries
        == xp.asarray(uni_lin_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_lin_sparse_stft * ary).entries
        == xp.asarray(uni_lin_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_lin_sparse_stft.entries) + xp.asarray(ary)
    uni_lin_sparse_stft += ary
    assert xp.all(uni_lin_sparse_stft.entries == expected)


def test_uni_uni_sparse_stft(
    xp: ModuleType,
    uni_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all((uni_uni_sparse_stft + 1).entries == uni_uni_sparse_stft.entries + 1)
    assert xp.all((uni_uni_sparse_stft * 2).entries == uni_uni_sparse_stft.entries * 2)
    # Non-scalar
    ary = uni_uni_sparse_stft.entries * 0.1
    assert xp.all(
        (uni_uni_sparse_stft + ary).entries
        == xp.asarray(uni_uni_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_uni_sparse_stft * ary).entries
        == xp.asarray(uni_uni_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_uni_sparse_stft.entries) + xp.asarray(ary)
    uni_uni_sparse_stft += ary
    assert xp.all(uni_uni_sparse_stft.entries == expected)


def test_uni_ary_sparse_stft(
    xp: ModuleType,
    uni_ary_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all((uni_ary_sparse_stft + 1).entries == uni_ary_sparse_stft.entries + 1)
    assert xp.all((uni_ary_sparse_stft * 2).entries == uni_ary_sparse_stft.entries * 2)
    # Non-scalar
    ary = uni_ary_sparse_stft.entries * 0.1
    assert xp.all(
        (uni_ary_sparse_stft + ary).entries
        == xp.asarray(uni_ary_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (uni_ary_sparse_stft * ary).entries
        == xp.asarray(uni_ary_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(uni_ary_sparse_stft.entries) + xp.asarray(ary)
    uni_ary_sparse_stft += ary
    assert xp.all(uni_ary_sparse_stft.entries == expected)


def test_ary_lin_sparse_stft(
    xp: ModuleType,
    ary_lin_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((ary_lin_sparse_stft + 1).entries == ary_lin_sparse_stft.entries + 1)
    assert xp.all((ary_lin_sparse_stft * 2).entries == ary_lin_sparse_stft.entries * 2)
    # Non-scalar
    ary = ary_lin_sparse_stft.entries * 0.1
    assert xp.all(
        (ary_lin_sparse_stft + ary).entries
        == xp.asarray(ary_lin_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_lin_sparse_stft * ary).entries
        == xp.asarray(ary_lin_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_lin_sparse_stft.entries) + xp.asarray(ary)
    ary_lin_sparse_stft += ary
    assert xp.all(ary_lin_sparse_stft.entries == expected)


def test_ary_uni_sparse_stft(
    xp: ModuleType,
    ary_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all((ary_uni_sparse_stft + 1).entries == ary_uni_sparse_stft.entries + 1)
    assert xp.all((ary_uni_sparse_stft * 2).entries == ary_uni_sparse_stft.entries * 2)
    # Non-scalar
    ary = ary_uni_sparse_stft.entries * 0.1
    assert xp.all(
        (ary_uni_sparse_stft + ary).entries
        == xp.asarray(ary_uni_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_uni_sparse_stft * ary).entries
        == xp.asarray(ary_uni_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_uni_sparse_stft.entries) + xp.asarray(ary)
    ary_uni_sparse_stft += ary
    assert xp.all(ary_uni_sparse_stft.entries == expected)


def test_ary_ary_sparse_stft(
    xp: ModuleType,
    ary_ary_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    # Scalar
    assert xp.all((ary_ary_sparse_stft + 1).entries == ary_ary_sparse_stft.entries + 1)
    assert xp.all((ary_ary_sparse_stft * 2).entries == ary_ary_sparse_stft.entries * 2)
    # Non-scalar
    ary = ary_ary_sparse_stft.entries * 0.1
    assert xp.all(
        (ary_ary_sparse_stft + ary).entries
        == xp.asarray(ary_ary_sparse_stft.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (ary_ary_sparse_stft * ary).entries
        == xp.asarray(ary_ary_sparse_stft.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(ary_ary_sparse_stft.entries) + xp.asarray(ary)
    ary_ary_sparse_stft += ary
    assert xp.all(ary_ary_sparse_stft.entries == expected)


def test_lin_lin_cartesian_wdm(
    xp: ModuleType,
    lin_lin_cartesian_wdm: WDM[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all(
        (lin_lin_cartesian_wdm + 1).entries == lin_lin_cartesian_wdm.entries + 1
    )
    assert xp.all(
        (lin_lin_cartesian_wdm * 2).entries == lin_lin_cartesian_wdm.entries * 2
    )
    # Non-scalar
    ary = lin_lin_cartesian_wdm.entries * 0.1
    assert xp.all(
        (lin_lin_cartesian_wdm + ary).entries
        == xp.asarray(lin_lin_cartesian_wdm.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_lin_cartesian_wdm * ary).entries
        == xp.asarray(lin_lin_cartesian_wdm.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_lin_cartesian_wdm.entries) + xp.asarray(ary)
    lin_lin_cartesian_wdm += ary
    assert xp.all(lin_lin_cartesian_wdm.entries == expected)


def test_lin_lin_sparse_wdm(
    xp: ModuleType,
    lin_lin_sparse_wdm: WDM[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((lin_lin_sparse_wdm + 1).entries == lin_lin_sparse_wdm.entries + 1)
    assert xp.all((lin_lin_sparse_wdm * 2).entries == lin_lin_sparse_wdm.entries * 2)
    # Non-scalar
    ary = lin_lin_sparse_wdm.entries * 0.1
    assert xp.all(
        (lin_lin_sparse_wdm + ary).entries
        == xp.asarray(lin_lin_sparse_wdm.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (lin_lin_sparse_wdm * ary).entries
        == xp.asarray(lin_lin_sparse_wdm.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(lin_lin_sparse_wdm.entries) + xp.asarray(ary)
    lin_lin_sparse_wdm += ary
    assert xp.all(lin_lin_sparse_wdm.entries == expected)


## Data


def test_tsdata(xp: ModuleType, tsdata: TSData):
    # Scalar
    assert xp.all((tsdata + 1).entries == tsdata.entries + 1)
    assert xp.all((tsdata * 2).entries == tsdata.entries * 2)
    assert xp.all((tsdata * 2).entries == tsdata.entries * 2)
    # Non-scalar
    ary = tsdata.entries * 0.1
    assert xp.all(
        (tsdata + ary).entries == xp.asarray(tsdata.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (tsdata * ary).entries == xp.asarray(tsdata.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(tsdata.entries) + xp.asarray(ary)
    tsdata += ary
    assert xp.all(tsdata.entries == expected)


def test_fsdata(xp: ModuleType, fsdata: FSData):
    # Scalar
    assert xp.all((fsdata + 1).entries == fsdata.entries + 1)
    assert xp.all((fsdata * 2).entries == fsdata.entries * 2)
    # Non-scalar
    ary = fsdata.entries * 0.1
    assert xp.all(
        (fsdata + ary).entries == xp.asarray(fsdata.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (fsdata * ary).entries == xp.asarray(fsdata.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(fsdata.entries) + xp.asarray(ary)
    fsdata += ary
    assert xp.all(fsdata.entries == expected)


def test_stftdata_cartesian(
    xp: ModuleType,
    stftdata_cartesian: STFTData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((stftdata_cartesian + 1).entries == stftdata_cartesian.entries + 1)
    assert xp.all((stftdata_cartesian * 2).entries == stftdata_cartesian.entries * 2)
    # Non-scalar
    ary = stftdata_cartesian.entries * 0.1
    assert xp.all(
        (stftdata_cartesian + ary).entries
        == xp.asarray(stftdata_cartesian.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (stftdata_cartesian * ary).entries
        == xp.asarray(stftdata_cartesian.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(stftdata_cartesian.entries) + xp.asarray(ary)
    stftdata_cartesian += ary
    assert xp.all(stftdata_cartesian.entries == expected)


def test_stftdata_sparse(
    xp: ModuleType,
    stftdata_sparse: STFTData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((stftdata_sparse + 1).entries == stftdata_sparse.entries + 1)
    assert xp.all((stftdata_sparse * 2).entries == stftdata_sparse.entries * 2)
    # Non-scalar
    ary = stftdata_sparse.entries * 0.1
    assert xp.all(
        (stftdata_sparse + ary).entries
        == xp.asarray(stftdata_sparse.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (stftdata_sparse * ary).entries
        == xp.asarray(stftdata_sparse.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(stftdata_sparse.entries) + xp.asarray(ary)
    stftdata_sparse += ary
    assert xp.all(stftdata_sparse.entries == expected)


def test_wdmdata_cartesian(
    xp: ModuleType,
    wdmdata_cartesian: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((wdmdata_cartesian + 1).entries == wdmdata_cartesian.entries + 1)
    assert xp.all((wdmdata_cartesian * 2).entries == wdmdata_cartesian.entries * 2)
    # Non-scalar
    ary = wdmdata_cartesian.entries * 0.1
    assert xp.all(
        (wdmdata_cartesian + ary).entries
        == xp.asarray(wdmdata_cartesian.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (wdmdata_cartesian * ary).entries
        == xp.asarray(wdmdata_cartesian.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(wdmdata_cartesian.entries) + xp.asarray(ary)
    wdmdata_cartesian += ary
    assert xp.all(wdmdata_cartesian.entries == expected)


def test_wdmdata_sparse(
    xp: ModuleType,
    wdmdata_sparse: WDMData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert xp.all((wdmdata_sparse + 1).entries == wdmdata_sparse.entries + 1)
    assert xp.all((wdmdata_sparse * 2).entries == wdmdata_sparse.entries * 2)
    # Non-scalar
    ary = wdmdata_sparse.entries * 0.1
    assert xp.all(
        (wdmdata_sparse + ary).entries
        == xp.asarray(wdmdata_sparse.entries) + xp.asarray(ary)
    )
    assert xp.all(
        (wdmdata_sparse * ary).entries
        == xp.asarray(wdmdata_sparse.entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(wdmdata_sparse.entries) + xp.asarray(ary)
    wdmdata_sparse += ary
    assert xp.all(wdmdata_sparse.entries == expected)


## Waveforms


def test_harmonic_waveform(
    xp: ModuleType,
    harmonic_waveform: HarmonicWaveform[Harmonic, UniformFrequencySeries],
):
    # Scalar
    assert xp.all(
        (harmonic_waveform + 1)[(2, 2)].entries == harmonic_waveform[(2, 2)].entries + 1
    )
    assert xp.all(
        (harmonic_waveform * 2)[(2, 2)].entries == harmonic_waveform[(2, 2)].entries * 2
    )
    # Non-scalar
    ary = harmonic_waveform[(2, 2)].entries * 0.1
    assert xp.all(
        (harmonic_waveform + ary)[(2, 2)].entries
        == xp.asarray(harmonic_waveform[(2, 2)].entries) + xp.asarray(ary)
    )
    assert xp.all(
        (harmonic_waveform * ary)[(2, 2)].entries
        == xp.asarray(harmonic_waveform[(2, 2)].entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(harmonic_waveform[(2, 2)].entries) + xp.asarray(ary)
    harmonic_waveform += ary
    assert xp.all(harmonic_waveform[(2, 2)].entries == expected)


def test_projected_waveform(
    xp: ModuleType,
    projected_waveform: ProjectedWaveform[UniformFrequencySeries],
):
    # Scalar
    assert xp.all(
        (projected_waveform + 1)["X"].entries == projected_waveform["X"].entries + 1
    )
    assert xp.all(
        (projected_waveform * 2)["X"].entries == projected_waveform["X"].entries * 2
    )
    # Non-scalar
    ary = projected_waveform["X"].entries * 0.1
    assert xp.all(
        (projected_waveform + ary)["X"].entries
        == xp.asarray(projected_waveform["X"].entries) + xp.asarray(ary)
    )
    assert xp.all(
        (projected_waveform * ary)["X"].entries
        == xp.asarray(projected_waveform["X"].entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(projected_waveform["X"].entries) + xp.asarray(ary)
    projected_waveform += ary
    assert xp.all(projected_waveform["X"].entries == expected)


def test_harmonic_projected_waveform(
    xp: ModuleType,
    harmonic_projected_waveform: HarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
):
    # Scalar
    assert xp.all(
        (harmonic_projected_waveform + 1)[(2, 2)].entries
        == harmonic_projected_waveform[(2, 2)].entries + 1
    )
    assert xp.all(
        (harmonic_projected_waveform * 2)[(2, 2)].entries
        == harmonic_projected_waveform[(2, 2)].entries * 2
    )
    # Non-scalar
    ary = harmonic_projected_waveform[(2, 2)].entries * 0.1
    assert xp.all(
        (harmonic_projected_waveform + ary)[(2, 2)].entries
        == xp.asarray(harmonic_projected_waveform[(2, 2)].entries) + xp.asarray(ary)
    )
    assert xp.all(
        (harmonic_projected_waveform * ary)[(2, 2)].entries
        == xp.asarray(harmonic_projected_waveform[(2, 2)].entries) * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(harmonic_projected_waveform[(2, 2)].entries) + xp.asarray(ary)
    harmonic_projected_waveform += ary
    assert xp.all(harmonic_projected_waveform[(2, 2)].entries == expected)


def test_homogeneous_harmonic_projected_waveform(
    xp: ModuleType,
    homogeneous_harmonic_projected_waveform: HomogeneousHarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
):
    # Scalar
    assert xp.all(
        (homogeneous_harmonic_projected_waveform + 1)[(2, 2)].entries
        == homogeneous_harmonic_projected_waveform[(2, 2)].entries + 1
    )
    assert xp.all(
        (homogeneous_harmonic_projected_waveform * 2)[(2, 2)].entries
        == homogeneous_harmonic_projected_waveform[(2, 2)].entries * 2
    )
    # Non-scalar
    ary = homogeneous_harmonic_projected_waveform[(2, 2)].entries * 0.1
    assert xp.all(
        (homogeneous_harmonic_projected_waveform + ary)[(2, 2)].entries
        == xp.asarray(homogeneous_harmonic_projected_waveform[(2, 2)].entries)
        + xp.asarray(ary)
    )
    assert xp.all(
        (homogeneous_harmonic_projected_waveform * ary)[(2, 2)].entries
        == xp.asarray(homogeneous_harmonic_projected_waveform[(2, 2)].entries)
        * xp.asarray(ary)
    )
    # In-place
    expected = xp.asarray(
        homogeneous_harmonic_projected_waveform[(2, 2)].entries
    ) + xp.asarray(ary)
    homogeneous_harmonic_projected_waveform += ary
    assert xp.all(homogeneous_harmonic_projected_waveform[(2, 2)].entries == expected)
