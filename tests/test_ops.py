import pytest

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


def test_lin_freq_series(lin_freq_series: UniformFrequencySeries):
    # Scalar
    assert ((lin_freq_series + 1).entries == lin_freq_series.entries + 1).all()
    assert ((lin_freq_series * 2).entries == lin_freq_series.entries * 2).all()
    # Non-scalar
    ary = lin_freq_series.entries * 0.1
    assert ((lin_freq_series + ary).entries == lin_freq_series.entries + ary).all()
    assert ((lin_freq_series * ary).entries == lin_freq_series.entries * ary).all()
    # In-place
    expected = lin_freq_series.entries + ary
    lin_freq_series += ary
    assert (lin_freq_series.entries == expected).all()


def test_uni_ary_freq_series(
    uni_ary_freq_series: FrequencySeries[Axis[Array]],
):
    # Scalar
    assert ((uni_ary_freq_series + 1).entries == uni_ary_freq_series.entries + 1).all()
    assert ((uni_ary_freq_series * 2).entries == uni_ary_freq_series.entries * 2).all()
    # Non-scalar
    ary = uni_ary_freq_series.entries * 0.1
    assert (
        (uni_ary_freq_series + ary).entries == uni_ary_freq_series.entries + ary
    ).all()
    assert (
        (uni_ary_freq_series * ary).entries == uni_ary_freq_series.entries * ary
    ).all()
    # In-place
    expected = uni_ary_freq_series.entries + ary
    uni_ary_freq_series += ary
    assert (uni_ary_freq_series.entries == expected).all()


def test_ary_freq_series(
    ary_freq_series: FrequencySeries[Axis[Array]],
):
    # Scalar
    assert ((ary_freq_series + 1).entries == ary_freq_series.entries + 1).all()
    assert ((ary_freq_series * 2).entries == ary_freq_series.entries * 2).all()
    # Non-scalar
    ary = ary_freq_series.entries * 0.1
    assert ((ary_freq_series + ary).entries == ary_freq_series.entries + ary).all()
    assert ((ary_freq_series * ary).entries == ary_freq_series.entries * ary).all()
    # In-place
    expected = ary_freq_series.entries + ary
    ary_freq_series += ary
    assert (ary_freq_series.entries == expected).all()


def test_lin_time_series(lin_time_series: UniformTimeSeries):
    # Scalar
    assert ((lin_time_series + 1).entries == lin_time_series.entries + 1).all()
    assert ((lin_time_series * 2).entries == lin_time_series.entries * 2).all()
    # Non-scalar
    ary = lin_time_series.entries * 0.1
    assert ((lin_time_series + ary).entries == lin_time_series.entries + ary).all()
    assert ((lin_time_series * ary).entries == lin_time_series.entries * ary).all()
    # In-place
    expected = lin_time_series.entries + ary
    lin_time_series += ary
    assert (lin_time_series.entries == expected).all()


def test_uni_ary_time_series(
    uni_ary_time_series: TimeSeries[Axis[Array]],
):
    # Scalar
    assert ((uni_ary_time_series + 1).entries == uni_ary_time_series.entries + 1).all()
    assert ((uni_ary_time_series * 2).entries == uni_ary_time_series.entries * 2).all()
    # Non-scalar
    ary = uni_ary_time_series.entries * 0.1
    assert (
        (uni_ary_time_series + ary).entries == uni_ary_time_series.entries + ary
    ).all()
    assert (
        (uni_ary_time_series * ary).entries == uni_ary_time_series.entries * ary
    ).all()
    # In-place
    expected = uni_ary_time_series.entries + ary
    uni_ary_time_series += ary
    assert (uni_ary_time_series.entries == expected).all()


def test_ary_time_series(
    ary_time_series: TimeSeries[Axis[Array]],
):
    # Scalar
    assert ((ary_time_series + 1).entries == ary_time_series.entries + 1).all()
    assert ((ary_time_series * 2).entries == ary_time_series.entries * 2).all()
    # Non-scalar
    ary = ary_time_series.entries * 0.1
    assert ((ary_time_series + ary).entries == ary_time_series.entries + ary).all()
    assert ((ary_time_series * ary).entries == ary_time_series.entries * ary).all()
    # In-place
    expected = ary_time_series.entries + ary
    ary_time_series += ary
    assert (ary_time_series.entries == expected).all()


def test_lin_phasor(lin_freq_phasor: Phasor[Axis[Linspace]]):
    # Catch error
    with pytest.raises(TypeError):
        _ = lin_freq_phasor + 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_uni_ary_phasor(uni_ary_phasor: Phasor[Axis[Array]]):
    # Catch error
    with pytest.raises(TypeError):
        _ = uni_ary_phasor + 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_ary_phasor(ary_phasor: Phasor[Axis[Array]]):
    # Catch error
    with pytest.raises(TypeError):
        _ = ary_phasor + 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


def test_lin_lin_cartesian_stft(
    lin_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert (
        (lin_lin_cartesian_stft + 1).entries == lin_lin_cartesian_stft.entries + 1
    ).all()
    assert (
        (lin_lin_cartesian_stft * 2).entries == lin_lin_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = lin_lin_cartesian_stft.entries * 0.1
    assert (
        (lin_lin_cartesian_stft + ary).entries == lin_lin_cartesian_stft.entries + ary
    ).all()
    assert (
        (lin_lin_cartesian_stft * ary).entries == lin_lin_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = lin_lin_cartesian_stft.entries + ary
    lin_lin_cartesian_stft += ary
    assert (lin_lin_cartesian_stft.entries == expected).all()


def test_lin_uni_cartesian_stft(
    lin_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[Array]]],
):
    # Scalar
    assert (
        (lin_uni_cartesian_stft + 1).entries == lin_uni_cartesian_stft.entries + 1
    ).all()
    assert (
        (lin_uni_cartesian_stft * 2).entries == lin_uni_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = lin_uni_cartesian_stft.entries * 0.1
    assert (
        (lin_uni_cartesian_stft + ary).entries == lin_uni_cartesian_stft.entries + ary
    ).all()
    assert (
        (lin_uni_cartesian_stft * ary).entries == lin_uni_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = lin_uni_cartesian_stft.entries + ary
    lin_uni_cartesian_stft += ary
    assert (lin_uni_cartesian_stft.entries == expected).all()


def test_lin_ary_cartesian_stft(
    lin_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[Array]]],
):
    # Scalar
    assert (
        (lin_ary_cartesian_stft + 1).entries == lin_ary_cartesian_stft.entries + 1
    ).all()
    assert (
        (lin_ary_cartesian_stft * 2).entries == lin_ary_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = lin_ary_cartesian_stft.entries * 0.1
    assert (
        (lin_ary_cartesian_stft + ary).entries == lin_ary_cartesian_stft.entries + ary
    ).all()
    assert (
        (lin_ary_cartesian_stft * ary).entries == lin_ary_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = lin_ary_cartesian_stft.entries + ary
    lin_ary_cartesian_stft += ary
    assert (lin_ary_cartesian_stft.entries == expected).all()


def test_ary_lin_cartesian_stft(
    ary_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[Array], Axis[Linspace]]],
):
    # Scalar
    assert (
        (ary_lin_cartesian_stft + 1).entries == ary_lin_cartesian_stft.entries + 1
    ).all()
    assert (
        (ary_lin_cartesian_stft * 2).entries == ary_lin_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = ary_lin_cartesian_stft.entries * 0.1
    assert (
        (ary_lin_cartesian_stft + ary).entries == ary_lin_cartesian_stft.entries + ary
    ).all()
    assert (
        (ary_lin_cartesian_stft * ary).entries == ary_lin_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = ary_lin_cartesian_stft.entries + ary
    ary_lin_cartesian_stft += ary
    assert (ary_lin_cartesian_stft.entries == expected).all()


def test_ary_uni_cartesian_stft(
    ary_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert (
        (ary_uni_cartesian_stft + 1).entries == ary_uni_cartesian_stft.entries + 1
    ).all()
    assert (
        (ary_uni_cartesian_stft * 2).entries == ary_uni_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = ary_uni_cartesian_stft.entries * 0.1
    assert (
        (ary_uni_cartesian_stft + ary).entries == ary_uni_cartesian_stft.entries + ary
    ).all()
    assert (
        (ary_uni_cartesian_stft * ary).entries == ary_uni_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = ary_uni_cartesian_stft.entries + ary
    ary_uni_cartesian_stft += ary
    assert (ary_uni_cartesian_stft.entries == expected).all()


def test_uni_lin_cartesian_stft(
    uni_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[Array], Axis[Linspace]]],
):
    # Scalar
    assert (
        (uni_lin_cartesian_stft + 1).entries == uni_lin_cartesian_stft.entries + 1
    ).all()
    assert (
        (uni_lin_cartesian_stft * 2).entries == uni_lin_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = uni_lin_cartesian_stft.entries * 0.1
    assert (
        (uni_lin_cartesian_stft + ary).entries == uni_lin_cartesian_stft.entries + ary
    ).all()
    assert (
        (uni_lin_cartesian_stft * ary).entries == uni_lin_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = uni_lin_cartesian_stft.entries + ary
    uni_lin_cartesian_stft += ary
    assert (uni_lin_cartesian_stft.entries == expected).all()


def test_uni_uni_cartesian_stft(
    uni_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert (
        (uni_uni_cartesian_stft + 1).entries == uni_uni_cartesian_stft.entries + 1
    ).all()
    assert (
        (uni_uni_cartesian_stft * 2).entries == uni_uni_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = uni_uni_cartesian_stft.entries * 0.1
    assert (
        (uni_uni_cartesian_stft + ary).entries == uni_uni_cartesian_stft.entries + ary
    ).all()
    assert (
        (uni_uni_cartesian_stft * ary).entries == uni_uni_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = uni_uni_cartesian_stft.entries + ary
    uni_uni_cartesian_stft += ary
    assert (uni_uni_cartesian_stft.entries == expected).all()


def test_uni_ary_cartesian_stft(
    uni_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert (
        (uni_ary_cartesian_stft + 1).entries == uni_ary_cartesian_stft.entries + 1
    ).all()
    assert (
        (uni_ary_cartesian_stft * 2).entries == uni_ary_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = uni_ary_cartesian_stft.entries * 0.1
    assert (
        (uni_ary_cartesian_stft + ary).entries == uni_ary_cartesian_stft.entries + ary
    ).all()
    assert (
        (uni_ary_cartesian_stft * ary).entries == uni_ary_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = uni_ary_cartesian_stft.entries + ary
    uni_ary_cartesian_stft += ary
    assert (uni_ary_cartesian_stft.entries == expected).all()


def test_ary_ary_cartesian_stft(
    ary_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert (
        (ary_ary_cartesian_stft + 1).entries == ary_ary_cartesian_stft.entries + 1
    ).all()
    assert (
        (ary_ary_cartesian_stft * 2).entries == ary_ary_cartesian_stft.entries * 2
    ).all()
    # Non-scalar
    ary = ary_ary_cartesian_stft.entries * 0.1
    assert (
        (ary_ary_cartesian_stft + ary).entries == ary_ary_cartesian_stft.entries + ary
    ).all()
    assert (
        (ary_ary_cartesian_stft * ary).entries == ary_ary_cartesian_stft.entries * ary
    ).all()
    # In-place
    expected = ary_ary_cartesian_stft.entries + ary
    ary_ary_cartesian_stft += ary
    assert (ary_ary_cartesian_stft.entries == expected).all()


def test_lin_lin_sparse_stft(
    lin_lin_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert ((lin_lin_sparse_stft + 1).entries == lin_lin_sparse_stft.entries + 1).all()
    assert ((lin_lin_sparse_stft * 2).entries == lin_lin_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = lin_lin_sparse_stft.entries * 0.1
    assert (
        (lin_lin_sparse_stft + ary).entries == lin_lin_sparse_stft.entries + ary
    ).all()
    assert (
        (lin_lin_sparse_stft * ary).entries == lin_lin_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = lin_lin_sparse_stft.entries + ary
    lin_lin_sparse_stft += ary
    assert (lin_lin_sparse_stft.entries == expected).all()


def test_lin_uni_sparse_stft(
    lin_uni_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[Array]]],
):
    # Scalar
    assert ((lin_uni_sparse_stft + 1).entries == lin_uni_sparse_stft.entries + 1).all()
    assert ((lin_uni_sparse_stft * 2).entries == lin_uni_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = lin_uni_sparse_stft.entries * 0.1
    assert (
        (lin_uni_sparse_stft + ary).entries == lin_uni_sparse_stft.entries + ary
    ).all()
    assert (
        (lin_uni_sparse_stft * ary).entries == lin_uni_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = lin_uni_sparse_stft.entries + ary
    lin_uni_sparse_stft += ary
    assert (lin_uni_sparse_stft.entries == expected).all()


def test_lin_ary_sparse_stft(
    lin_ary_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[Array]]],
):
    # Scalar
    assert ((lin_ary_sparse_stft + 1).entries == lin_ary_sparse_stft.entries + 1).all()
    assert ((lin_ary_sparse_stft * 2).entries == lin_ary_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = lin_ary_sparse_stft.entries * 0.1
    assert (
        (lin_ary_sparse_stft + ary).entries == lin_ary_sparse_stft.entries + ary
    ).all()
    assert (
        (lin_ary_sparse_stft * ary).entries == lin_ary_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = lin_ary_sparse_stft.entries + ary
    lin_ary_sparse_stft += ary
    assert (lin_ary_sparse_stft.entries == expected).all()


def test_uni_lin_sparse_stft(
    uni_lin_sparse_stft: STFT[Grid2DSparse[Axis[Array], Axis[Linspace]]],
):
    # Scalar
    assert ((uni_lin_sparse_stft + 1).entries == uni_lin_sparse_stft.entries + 1).all()
    assert ((uni_lin_sparse_stft * 2).entries == uni_lin_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = uni_lin_sparse_stft.entries * 0.1
    assert (
        (uni_lin_sparse_stft + ary).entries == uni_lin_sparse_stft.entries + ary
    ).all()
    assert (
        (uni_lin_sparse_stft * ary).entries == uni_lin_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = uni_lin_sparse_stft.entries + ary
    uni_lin_sparse_stft += ary
    assert (uni_lin_sparse_stft.entries == expected).all()


def test_uni_uni_sparse_stft(
    uni_uni_sparse_stft: STFT[Grid2DSparse[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert ((uni_uni_sparse_stft + 1).entries == uni_uni_sparse_stft.entries + 1).all()
    assert ((uni_uni_sparse_stft * 2).entries == uni_uni_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = uni_uni_sparse_stft.entries * 0.1
    assert (
        (uni_uni_sparse_stft + ary).entries == uni_uni_sparse_stft.entries + ary
    ).all()
    assert (
        (uni_uni_sparse_stft * ary).entries == uni_uni_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = uni_uni_sparse_stft.entries + ary
    uni_uni_sparse_stft += ary
    assert (uni_uni_sparse_stft.entries == expected).all()


def test_uni_ary_sparse_stft(
    uni_ary_sparse_stft: STFT[Grid2DSparse[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert ((uni_ary_sparse_stft + 1).entries == uni_ary_sparse_stft.entries + 1).all()
    assert ((uni_ary_sparse_stft * 2).entries == uni_ary_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = uni_ary_sparse_stft.entries * 0.1
    assert (
        (uni_ary_sparse_stft + ary).entries == uni_ary_sparse_stft.entries + ary
    ).all()
    assert (
        (uni_ary_sparse_stft * ary).entries == uni_ary_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = uni_ary_sparse_stft.entries + ary
    uni_ary_sparse_stft += ary
    assert (uni_ary_sparse_stft.entries == expected).all()


def test_ary_lin_sparse_stft(
    ary_lin_sparse_stft: STFT[Grid2DSparse[Axis[Array], Axis[Linspace]]],
):
    # Scalar
    assert ((ary_lin_sparse_stft + 1).entries == ary_lin_sparse_stft.entries + 1).all()
    assert ((ary_lin_sparse_stft * 2).entries == ary_lin_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = ary_lin_sparse_stft.entries * 0.1
    assert (
        (ary_lin_sparse_stft + ary).entries == ary_lin_sparse_stft.entries + ary
    ).all()
    assert (
        (ary_lin_sparse_stft * ary).entries == ary_lin_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = ary_lin_sparse_stft.entries + ary
    ary_lin_sparse_stft += ary
    assert (ary_lin_sparse_stft.entries == expected).all()


def test_ary_uni_sparse_stft(
    ary_uni_sparse_stft: STFT[Grid2DSparse[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert ((ary_uni_sparse_stft + 1).entries == ary_uni_sparse_stft.entries + 1).all()
    assert ((ary_uni_sparse_stft * 2).entries == ary_uni_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = ary_uni_sparse_stft.entries * 0.1
    assert (
        (ary_uni_sparse_stft + ary).entries == ary_uni_sparse_stft.entries + ary
    ).all()
    assert (
        (ary_uni_sparse_stft * ary).entries == ary_uni_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = ary_uni_sparse_stft.entries + ary
    ary_uni_sparse_stft += ary
    assert (ary_uni_sparse_stft.entries == expected).all()


def test_ary_ary_sparse_stft(
    ary_ary_sparse_stft: STFT[Grid2DSparse[Axis[Array], Axis[Array]]],
):
    # Scalar
    assert ((ary_ary_sparse_stft + 1).entries == ary_ary_sparse_stft.entries + 1).all()
    assert ((ary_ary_sparse_stft * 2).entries == ary_ary_sparse_stft.entries * 2).all()
    # Non-scalar
    ary = ary_ary_sparse_stft.entries * 0.1
    assert (
        (ary_ary_sparse_stft + ary).entries == ary_ary_sparse_stft.entries + ary
    ).all()
    assert (
        (ary_ary_sparse_stft * ary).entries == ary_ary_sparse_stft.entries * ary
    ).all()
    # In-place
    expected = ary_ary_sparse_stft.entries + ary
    ary_ary_sparse_stft += ary
    assert (ary_ary_sparse_stft.entries == expected).all()


def test_lin_lin_cartesian_wdm(
    lin_lin_cartesian_wdm: WDM[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert (
        (lin_lin_cartesian_wdm + 1).entries == lin_lin_cartesian_wdm.entries + 1
    ).all()
    assert (
        (lin_lin_cartesian_wdm * 2).entries == lin_lin_cartesian_wdm.entries * 2
    ).all()
    # Non-scalar
    ary = lin_lin_cartesian_wdm.entries * 0.1
    assert (
        (lin_lin_cartesian_wdm + ary).entries == lin_lin_cartesian_wdm.entries + ary
    ).all()
    assert (
        (lin_lin_cartesian_wdm * ary).entries == lin_lin_cartesian_wdm.entries * ary
    ).all()
    # In-place
    expected = lin_lin_cartesian_wdm.entries + ary
    lin_lin_cartesian_wdm += ary
    assert (lin_lin_cartesian_wdm.entries == expected).all()


def test_lin_lin_sparse_wdm(
    lin_lin_sparse_wdm: WDM[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert ((lin_lin_sparse_wdm + 1).entries == lin_lin_sparse_wdm.entries + 1).all()
    assert ((lin_lin_sparse_wdm * 2).entries == lin_lin_sparse_wdm.entries * 2).all()
    # Non-scalar
    ary = lin_lin_sparse_wdm.entries * 0.1
    assert (
        (lin_lin_sparse_wdm + ary).entries == lin_lin_sparse_wdm.entries + ary
    ).all()
    assert (
        (lin_lin_sparse_wdm * ary).entries == lin_lin_sparse_wdm.entries * ary
    ).all()
    # In-place
    expected = lin_lin_sparse_wdm.entries + ary
    lin_lin_sparse_wdm += ary
    assert (lin_lin_sparse_wdm.entries == expected).all()


## Data


def test_tsdata(tsdata: TSData):
    # Scalar
    assert ((tsdata + 1).entries == tsdata.entries + 1).all
    assert ((tsdata * 2).entries == tsdata.entries * 2).all()
    # Non-scalar
    ary = tsdata.entries * 0.1
    assert ((tsdata + ary).entries == tsdata.entries + ary).all()
    assert ((tsdata * ary).entries == tsdata.entries * ary).all()
    # In-place
    expected = tsdata.entries + ary
    tsdata += ary
    assert (tsdata.entries == expected).all()


def test_fsdata(fsdata: FSData):
    # Scalar
    assert ((fsdata + 1).entries == fsdata.entries + 1).all
    assert ((fsdata * 2).entries == fsdata.entries * 2).all()
    # Non-scalar
    ary = fsdata.entries * 0.1
    assert ((fsdata + ary).entries == fsdata.entries + ary).all()
    assert ((fsdata * ary).entries == fsdata.entries * ary).all()
    # In-place
    expected = fsdata.entries + ary
    fsdata += ary
    assert (fsdata.entries == expected).all()


def test_stftdata_cartesian(
    stftdata_cartesian: STFTData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert ((stftdata_cartesian + 1).entries == stftdata_cartesian.entries + 1).all()
    assert ((stftdata_cartesian * 2).entries == stftdata_cartesian.entries * 2).all()
    # Non-scalar
    ary = stftdata_cartesian.entries * 0.1
    assert (
        (stftdata_cartesian + ary).entries == stftdata_cartesian.entries + ary
    ).all()
    assert (
        (stftdata_cartesian * ary).entries == stftdata_cartesian.entries * ary
    ).all()
    # In-place
    expected = stftdata_cartesian.entries + ary
    stftdata_cartesian += ary
    assert (stftdata_cartesian.entries == expected).all()


def test_stftdata_sparse(
    stftdata_sparse: STFTData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert ((stftdata_sparse + 1).entries == stftdata_sparse.entries + 1).all()
    assert ((stftdata_sparse * 2).entries == stftdata_sparse.entries * 2).all()
    # Non-scalar
    ary = stftdata_sparse.entries * 0.1
    assert ((stftdata_sparse + ary).entries == stftdata_sparse.entries + ary).all()
    assert ((stftdata_sparse * ary).entries == stftdata_sparse.entries * ary).all()
    # In-place
    expected = stftdata_sparse.entries + ary
    stftdata_sparse += ary
    assert (stftdata_sparse.entries == expected).all()


def test_wdmdata_cartesian(
    wdmdata_cartesian: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert ((wdmdata_cartesian + 1).entries == wdmdata_cartesian.entries + 1).all()
    assert ((wdmdata_cartesian * 2).entries == wdmdata_cartesian.entries * 2).all()
    # Non-scalar
    ary = wdmdata_cartesian.entries * 0.1
    assert ((wdmdata_cartesian + ary).entries == wdmdata_cartesian.entries + ary).all()
    assert ((wdmdata_cartesian * ary).entries == wdmdata_cartesian.entries * ary).all()
    # In-place
    expected = wdmdata_cartesian.entries + ary
    wdmdata_cartesian += ary
    assert (wdmdata_cartesian.entries == expected).all()


def test_wdmdata_sparse(
    wdmdata_sparse: WDMData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    # Scalar
    assert ((wdmdata_sparse + 1).entries == wdmdata_sparse.entries + 1).all()
    assert ((wdmdata_sparse * 2).entries == wdmdata_sparse.entries * 2).all()
    # Non-scalar
    ary = wdmdata_sparse.entries * 0.1
    assert ((wdmdata_sparse + ary).entries == wdmdata_sparse.entries + ary).all()
    assert ((wdmdata_sparse * ary).entries == wdmdata_sparse.entries * ary).all()
    # In-place
    expected = wdmdata_sparse.entries + ary
    wdmdata_sparse += ary
    assert (wdmdata_sparse.entries == expected).all()


## Waveforms


def test_harmonic_waveform(
    harmonic_waveform: HarmonicWaveform[Harmonic, UniformFrequencySeries],
):
    # Scalar
    assert (
        (harmonic_waveform + 1)[(2, 2)].entries == harmonic_waveform[(2, 2)].entries + 1
    ).all()
    assert (
        (harmonic_waveform * 2)[(2, 2)].entries == harmonic_waveform[(2, 2)].entries * 2
    ).all()
    # Non-scalar
    ary = harmonic_waveform[(2, 2)].entries * 0.1
    assert (
        (harmonic_waveform + ary)[(2, 2)].entries
        == harmonic_waveform[(2, 2)].entries + ary
    ).all()
    assert (
        (harmonic_waveform * ary)[(2, 2)].entries
        == harmonic_waveform[(2, 2)].entries * ary
    ).all()
    # In-place
    expected = harmonic_waveform[(2, 2)].entries + ary
    harmonic_waveform += ary
    assert (harmonic_waveform[(2, 2)].entries == expected).all()


def test_projected_waveform(
    projected_waveform: ProjectedWaveform[UniformFrequencySeries],
):
    # Scalar
    assert (
        (projected_waveform + 1)["X"].entries == projected_waveform["X"].entries + 1
    ).all()
    assert (
        (projected_waveform * 2)["X"].entries == projected_waveform["X"].entries * 2
    ).all()
    # Non-scalar
    ary = projected_waveform["X"].entries * 0.1
    assert (
        (projected_waveform + ary)["X"].entries == projected_waveform["X"].entries + ary
    ).all()
    assert (
        (projected_waveform * ary)["X"].entries == projected_waveform["X"].entries * ary
    ).all()
    # In-place
    expected = projected_waveform["X"].entries + ary
    projected_waveform += ary
    assert (projected_waveform["X"].entries == expected).all()


def test_harmonic_projected_waveform(
    harmonic_projected_waveform: HarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
):
    # Scalar
    assert (
        (harmonic_projected_waveform + 1)[(2, 2)].entries
        == harmonic_projected_waveform[(2, 2)].entries + 1
    ).all()
    assert (
        (harmonic_projected_waveform * 2)[(2, 2)].entries
        == harmonic_projected_waveform[(2, 2)].entries * 2
    ).all()
    # Non-scalar
    ary = harmonic_projected_waveform[(2, 2)].entries * 0.1
    assert (
        (harmonic_projected_waveform + ary)[(2, 2)].entries
        == harmonic_projected_waveform[(2, 2)].entries + ary
    ).all()
    assert (
        (harmonic_projected_waveform * ary)[(2, 2)].entries
        == harmonic_projected_waveform[(2, 2)].entries * ary
    ).all()
    # In-place
    expected = harmonic_projected_waveform[(2, 2)].entries + ary
    harmonic_projected_waveform += ary
    assert (harmonic_projected_waveform[(2, 2)].entries == expected).all()


def test_homogeneous_harmonic_projected_waveform(
    homogeneous_harmonic_projected_waveform: HomogeneousHarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
):
    # Scalar
    assert (
        (homogeneous_harmonic_projected_waveform + 1)[(2, 2)].entries
        == homogeneous_harmonic_projected_waveform[(2, 2)].entries + 1
    ).all()
    assert (
        (homogeneous_harmonic_projected_waveform * 2)[(2, 2)].entries
        == homogeneous_harmonic_projected_waveform[(2, 2)].entries * 2
    ).all()
    # Non-scalar
    ary = homogeneous_harmonic_projected_waveform[(2, 2)].entries * 0.1
    assert (
        (homogeneous_harmonic_projected_waveform + ary)[(2, 2)].entries
        == homogeneous_harmonic_projected_waveform[(2, 2)].entries + ary
    ).all()
    assert (
        (homogeneous_harmonic_projected_waveform * ary)[(2, 2)].entries
        == homogeneous_harmonic_projected_waveform[(2, 2)].entries * ary
    ).all()
    # In-place
    expected = homogeneous_harmonic_projected_waveform[(2, 2)].entries + ary
    homogeneous_harmonic_projected_waveform += ary
    assert (homogeneous_harmonic_projected_waveform[(2, 2)].entries == expected).all()
