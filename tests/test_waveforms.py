import pytest

from typed_lisa_toolkit import (
    densify_phasor_hpw,
    densify_phasor_hw,
    densify_phasor_pw,
    get_dense_maker,
    homogeneous_harmonic_projected_waveform,
    homogeneous_harmonic_waveform,
    phasor_to_fs_hpw,
    phasor_to_fs_hw,
    phasor_to_fs_pw,
    plus_cross_waveform,
    projected_waveform,
)
from typed_lisa_toolkit.types import (
    Array,
    Axis,
    FrequencySeries,
    Harmonic,
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    HomogeneousHarmonicProjectedWaveform,
    HomogeneousHarmonicWaveform,
    Interpolator,
    Linspace,
    Phasor,
    ProjectedWaveform,
)


def test_densify_phasor_hw(
    hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[Array]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[Array],
):
    hhw = densify_phasor_hw(hw_phasor, linear_interpolator, dense_ary_freq_axis)
    assert isinstance(hhw, HomogeneousHarmonicWaveform)


def test_densify_phasor_pw(
    pw_phasor: ProjectedWaveform[Phasor[Axis[Array]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[Array],
):
    pw = densify_phasor_pw(pw_phasor, linear_interpolator, dense_ary_freq_axis)
    assert isinstance(pw, ProjectedWaveform)


def test_densify_phasor_hpw(
    hpw_phasor: HarmonicProjectedWaveform[Harmonic, Phasor[Axis[Array]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[Array],
):
    hpw = densify_phasor_hpw(hpw_phasor, linear_interpolator, dense_ary_freq_axis)
    assert isinstance(hpw, HomogeneousHarmonicProjectedWaveform)


def test_phasor_to_fs_hw(hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[Array]]]):
    fs_hw = phasor_to_fs_hw(hw_phasor)
    assert isinstance(fs_hw, HarmonicWaveform)


def test_phasor_to_fs_pw(
    pw_phasor: ProjectedWaveform[Phasor[Axis[Array]]],
):
    fs_pw = phasor_to_fs_pw(pw_phasor)
    assert isinstance(fs_pw, ProjectedWaveform)


def test_phasor_to_fs_hpw(
    hpw_phasor: HarmonicProjectedWaveform[Harmonic, Phasor[Axis[Array]]],
):
    fs_hw = phasor_to_fs_hpw(hpw_phasor)
    assert isinstance(fs_hw, HarmonicProjectedWaveform)


def test_plus_cross_waveform_accessors_and_kind(
    lin_freq_series: FrequencySeries[Axis[Linspace]],
):
    pcw = plus_cross_waveform({"plus": lin_freq_series, "cross": lin_freq_series})
    assert pcw.plus is pcw["plus"]
    assert pcw.cross is pcw["cross"]
    assert pcw.kind == lin_freq_series.kind


def test_plus_cross_waveform_rejects_invalid_mapping(
    lin_freq_series: FrequencySeries[Axis[Linspace]],
):
    bad_rep = type(lin_freq_series)(
        grid=lin_freq_series.grid,
        entries=lin_freq_series.xp.ones(
            (1, 1, 1, 1, 2), dtype=lin_freq_series.xp.float64
        ),
    )
    with pytest.raises(ValueError, match=r".+"):
        plus_cross_waveform({"plus": lin_freq_series, "cross": bad_rep})


def test_harmonic_projected_waveform_rejects_invalid_projected_mapping(
    lin_freq_series: FrequencySeries[Axis[Linspace]],
):
    bad_rep = type(lin_freq_series)(
        grid=lin_freq_series.grid,
        entries=lin_freq_series.xp.ones(
            (1, 1, 1, 1, 2), dtype=lin_freq_series.xp.float64
        ),
    )
    with pytest.raises(ValueError, match="Invalid representation"):
        _ = projected_waveform({"X": bad_rep})


def test_homogeneous_harmonic_waveform_kernel(
    hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[Array]]],
):
    hhw = homogeneous_harmonic_waveform({mode: rep for mode, rep in hw_phasor.items()})
    kernel = hhw.get_kernel()
    assert kernel.shape[2] == len(hhw.harmonics)


def test_phasor_to_fs_hw_homogeneous_branch(
    hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[Array]]],
):
    hhw = homogeneous_harmonic_waveform({mode: rep for mode, rep in hw_phasor.items()})
    fs_hhw = phasor_to_fs_hw(hhw)
    assert isinstance(fs_hhw, type(hhw))


def test_phasor_to_fs_hpw_homogeneous_branch(
    hhpw_phasor: HomogeneousHarmonicProjectedWaveform[Harmonic, Phasor[Axis[Array]]],
):
    fs_hhpw = phasor_to_fs_hpw(hhpw_phasor)
    assert isinstance(fs_hhpw, type(hhpw_phasor))


def test_get_dense_maker(
    hpw_phasor: HarmonicProjectedWaveform[Harmonic, Phasor[Axis[Array]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[Array],
):
    make = get_dense_maker(linear_interpolator)

    densifier = make(dense_ary_freq_axis, embed=False)
    dense_hpw = densifier(hpw_phasor)
    assert isinstance(dense_hpw, HarmonicProjectedWaveform)

    densifier_embed = make(dense_ary_freq_axis, embed=True)
    dense_hhpw = densifier_embed(hpw_phasor)
    homogeneous = homogeneous_harmonic_projected_waveform(
        {mode: dense_hhpw[mode] for mode in dense_hhpw}
    )
    assert isinstance(homogeneous, HomogeneousHarmonicProjectedWaveform)
