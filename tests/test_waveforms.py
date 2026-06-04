import pytest

from typed_lisa_toolkit import (
    densify_phasor,
    densify_phasor_hpw,
    densify_phasor_hw,
    densify_phasor_pw,
    get_dense_maker,
    homogeneous_harmonic_projected_waveform,
    homogeneous_harmonic_waveform,
    phasor_to_fs_hpw,
    phasor_to_fs_hw,
    phasor_to_fs_pw,
    phasor_to_series,
    plus_cross_waveform,
    projected_waveform,
)
from typed_lisa_toolkit.types import (
    AnyArray,
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
    hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[AnyArray]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[AnyArray],
):
    with pytest.warns(DeprecationWarning, match="densify_phasor_hw"):
        _hhw = densify_phasor_hw(hw_phasor, linear_interpolator, dense_ary_freq_axis)
    assert isinstance(_hhw, HomogeneousHarmonicWaveform)
    hhw = densify_phasor(hw_phasor, linear_interpolator, dense_ary_freq_axis)
    assert isinstance(hhw, HomogeneousHarmonicWaveform)


def test_densify_phasor_pw(
    pw_freq_phasor: ProjectedWaveform[Phasor[Axis[AnyArray]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[AnyArray],
):
    with pytest.warns(DeprecationWarning, match="densify_phasor_pw"):
        _pw = densify_phasor_pw(
            pw_freq_phasor, linear_interpolator, dense_ary_freq_axis
        )
    assert isinstance(_pw, ProjectedWaveform)
    pw = densify_phasor(pw_freq_phasor, linear_interpolator, dense_ary_freq_axis)
    assert isinstance(pw, ProjectedWaveform)


def test_densify_phasor_hpw(
    hpw_freq_phasor: HarmonicProjectedWaveform[Harmonic, Phasor[Axis[AnyArray]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[AnyArray],
):
    with pytest.warns(DeprecationWarning, match="densify_phasor_hpw"):
        _hpw = densify_phasor_hpw(
            hpw_freq_phasor, linear_interpolator, dense_ary_freq_axis
        )
    assert isinstance(_hpw, HarmonicProjectedWaveform)
    hpw = densify_phasor(hpw_freq_phasor, linear_interpolator, dense_ary_freq_axis)
    assert isinstance(hpw, HomogeneousHarmonicProjectedWaveform)


def test_phasor_to_fs_hw(hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[AnyArray]]]):
    with pytest.warns(DeprecationWarning, match="phasor_to_series"):
        _fs_hw = phasor_to_fs_hw(hw_phasor)
    assert isinstance(_fs_hw, HarmonicWaveform)
    fs_hw = phasor_to_series(hw_phasor)
    assert isinstance(fs_hw, HarmonicWaveform)


def test_phasor_to_fs_pw(
    pw_freq_phasor: ProjectedWaveform[Phasor[Axis[AnyArray]]],
):
    with pytest.warns(DeprecationWarning, match="phasor_to_series"):
        _fs_pw = phasor_to_fs_pw(pw_freq_phasor)
    assert isinstance(_fs_pw, ProjectedWaveform)
    fs_pw = phasor_to_series(pw_freq_phasor)
    assert isinstance(fs_pw, ProjectedWaveform)


def test_phasor_to_fs_hpw(
    hpw_freq_phasor: HarmonicProjectedWaveform[Harmonic, Phasor[Axis[AnyArray]]],
):
    with pytest.warns(DeprecationWarning, match="phasor_to_series"):
        _fs_hpw = phasor_to_fs_hpw(hpw_freq_phasor)
    assert isinstance(_fs_hpw, HarmonicProjectedWaveform)
    fs_hpw = phasor_to_series(hpw_freq_phasor)
    assert isinstance(fs_hpw, HarmonicProjectedWaveform)


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
    hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[AnyArray]]],
):
    hhw = homogeneous_harmonic_waveform({mode: rep for mode, rep in hw_phasor.items()})
    kernel = hhw.get_kernel()
    assert kernel.shape[2] == len(hhw.harmonics)


def test_phasor_to_fs_hw_homogeneous_branch(
    hw_phasor: HarmonicWaveform[Harmonic, Phasor[Axis[AnyArray]]],
):
    hhw = homogeneous_harmonic_waveform({mode: rep for mode, rep in hw_phasor.items()})
    with pytest.warns(DeprecationWarning, match="phasor_to_series"):
        _fs_hhw = phasor_to_fs_hw(hhw)
    assert isinstance(_fs_hhw, type(hhw))
    fs_hhw = phasor_to_series(hhw)
    assert isinstance(fs_hhw, type(hhw))


def test_phasor_to_fs_hpw_homogeneous_branch(
    hhpw_freq_phasor: HomogeneousHarmonicProjectedWaveform[
        Harmonic, Phasor[Axis[AnyArray]]
    ],
):
    with pytest.warns(DeprecationWarning, match="phasor_to_series"):
        fs_hhpw = phasor_to_fs_hpw(hhpw_freq_phasor)
    assert isinstance(fs_hhpw, type(hhpw_freq_phasor))


def test_get_dense_maker(
    hpw_freq_phasor: HarmonicProjectedWaveform[Harmonic, Phasor[Axis[AnyArray]]],
    linear_interpolator: Interpolator,
    dense_ary_freq_axis: Axis[AnyArray],
):
    with pytest.warns(DeprecationWarning, match="get_dense_maker"):
        make = get_dense_maker(linear_interpolator)

    densifier = make(dense_ary_freq_axis, embed=False)
    dense_hpw = densifier(hpw_freq_phasor)
    assert isinstance(dense_hpw, HarmonicProjectedWaveform)

    densifier_embed = make(dense_ary_freq_axis, embed=True)
    dense_hhpw = densifier_embed(hpw_freq_phasor)
    homogeneous = homogeneous_harmonic_projected_waveform(
        {mode: dense_hhpw[mode] for mode in dense_hhpw}
    )
    assert isinstance(homogeneous, HomogeneousHarmonicProjectedWaveform)
