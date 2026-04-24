"""Tests for waveform containers with JAX arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from typed_lisa_toolkit import (
    axis,
    densify_phasor,
    densify_phasor_hpw,
    densify_phasor_hw,
    densify_phasor_pw,
    get_dense_maker,
    harmonic_projected_waveform,
    harmonic_waveform,
    hhpw,
    homogeneous_harmonic_projected_waveform,
    hpw,
    hw,
    phasor_to_fs_hpw,
    phasor_to_fs_hw,
    phasor_to_fs_pw,
    projected_waveform,
    pw,
    sum_harmonics,
)
from typed_lisa_toolkit.types import modes

if TYPE_CHECKING:
    from conftest import (
        build_harmonic_projected_frequency_waveform,
        build_harmonic_projected_phasor_waveform,
        build_harmonic_projected_waveform_constructor,
        build_harmonic_projected_waveform_from_mapping,
        build_harmonic_waveform_constructor,
        build_harmonic_waveform_frequency_series,
        build_harmonic_waveform_from_mapping,
        build_homogeneous_harmonic_projected_waveform_constructor,
        build_projected_waveform_from_mapping,
        build_test_phasor,
    )

jax.config.update("jax_enable_x64", val=True)


def _linear_interpolator(x, y):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y)

    def _interp(x_new):
        values = np.interp(np.asarray(x_new, dtype=np.float64), x_arr, y_arr)
        return jnp.asarray(values)

    return _interp


class TestHarmonicWaveformJAX:
    def test_constructor_normalizes_tuple_mode_keys(self):
        wf = build_harmonic_waveform_constructor(jnp)

        assert tuple(wf.keys()) == (modes.Harmonic(2, 2), modes.Harmonic(3, 3))

    def test_domain_and_pick(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        wf = case["actual"]["wf"]

        assert wf.domain == "frequency"

        picked = wf.pick(case["expected"]["mode_33"])
        assert tuple(picked.keys()) == (case["expected"]["mode_33"],)
        npt.assert_allclose(
            np.asarray(picked[case["expected"]["mode_33"]].entries),
            np.asarray(case["expected"]["wf_33"].entries),
        )

    def test_pick_missing_mode_raises(self):
        case = build_harmonic_waveform_frequency_series(jnp)

        with pytest.raises(KeyError):
            case["actual"]["wf"].pick(modes.Harmonic(9, 9))

    def test_pick_tuple_preserves_requested_order(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        wf = case["actual"]["wf"]

        picked = wf.pick((case["expected"]["mode_33"], case["expected"]["mode_22"]))

        assert tuple(picked.keys()) == (
            case["expected"]["mode_33"],
            case["expected"]["mode_22"],
        )

    def test_repr_does_not_recurse(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        rep = repr(case["actual"]["wf"])
        assert "HarmonicWaveform" in rep

    def test_xp_namespace_for_harmonic_waveform(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        xp = case["actual"]["wf"].__xp__()
        assert xp.__name__ == "jax.numpy"


class TestHarmonicProjectedWaveformJAX:
    def test_homogeneous_properties_and_kernel_shape(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        wf = case["actual"]["wf"]

        assert wf.channel_names == ("X", "Y", "Z")

        kernel = np.asarray(wf.get_kernel())
        assert kernel.shape == (1, 3, 2, 1, 3)

        expected = np.concatenate(
            [
                np.asarray(case["expected"]["resp_22"].get_kernel()),
                np.asarray(case["expected"]["resp_33"].get_kernel()),
            ],
            axis=2,
        )
        npt.assert_allclose(kernel, expected)

    def test_sum_harmonics_matches_manual_sum(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        wf = case["actual"]["wf"]

        summed = sum_harmonics(wf)
        got = np.asarray(summed.get_kernel())
        manual = np.asarray(wf.get_kernel()).sum(axis=2, keepdims=True)

        assert summed.channel_names == ("X", "Y", "Z")
        assert got.shape == (1, 3, 1, 1, 3)
        npt.assert_allclose(got, manual)

    def test_xp_namespace_for_harmonic_projected_waveform(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        xp = case["actual"]["wf"].__xp__()
        assert xp.__name__ == "jax.numpy"

    def test_pick_tuple_preserves_requested_order(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        wf = case["actual"]["wf"]

        picked = wf.pick((case["expected"]["mode_33"], case["expected"]["mode_22"]))

        assert tuple(picked.keys()) == (
            case["expected"]["mode_33"],
            case["expected"]["mode_22"],
        )

    def test_mode_channels_ops_scalar_and_unary(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        wf = case["actual"]["wf"]
        kernel = np.asarray(wf.get_kernel())

        shifted = wf + 2.0
        reflected = 2.0 + wf
        negated = -wf

        assert isinstance(shifted, type(wf))
        npt.assert_allclose(np.asarray(shifted.get_kernel()), kernel + 2.0)
        npt.assert_allclose(np.asarray(reflected.get_kernel()), 2.0 + kernel)  # pyright: ignore[reportOperatorIssue]
        npt.assert_allclose(np.asarray(negated.get_kernel()), -kernel)

    def test_mode_channels_ops_mapping_and_inplace(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        left = case["actual"]["wf"]
        right = build_harmonic_projected_waveform_constructor(jnp)
        kernel = np.asarray(left.get_kernel())

        combined = left + right
        npt.assert_allclose(np.asarray(combined.get_kernel()), kernel + kernel)

        left_copy = build_harmonic_projected_waveform_constructor(jnp)
        left_copy += right

        mode = case["expected"]["mode_22"]
        expected_x = 2.0 * np.asarray(case["expected"]["resp_22_map"]["X"].entries)
        expected_y = 2.0 * np.asarray(case["expected"]["resp_22_map"]["Y"].entries)
        npt.assert_allclose(np.asarray(left_copy[mode]["X"].entries), expected_x)
        npt.assert_allclose(np.asarray(left_copy[mode]["Y"].entries), expected_y)

    def test_mode_channels_ops_mode_mismatch_raises(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        left = case["actual"]["wf"]
        mismatched = build_harmonic_projected_waveform_from_mapping(
            {case["expected"]["mode_22"]: case["expected"]["resp_22"]},
        )

        with pytest.raises(ValueError, match=r".+"):
            _ = left + mismatched


class TestDenseMakerJAX:
    def test_dense_maker_embed_false_matches_densify_helper(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        wf, _ = build_harmonic_projected_phasor_waveform(frequencies=frequencies)

        maker = get_dense_maker(_linear_interpolator)
        fn = maker(frequencies, embed=False)
        out = fn(wf)

        assert type(out).__name__ == "HarmonicProjectedWaveform"
        assert tuple(out.keys()) == tuple(wf.keys())
        for harmonic in wf.harmonics:
            assert tuple(out[harmonic].keys()) == tuple(wf[harmonic].keys())
            for channel in wf[harmonic].channel_names:
                expected = densify_phasor(
                    wf[harmonic][channel],
                    _linear_interpolator,
                    frequencies,
                    embed=False,
                )
                npt.assert_allclose(
                    np.asarray(out[harmonic][channel].entries),
                    np.asarray(expected.entries),
                )

    def test_dense_maker_embed_true_matches_densify_helper(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        wf, _ = build_harmonic_projected_phasor_waveform(frequencies=frequencies)

        maker = get_dense_maker(_linear_interpolator)
        fn = maker(frequencies, embed=True)
        out = fn(wf)

        assert type(out).__name__ == "HarmonicProjectedWaveform"
        assert tuple(out.keys()) == tuple(wf.keys())
        for harmonic in wf.harmonics:
            assert tuple(out[harmonic].keys()) == tuple(wf[harmonic].keys())
            for channel in wf[harmonic].channel_names:
                expected = densify_phasor(
                    wf[harmonic][channel],
                    _linear_interpolator,
                    frequencies,
                    embed=True,
                )
                npt.assert_allclose(
                    np.asarray(out[harmonic][channel].entries),
                    np.asarray(expected.entries),
                )


class TestDensifyHelpersJAX:
    def test_densify_phasor_embed_false(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        phasor = build_test_phasor(
            f_min=1.0,
            f_max=3.0,
            frequencies=frequencies,
        )

        out = densify_phasor(phasor, _linear_interpolator, frequencies, embed=False)

        npt.assert_allclose(np.asarray(out.frequencies), np.array([1.0, 2.0, 3.0]))
        assert out.entries.shape[-1] == 3

    def test_densify_phasor_embed_true(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        phasor = build_test_phasor(
            f_min=1.0,
            f_max=3.0,
            frequencies=frequencies,
        )

        out = densify_phasor(phasor, _linear_interpolator, frequencies, embed=True)

        npt.assert_allclose(np.asarray(out.frequencies), np.asarray(frequencies))
        assert out.entries.shape[-1] == len(frequencies)

    def test_densify_phasor_pw_preserves_channels(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        fake_hpw, _ = build_harmonic_projected_phasor_waveform(
            frequencies=frequencies,
        )
        mode = fake_hpw.harmonics[0]
        wf = fake_hpw[mode]

        out = densify_phasor_pw(wf, _linear_interpolator, frequencies, embed=False)

        assert tuple(out.keys()) == tuple(wf.keys())
        for channel in wf.channel_names:
            expected = densify_phasor(
                wf[channel],
                _linear_interpolator,
                frequencies,
                embed=False,
            )
            npt.assert_allclose(
                np.asarray(out[channel].entries), np.asarray(expected.entries)
            )

    def test_densify_phasor_hw_preserves_harmonics(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        mode_22 = modes.Harmonic(2, 2)
        mode_33 = modes.Harmonic(3, 3)
        p22 = build_test_phasor(f_min=1.0, f_max=3.0, frequencies=frequencies)
        p33 = build_test_phasor(f_min=0.5, f_max=2.0, frequencies=frequencies)
        wf = build_harmonic_waveform_from_mapping({mode_22: p22, mode_33: p33})

        out = densify_phasor_hw(wf, _linear_interpolator, frequencies, embed=False)

        assert type(out).__name__ == "HomogeneousHarmonicWaveform"
        assert tuple(out.keys()) == (mode_22, mode_33)
        expected_22 = densify_phasor(
            p22, _linear_interpolator, frequencies, embed=False
        )
        expected_33 = densify_phasor(
            p33, _linear_interpolator, frequencies, embed=False
        )
        npt.assert_allclose(
            np.asarray(out[mode_22].entries), np.asarray(expected_22.entries)
        )
        npt.assert_allclose(
            np.asarray(out[mode_33].entries), np.asarray(expected_33.entries)
        )

    def test_densify_phasor_hpw_returns_homogeneous_container(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        wf, _ = build_harmonic_projected_phasor_waveform(frequencies=frequencies)

        out = densify_phasor_hpw(wf, _linear_interpolator, frequencies, embed=False)

        assert type(out).__name__ == "HomogeneousHarmonicProjectedWaveform"
        assert tuple(out.keys()) == tuple(wf.keys())
        assert out.channel_names == ("X", "Y", "Z")
        for mode in wf.harmonics:
            for channel in wf[mode].channel_names:
                expected = densify_phasor(
                    wf[mode][channel],
                    _linear_interpolator,
                    frequencies,
                    embed=False,
                )
                npt.assert_allclose(
                    np.asarray(out[mode][channel].entries),
                    np.asarray(expected.entries),
                )


class TestCombineHelpersJAX:
    def test_phasor_to_fs_hw_converts_each_mode(self):
        mode_22 = modes.Harmonic(2, 2)
        mode_33 = modes.Harmonic(3, 3)
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        p22 = build_test_phasor(
            f_min=1.0,
            f_max=3.0,
            frequencies=frequencies,
            amplitude_scale=1.0,
        )
        p33 = build_test_phasor(
            f_min=0.5,
            f_max=2.0,
            frequencies=frequencies,
            amplitude_scale=0.8,
        )
        wf = build_harmonic_waveform_from_mapping({mode_22: p22, mode_33: p33})

        out = phasor_to_fs_hw(wf)

        assert type(out).__name__ == "HarmonicWaveform"
        npt.assert_allclose(
            np.asarray(out[mode_22].entries),
            np.asarray(p22.to_frequency_series().entries),
        )
        npt.assert_allclose(
            np.asarray(out[mode_33].entries),
            np.asarray(p33.to_frequency_series().entries),
        )

    def test_phasor_to_fs_pw_converts_each_channel(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        wf = projected_waveform(
            {
                "X": build_test_phasor(
                    f_min=1.0,
                    f_max=3.0,
                    frequencies=frequencies,
                    amplitude_scale=1.0,
                ),
                "Y": build_test_phasor(
                    f_min=1.0,
                    f_max=3.0,
                    frequencies=frequencies,
                    amplitude_scale=0.9,
                ),
                "Z": build_test_phasor(
                    f_min=1.0,
                    f_max=3.0,
                    frequencies=frequencies,
                    amplitude_scale=1.1,
                ),
            },
        )

        out = phasor_to_fs_pw(wf)

        assert tuple(out.keys()) == ("X", "Y", "Z")
        for channel in wf.channel_names:
            npt.assert_allclose(
                np.asarray(out[channel].entries),
                np.asarray(wf[channel].to_frequency_series().entries),
            )

    def test_phasor_to_fs_hpw_converts_nested_leaves(self):
        frequencies = axis(jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64))
        wf, _ = build_harmonic_projected_phasor_waveform(frequencies=frequencies)
        out = phasor_to_fs_hpw(wf)

        assert type(out).__name__ == "HarmonicProjectedWaveform"
        assert tuple(out.keys()) == tuple(wf.keys())
        for mode in wf.harmonics:
            for channel in wf[mode].channel_names:
                npt.assert_allclose(
                    np.asarray(out[mode][channel].entries),
                    np.asarray(wf[mode][channel].to_frequency_series().entries),
                )


class TestWaveformConstructorsJAX:
    def test_constructor_aliases(self):
        assert hw is harmonic_waveform
        assert pw is projected_waveform
        assert hpw is harmonic_projected_waveform
        assert hhpw is homogeneous_harmonic_projected_waveform

    def test_harmonic_waveform_constructor(self):
        wf = build_harmonic_waveform_constructor(jnp)

        assert type(wf).__name__ == "HarmonicWaveform"
        assert tuple(wf.keys()) == (modes.Harmonic(2, 2), modes.Harmonic(3, 3))

    def test_projected_waveform_constructor(self):
        resp = build_projected_waveform_from_mapping(
            build_harmonic_projected_frequency_waveform(jnp)["expected"]["resp_22_map"],
        )

        assert type(resp).__name__ == "ProjectedWaveform"
        assert resp.channel_names == ("X", "Y", "Z")

    def test_harmonic_projected_waveform_constructor(self):
        wf = build_harmonic_projected_waveform_constructor(jnp)

        assert type(wf).__name__ == "HarmonicProjectedWaveform"
        assert tuple(wf.keys()) == (modes.Harmonic(2, 2), modes.Harmonic(3, 3))

    def test_homogeneous_harmonic_projected_waveform_constructor(self):
        wf = build_homogeneous_harmonic_projected_waveform_constructor(jnp)

        assert type(wf).__name__ == "HomogeneousHarmonicProjectedWaveform"
        assert wf.channel_names == ("X", "Y", "Z")
