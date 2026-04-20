"""Tests for waveform containers with NumPy arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pytest

from tests._helpers import (
    build_harmonic_projected_frequency_waveform,
    build_harmonic_projected_phasor_waveform,
    build_harmonic_waveform_frequency_series,
    make_mock_phasor,
    make_valid_mock_representation,
)
from typed_lisa_toolkit import (
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


class TestHarmonicWaveformNumpy(unittest.TestCase):
    def test_domain_and_pick(self):
        case = build_harmonic_waveform_frequency_series(np)
        wf = case["wf"]

        assert wf.domain == "frequency"

        picked = wf.pick(case["mode_22"])
        assert tuple(picked.keys()) == (case["mode_22"],)
        npt.assert_allclose(
            np.asarray(picked[case["mode_22"]].entries),
            np.asarray(case["wf_22"].entries),
        )

    def test_pick_missing_mode_raises(self):
        case = build_harmonic_waveform_frequency_series(np)
        wf = case["wf"]

        with pytest.raises(KeyError):
            wf.pick(modes.Harmonic(9, 9))

    def test_pick_tuple_preserves_requested_order(self):
        case = build_harmonic_waveform_frequency_series(np)
        wf = case["wf"]

        picked = wf.pick((case["mode_33"], case["mode_22"]))

        assert tuple(picked.keys()) == (case["mode_33"], case["mode_22"])

    def test_repr_does_not_recurse(self):
        case = build_harmonic_waveform_frequency_series(np)
        rep = repr(case["wf"])
        assert "HarmonicWaveform" in rep

    def test_xp_namespace_for_harmonic_waveform(self):
        case = build_harmonic_waveform_frequency_series(np)
        xp = case["wf"].__xp__()
        assert xp.__name__ == "array_api_compat.numpy"


class TestHarmonicProjectedWaveformNumpy(unittest.TestCase):
    def test_homogeneous_properties_and_kernel_shape(self):
        case = build_harmonic_projected_frequency_waveform(np)
        wf = case["wf"]

        assert wf.channel_names == ("X", "Y")

        kernel = np.asarray(wf.get_kernel())
        assert kernel.shape == (1, 2, 2, 1, 3)

        expected = np.concatenate(
            [
                np.asarray(case["resp_22"].get_kernel()),
                np.asarray(case["resp_33"].get_kernel()),
            ],
            axis=2,
        )
        npt.assert_allclose(kernel, expected)

    def test_sum_harmonics_matches_manual_sum(self):
        case = build_harmonic_projected_frequency_waveform(np)
        wf = case["wf"]

        summed = sum_harmonics(wf)
        got = np.asarray(summed.get_kernel())

        manual = np.asarray(wf.get_kernel()).sum(axis=2, keepdims=True)

        assert summed.channel_names == ("X", "Y")
        assert got.shape == (1, 2, 1, 1, 3)
        npt.assert_allclose(got, manual)

    def test_xp_namespace_for_harmonic_projected_waveform(self):
        case = build_harmonic_projected_frequency_waveform(np)
        xp = case["wf"].__xp__()
        assert xp.__name__ == "array_api_compat.numpy"

    def test_pick_tuple_preserves_requested_order(self):
        case = build_harmonic_projected_frequency_waveform(np)
        wf = case["wf"]

        picked = wf.pick((case["mode_33"], case["mode_22"]))

        assert tuple(picked.keys()) == (case["mode_33"], case["mode_22"])

    def test_mode_channels_ops_scalar_and_unary(self):
        case = build_harmonic_projected_frequency_waveform(np)
        wf = case["wf"]
        kernel = np.asarray(wf.get_kernel())

        shifted = wf + 2.0
        reflected = 2.0 + wf
        negated = -wf

        assert isinstance(shifted, type(wf))
        npt.assert_allclose(np.asarray(shifted.get_kernel()), kernel + 2.0)
        npt.assert_allclose(np.asarray(reflected.get_kernel()), 2.0 + kernel)
        npt.assert_allclose(np.asarray(negated.get_kernel()), -kernel)

    def test_mode_channels_ops_mapping_and_inplace(self):
        case = build_harmonic_projected_frequency_waveform(np)
        left = case["wf"]
        right = build_harmonic_projected_frequency_waveform(np)["wf"]
        kernel = np.asarray(left.get_kernel())

        combined = left + right
        npt.assert_allclose(np.asarray(combined.get_kernel()), kernel + kernel)

        left_copy = build_harmonic_projected_frequency_waveform(np)["wf"]
        left_copy += right
        npt.assert_allclose(np.asarray(left_copy.get_kernel()), kernel + kernel)

        mode = case["mode_22"]
        expected_x = 2.0 * np.asarray(case["resp_22_map"]["X"].entries)
        npt.assert_allclose(np.asarray(left_copy[mode]["X"].entries), expected_x)

    def test_mode_channels_ops_mode_mismatch_raises(self):
        case = build_harmonic_projected_frequency_waveform(np)
        left = case["wf"]
        mismatched = harmonic_projected_waveform({case["mode_22"]: case["resp_22"]})

        with pytest.raises(ValueError, match=r".+"):
            _ = left + mismatched


class TestDenseMakerNumpy(unittest.TestCase):
    @staticmethod
    def _identity_mapping(d: dict[str, Any]) -> dict[str, Any]:
        # Replaces ProjectedWaveform.from_dict so the test never touches the real
        # container constructor; the dict of channel→mock passes straight through.
        return d

    @staticmethod
    def _windowed(frequencies, phasor):
        # Mirror the frequency-windowing logic in get_dense_maker so each test
        # can compute the exact slice it expects to see passed to get_interpolated.
        return np.asarray(frequencies)[
            np.searchsorted(
                np.asarray(frequencies),
                phasor.f_min,
                side="left",
            ) : np.searchsorted(np.asarray(frequencies), phasor.f_max, side="right")
        ]

    @staticmethod
    def _bind_entries(handles, frequencies):
        for channels in handles.values():
            for phasor, _, _ in channels.values():
                phasor.entries = frequencies

    def test_dense_maker_embed_false_calls_interpolated_only(self):
        # Full frequency grid passed to `make`; each phasor covers only a sub-range.
        frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        interpolator = MagicMock(name="interpolator")
        wf, handles = build_harmonic_projected_phasor_waveform()
        self._bind_entries(handles, frequencies)

        # Build the two-level closure: get_dense_maker binds the interpolator,
        # maker(frequencies) binds the target grid and embed flag.
        maker = get_dense_maker(interpolator)
        fn = maker(frequencies, embed=False)

        # Patch ProjectedWaveform.from_dict to an identity so structural assertions
        # can be made without constructing real ProjectedWaveform objects.
        with patch(
            "typed_lisa_toolkit.types.ProjectedWaveform.from_dict",
            side_effect=self._identity_mapping,
        ) as from_dict_mock:
            out = fn(wf)

        # from_dict must be called once per harmonic (2 modes → 2 calls).
        assert type(out).__name__ == "HarmonicProjectedWaveform"
        assert from_dict_mock.call_count == 2
        # Output preserves harmonic and channel key order.
        assert tuple(out.keys()) == tuple(wf.keys())

        for harmonic in wf.harmonics:
            assert tuple(out[harmonic].keys()) == tuple(wf[harmonic].keys())
            for channel in wf[harmonic].channel_names:
                phasor, interpolated, _ = handles[harmonic][channel]
                # get_interpolated must be called exactly once with the per-phasor
                # windowed slice (not the full grid) and the bound interpolator.
                phasor.get_interpolated.assert_called_once()
                args, kwargs = phasor.get_interpolated.call_args
                assert kwargs == {}
                npt.assert_allclose(args[0], self._windowed(frequencies, phasor))
                assert args[1] is interpolator
                # embed=False: output is the interpolated mock, not the embedded one.
                assert out[harmonic][channel] is interpolated
                interpolated.get_embedded.assert_not_called()

    def test_dense_maker_embed_true_calls_embedded(self):
        frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        interpolator = MagicMock(name="interpolator")
        wf, handles = build_harmonic_projected_phasor_waveform()
        self._bind_entries(handles, frequencies)

        maker = get_dense_maker(interpolator)
        fn = maker(frequencies, embed=True)

        with patch(
            "typed_lisa_toolkit.types.ProjectedWaveform.from_dict",
            side_effect=self._identity_mapping,
        ) as from_dict_mock:
            out = fn(wf)

        assert type(out).__name__ == "HarmonicProjectedWaveform"
        assert from_dict_mock.call_count == 2
        assert tuple(out.keys()) == tuple(wf.keys())

        for harmonic in wf.harmonics:
            assert tuple(out[harmonic].keys()) == tuple(wf[harmonic].keys())
            for channel in wf[harmonic].channel_names:
                phasor, interpolated, embedded = handles[harmonic][channel]
                # Still windowed before interpolation, same as embed=False.
                phasor.get_interpolated.assert_called_once()
                args, _ = phasor.get_interpolated.call_args
                npt.assert_allclose(args[0], self._windowed(frequencies, phasor))
                assert args[1] is interpolator

                # embed=True: get_embedded is called on the interpolated result
                # with the *full* frequency grid to zero-pad outside the phasor range.
                interpolated.get_embedded.assert_called_once()
                args, kwargs = interpolated.get_embedded.call_args
                assert isinstance(args[0], tuple)
                assert len(args[0]) == 1
                assert args[0][0] is frequencies
                assert "known_slices" in kwargs
                assert len(kwargs["known_slices"]) == 1
                # Output is the embedded mock, not the raw interpolated one.
                assert out[harmonic][channel] is embedded


class TestDensifyHelpersNumpy(unittest.TestCase):
    def test_densify_phasor_embed_false(self):
        frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        interpolator = MagicMock(name="interpolator")
        phasor, interpolated, _ = make_mock_phasor(
            f_min=1.0,
            f_max=3.0,
            frequencies=frequencies,
        )

        out = densify_phasor(phasor, interpolator, frequencies, embed=False)

        phasor.get_interpolated.assert_called_once()
        args, kwargs = phasor.get_interpolated.call_args
        assert kwargs == {}
        npt.assert_allclose(args[0], np.array([1.0, 2.0, 3.0]))
        assert args[1] is interpolator
        interpolated.get_embedded.assert_not_called()
        assert out is interpolated

    def test_densify_phasor_embed_true(self):
        frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        interpolator = MagicMock(name="interpolator")
        phasor, interpolated, embedded = make_mock_phasor(
            f_min=1.0,
            f_max=3.0,
            frequencies=frequencies,
        )

        out = densify_phasor(phasor, interpolator, frequencies, embed=True)

        interpolated.get_embedded.assert_called_once()
        args, kwargs = interpolated.get_embedded.call_args
        assert args[0][0] is frequencies
        assert "known_slices" in kwargs
        assert len(kwargs["known_slices"]) == 1
        assert out is embedded

    def test_densify_phasor_pw_preserves_channels(self):
        frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        interpolator = MagicMock(name="interpolator")
        fake_hpw, handles = build_harmonic_projected_phasor_waveform(
            frequencies=frequencies,
        )
        mode = fake_hpw.harmonics[0]
        wf = fake_hpw[mode]

        with patch(
            "typed_lisa_toolkit.types.ProjectedWaveform.from_dict",
            side_effect=lambda d: d,
        ):
            out = densify_phasor_pw(wf, interpolator, frequencies, embed=False)

        assert tuple(out.keys()) == tuple(wf.keys())
        for channel in wf.channel_names:
            _, interpolated, _ = handles[mode][channel]
            assert out[channel] is interpolated

    def test_densify_phasor_hw_preserves_harmonics(self):
        frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        interpolator = MagicMock(name="interpolator")
        mode_22 = modes.Harmonic(2, 2)
        mode_33 = modes.Harmonic(3, 3)
        p22, i22, _ = make_mock_phasor(f_min=1.0, f_max=3.0, frequencies=frequencies)
        p33, i33, _ = make_mock_phasor(f_min=0.5, f_max=2.0, frequencies=frequencies)
        wf = harmonic_waveform({mode_22: p22, mode_33: p33})

        out = densify_phasor_hw(wf, interpolator, frequencies, embed=False)

        assert type(out).__name__ == "HomogeneousHarmonicWaveform"
        assert tuple(out.keys()) == (mode_22, mode_33)
        assert out[mode_22] is i22
        assert out[mode_33] is i33

    def test_densify_phasor_hpw_returns_homogeneous_container(self):
        frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        interpolator = MagicMock(name="interpolator")
        wf, handles = build_harmonic_projected_phasor_waveform(frequencies=frequencies)

        with patch(
            "typed_lisa_toolkit.types.ProjectedWaveform.from_dict",
            side_effect=lambda d: d,
        ):
            out = densify_phasor_hpw(wf, interpolator, frequencies, embed=False)

        assert type(out).__name__ == "HomogeneousHarmonicProjectedWaveform"
        assert tuple(out.keys()) == tuple(wf.keys())
        assert out.channel_names == ("X", "Y")
        for mode in wf.harmonics:
            for channel in wf[mode].channel_names:
                _, interpolated, _ = handles[mode][channel]
                assert out[mode][channel] is interpolated


class TestCombineHelpersNumpy(unittest.TestCase):
    def test_phasor_to_fs_hw_converts_each_mode(self):
        mode_22 = modes.Harmonic(2, 2)
        mode_33 = modes.Harmonic(3, 3)
        p22 = make_valid_mock_representation(name="p22")
        p33 = make_valid_mock_representation(name="p33")
        fs22 = make_valid_mock_representation(name="fs22")
        fs33 = make_valid_mock_representation(name="fs33")
        p22.to_frequency_series.return_value = fs22
        p33.to_frequency_series.return_value = fs33
        wf = harmonic_waveform({mode_22: p22, mode_33: p33})

        out = phasor_to_fs_hw(wf)

        assert type(out).__name__ == "HarmonicWaveform"
        assert out[mode_22] is fs22
        assert out[mode_33] is fs33

    def test_phasor_to_fs_pw_converts_each_channel(self):
        p_x = make_valid_mock_representation(name="p_x")
        p_y = make_valid_mock_representation(name="p_y")
        fs_x = make_valid_mock_representation(name="fs_x")
        fs_y = make_valid_mock_representation(name="fs_y")
        p_x.to_frequency_series.return_value = fs_x
        p_y.to_frequency_series.return_value = fs_y
        fake_hpw, _ = build_harmonic_projected_phasor_waveform()
        wf = fake_hpw[fake_hpw.harmonics[0]]
        wf["X"] = p_x
        wf["Y"] = p_y

        with patch(
            "typed_lisa_toolkit.types.ProjectedWaveform.from_dict",
            side_effect=lambda d: d,
        ):
            out = phasor_to_fs_pw(wf)

        assert tuple(out.keys()) == ("X", "Y")
        assert out["X"] is fs_x
        assert out["Y"] is fs_y

    def test_phasor_to_fs_hpw_converts_nested_leaves(self):
        wf, _ = build_harmonic_projected_phasor_waveform()
        expected = {}
        for mode in wf.harmonics:
            expected[mode] = {}
            for channel in wf[mode].channel_names:
                p = wf[mode][channel]
                fs = make_valid_mock_representation(name=f"fs_{mode}_{channel}")
                p.to_frequency_series.return_value = fs
                expected[mode][channel] = fs

        with patch(
            "typed_lisa_toolkit.types.ProjectedWaveform.from_dict",
            side_effect=lambda d: d,
        ):
            out = phasor_to_fs_hpw(wf)

        assert type(out).__name__ == "HarmonicProjectedWaveform"
        assert tuple(out.keys()) == tuple(wf.keys())
        for mode in wf.harmonics:
            for channel in wf[mode].channel_names:
                assert out[mode][channel] is expected[mode][channel]


class TestWaveformConstructorsNumpy(unittest.TestCase):
    def test_constructor_aliases(self):
        assert hw is harmonic_waveform
        assert pw is projected_waveform
        assert hpw is harmonic_projected_waveform
        assert hhpw is homogeneous_harmonic_projected_waveform

    def test_harmonic_waveform_constructor(self):
        case = build_harmonic_waveform_frequency_series(np)
        wf = harmonic_waveform(
            {case["mode_22"]: case["wf_22"], case["mode_33"]: case["wf_33"]},
        )

        assert type(wf).__name__ == "HarmonicWaveform"
        assert tuple(wf.keys()) == (case["mode_22"], case["mode_33"])

    def test_projected_waveform_constructor(self):
        case = build_harmonic_projected_frequency_waveform(np)
        resp = projected_waveform(case["resp_22_map"])

        assert type(resp).__name__ == "ProjectedWaveform"
        assert resp.channel_names == ("X", "Y")

    def test_harmonic_projected_waveform_constructor(self):
        case = build_harmonic_projected_frequency_waveform(np)
        wf = harmonic_projected_waveform(
            {
                case["mode_22"]: case["resp_22_map"],
                case["mode_33"]: case["resp_33_map"],
            },
        )

        assert type(wf).__name__ == "HarmonicProjectedWaveform"
        assert tuple(wf.keys()) == (case["mode_22"], case["mode_33"])

    def test_homogeneous_harmonic_projected_waveform_constructor(self):
        case = build_harmonic_projected_frequency_waveform(np)
        wf = homogeneous_harmonic_projected_waveform(
            {
                case["mode_22"]: case["resp_22_map"],
                case["mode_33"]: case["resp_33_map"],
            },
        )

        assert type(wf).__name__ == "HomogeneousHarmonicProjectedWaveform"
        assert wf.channel_names == ("X", "Y")
