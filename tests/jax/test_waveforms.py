"""Tests for waveform containers with JAX arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit.containers import modes
from typed_lisa_toolkit.containers.waveforms import get_dense_maker, sum_harmonics
from tests._shared.waveforms_helpers import (
    build_harmonic_projected_frequency_waveform,
    build_harmonic_waveform_frequency_series,
    build_nonhomogeneous_harmonic_projected_frequency_waveform,
)
from tests._shared.waveforms_helpers import (
    FakeHarmonicWaveform,
    build_fake_harmonic_projected_waveform,
)


class TestHarmonicWaveformJAX(unittest.TestCase):
    def test_domain_and_pick(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        wf = case["wf"]

        self.assertEqual(wf.domain, "frequency")

        picked = wf.pick(case["mode_33"])
        self.assertEqual(tuple(picked.keys()), (case["mode_33"],))
        npt.assert_allclose(
            np.asarray(picked[case["mode_33"]].entries),
            np.asarray(case["wf_33"].entries),
        )

    def test_pick_missing_mode_raises(self):
        case = build_harmonic_waveform_frequency_series(jnp)

        with self.assertRaises(KeyError):
            case["wf"].pick(modes.Harmonic(9, 9))

    def test_pick_tuple_preserves_requested_order(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        wf = case["wf"]

        picked = wf.pick((case["mode_33"], case["mode_22"]))

        self.assertEqual(tuple(picked.keys()), (case["mode_33"], case["mode_22"]))

    def test_repr_does_not_recurse(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        rep = repr(case["wf"])
        self.assertIn("HarmonicWaveform", rep)

    def test_xp_namespace_for_harmonic_waveform(self):
        case = build_harmonic_waveform_frequency_series(jnp)
        xp = case["wf"].__xp__()
        self.assertEqual(xp.__name__, "jax.numpy")


class TestHarmonicProjectedWaveformJAX(unittest.TestCase):
    def test_homogeneous_properties_and_kernel_shape(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        wf = case["wf"]

        self.assertEqual(wf.domain, "frequency")
        self.assertEqual(wf.channel_names, ("X", "Y"))
        self.assertTrue(wf.is_homogeneous)

        kernel = np.asarray(wf.get_kernel())
        self.assertEqual(kernel.shape, (1, 2, 2, 1, 1, 3))

        expected = np.stack(
            [
                np.asarray(case["resp_22"].entries),
                np.asarray(case["resp_33"].entries),
            ],
            axis=2,
        )
        npt.assert_allclose(kernel, expected)

    def test_get_kernel_raises_for_nonhomogeneous_waveform(self):
        case = build_nonhomogeneous_harmonic_projected_frequency_waveform(jnp)

        self.assertFalse(case["wf"].is_homogeneous)
        with self.assertRaises(ValueError):
            case["wf"].get_kernel()

    def test_sum_harmonics_matches_manual_sum(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        wf = case["wf"]

        summed = sum_harmonics(wf)
        got = np.asarray(summed.get_kernel())
        manual = np.asarray(wf.get_kernel()).sum(axis=2)

        self.assertEqual(summed.channel_names, ("X", "Y"))
        self.assertEqual(got.shape, (1, 2, 1, 1, 3))
        npt.assert_allclose(got, manual)

    def test_xp_namespace_for_harmonic_projected_waveform(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        xp = case["wf"].__xp__()
        self.assertEqual(xp.__name__, "jax.numpy")

    def test_pick_tuple_preserves_requested_order(self):
        case = build_harmonic_projected_frequency_waveform(jnp)
        wf = case["wf"]

        picked = wf.pick((case["mode_33"], case["mode_22"]))

        self.assertEqual(tuple(picked.keys()), (case["mode_33"], case["mode_22"]))


class TestDenseMakerJAX(unittest.TestCase):
    @staticmethod
    def _identity_mapping(d: dict[str, Any]) -> dict[str, Any]:
        # Replaces ProjectedWaveform.from_dict so the test never touches the real
        # container constructor; the dict of channel→mock passes straight through.
        return d

    @staticmethod
    def _windowed(frequencies, phasor):
        # JAX arrays are converted to NumPy for searchsorted; mirrors the windowing
        # inside get_dense_maker so we can check the exact slice sent to get_interpolated.
        freqs = np.asarray(frequencies)
        return freqs[
            np.searchsorted(freqs, phasor.f_min, side="left") : np.searchsorted(
                freqs, phasor.f_max, side="right"
            )
        ]

    def test_dense_maker_embed_false_calls_interpolated_only(self):
        # Full frequency grid (JAX array) passed to `make`; each phasor covers a sub-range.
        frequencies = jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        interpolator = MagicMock(name="interpolator")
        wf, handles = build_fake_harmonic_projected_waveform()

        # Build the two-level closure: get_dense_maker binds the interpolator,
        # maker(frequencies) binds the target grid and embed flag.
        maker = get_dense_maker(interpolator)
        fn = maker(frequencies, embed=False)

        # Patch ProjectedWaveform.from_dict to an identity so structural assertions
        # can be made without constructing real ProjectedWaveform objects.
        with patch(
            "typed_lisa_toolkit.containers.waveforms.ProjectedWaveform.from_dict",
            side_effect=self._identity_mapping,
        ) as from_dict_mock:
            out = fn(wf)

        # from_dict must be called once per harmonic (2 modes → 2 calls).
        self.assertIsInstance(out, FakeHarmonicWaveform)
        self.assertEqual(from_dict_mock.call_count, 2)
        # Output preserves harmonic and channel key order.
        self.assertEqual(tuple(out.keys()), tuple(wf.keys()))

        for harmonic in wf.harmonics:
            self.assertEqual(tuple(out[harmonic].keys()), tuple(wf[harmonic].keys()))
            for channel in wf[harmonic].channel_names:
                phasor, interpolated, _ = handles[harmonic][channel]
                # get_interpolated must be called exactly once with the per-phasor
                # windowed slice (not the full grid) and the bound interpolator.
                phasor.get_interpolated.assert_called_once()
                args, kwargs = phasor.get_interpolated.call_args
                self.assertEqual(kwargs, {})
                npt.assert_allclose(args[0], self._windowed(frequencies, phasor))
                self.assertIs(args[1], interpolator)
                # embed=False: output is the interpolated mock, not the embedded one.
                self.assertIs(out[harmonic][channel], interpolated)
                interpolated.get_embedded.assert_not_called()

    def test_dense_maker_embed_true_calls_embedded(self):
        frequencies = jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        interpolator = MagicMock(name="interpolator")
        wf, handles = build_fake_harmonic_projected_waveform()

        maker = get_dense_maker(interpolator)
        fn = maker(frequencies, embed=True)

        with patch(
            "typed_lisa_toolkit.containers.waveforms.ProjectedWaveform.from_dict",
            side_effect=self._identity_mapping,
        ) as from_dict_mock:
            out = fn(wf)

        self.assertIsInstance(out, FakeHarmonicWaveform)
        self.assertEqual(from_dict_mock.call_count, 2)
        self.assertEqual(tuple(out.keys()), tuple(wf.keys()))

        for harmonic in wf.harmonics:
            self.assertEqual(tuple(out[harmonic].keys()), tuple(wf[harmonic].keys()))
            for channel in wf[harmonic].channel_names:
                phasor, interpolated, embedded = handles[harmonic][channel]
                # Still windowed before interpolation, same as embed=False.
                phasor.get_interpolated.assert_called_once()
                args, kwargs = phasor.get_interpolated.call_args
                self.assertEqual(kwargs, {})
                npt.assert_allclose(args[0], self._windowed(frequencies, phasor))
                self.assertIs(args[1], interpolator)

                # embed=True: get_embedded is called on the interpolated result
                # with the *full* frequency grid to zero-pad outside the phasor range.
                interpolated.get_embedded.assert_called_once()
                args, kwargs = interpolated.get_embedded.call_args
                self.assertEqual(kwargs, {})
                self.assertIs(args[0], frequencies)
                # Output is the embedded mock, not the raw interpolated one.
                self.assertIs(out[harmonic][channel], embedded)
