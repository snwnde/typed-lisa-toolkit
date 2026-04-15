"""Tests for likelihood computations with NumPy arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest

import numpy as np
import numpy.testing as npt

from tests._helpers import (
    build_fd_pair,
    build_harmonic_projected_frequency_waveform,
    dense_kernel_2ch,
)
from typed_lisa_toolkit import (
    frequency_series,
    fsdata,
    linspace_from_array,
    make_sdm,
    noise_model,
    sum_harmonics,
    whittle,
)
from typed_lisa_toolkit.types import (
    FDWhittleLikelihood,
)


def _build_fsdata(freqs, x_values, y_values):
    return fsdata(
        {
            "X": frequency_series(freqs, x_values[None, None, None, None, :]),
            "Y": frequency_series(freqs, y_values[None, None, None, None, :]),
        }
    )


class TestFDWhittleLikelihoodNumpy(unittest.TestCase):
    def test_classmethod_formulas(self):
        self.assertEqual(FDWhittleLikelihood.log_likelihood_ratio(5.0, 2.0), 4.0)
        self.assertEqual(FDWhittleLikelihood.log_likelihood(4.0, 2.0), 3.0)

    def test_cross_product_and_template_square_match_noise_model(self):
        case = build_fd_pair(np)
        sdm = make_sdm(
            dense_kernel_2ch(np),
            frequencies=case["frequencies"],
            channel_names=("X", "Y"),
        )
        model = noise_model(sdm)
        likelihood = whittle(case["left"], model)

        cross = np.asarray(likelihood.get_cross_product(case["right"]))
        template_square = np.asarray(likelihood.get_template_square(case["right"]))

        npt.assert_allclose(
            cross,
            np.asarray(model.reset().get_scalar_product(case["left"], case["right"])),
        )
        npt.assert_allclose(
            template_square,
            np.asarray(model.reset().get_scalar_product(case["right"], case["right"])),
        )

    def test_log_likelihood_matches_closed_form(self):
        case = build_fd_pair(np)
        sdm = make_sdm(
            dense_kernel_2ch(np),
            frequencies=case["frequencies"],
            channel_names=("X", "Y"),
        )
        model = noise_model(sdm)
        likelihood = whittle(case["left"], model)

        got = np.asarray(likelihood.get_log_likelihood(case["right"]))
        expected = (
            np.asarray(model.reset().get_scalar_product(case["left"], case["right"]))
            - 0.5
            * np.asarray(model.reset().get_scalar_product(case["right"], case["right"]))
            - 0.5
            * np.asarray(model.reset().get_scalar_product(case["left"], case["left"]))
        )

        npt.assert_allclose(got, expected)

    def test_harmonic_projected_template_is_summed_before_evaluation(self):
        case = build_harmonic_projected_frequency_waveform(np)
        data = fsdata(sum_harmonics(case["wf"]))
        sdm = make_sdm(
            dense_kernel_2ch(np),
            frequencies=case["frequencies"],
            channel_names=("X", "Y"),
        )
        model = noise_model(sdm)
        likelihood = whittle(data, model)

        got = np.asarray(likelihood.get_cross_product(case["wf"]))
        expected = np.asarray(
            model.reset().get_scalar_product(
                data, fsdata(sum_harmonics(case["wf"]))
            )
        )

        npt.assert_allclose(got, expected)

    def test_template_is_restricted_to_its_frequency_band(self):
        freqs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        left = _build_fsdata(
            linspace_from_array(freqs),
            np.array([1.0 + 0.0j, 0.5 + 0.1j, 2.0 - 0.2j, 1.0 + 0.5j, 0.1 + 0.0j]),
            np.array([0.25 + 0.0j, -0.5 + 0.25j, 1.0 + 0.0j, 0.5 - 0.25j, -0.1 + 0.0j]),
        )
        template = left.get_subset(interval=(1.0, 3.0))
        kernel = np.broadcast_to(np.eye(2), (len(freqs), 2, 2)).copy()
        model = noise_model(make_sdm(kernel, frequencies=freqs, channel_names=("X", "Y")))
        likelihood = whittle(left, model)

        got = np.asarray(likelihood.get_cross_product(template))
        expected = np.asarray(
            noise_model(
                make_sdm(
                    kernel[1:4],
                    frequencies=np.asarray(template.frequencies),
                    channel_names=("X", "Y"),
                )
            ).get_scalar_product(left.get_subset(interval=(1.0, 3.0)), template)
        )

        npt.assert_allclose(got, expected)

    def test_whittle_factory_returns_fd_whittle_likelihood(self):
        case = build_fd_pair(np)
        model = noise_model(
            make_sdm(
                dense_kernel_2ch(np),
                frequencies=case["frequencies"],
                channel_names=("X", "Y"),
            )
        )

        likelihood = whittle(case["left"], model)

        self.assertIsInstance(likelihood, FDWhittleLikelihood)
