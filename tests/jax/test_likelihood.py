"""Tests for likelihood computations with JAX arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from tests._helpers import (
    build_fd_pair,
    build_harmonic_projected_frequency_waveform,
    dense_kernel_2ch,
)
from typed_lisa_toolkit import fsdata, sum_harmonics
from typed_lisa_toolkit.types import (
    FDNoiseModel,
    FDWhittleLikelihood,
    FrequencySeries,
    FSData,
    SpectralDensity,
)


def _build_fsdata(freqs, x_values, y_values):
    return FSData.from_dict(
        {
            "X": FrequencySeries((freqs,), x_values[None, None, None, None, :]),
            "Y": FrequencySeries((freqs,), y_values[None, None, None, None, :]),
        }
    )


class TestFDWhittleLikelihoodJAX(unittest.TestCase):
    def test_classmethod_formulas(self):
        self.assertEqual(FDWhittleLikelihood.log_likelihood_ratio(5.0, 2.0), 4.0)
        self.assertEqual(FDWhittleLikelihood.log_likelihood(4.0, 2.0), 3.0)

    def test_cross_product_and_template_square_match_noise_model(self):
        case = build_fd_pair(jnp)
        model = FDNoiseModel(
            SpectralDensity(case["frequencies"], dense_kernel_2ch(jnp), ["X", "Y"])
        )
        likelihood = FDWhittleLikelihood(case["left"], model)

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
        case = build_fd_pair(jnp)
        model = FDNoiseModel(
            SpectralDensity(case["frequencies"], dense_kernel_2ch(jnp), ["X", "Y"])
        )
        likelihood = FDWhittleLikelihood(case["left"], model)

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
        case = build_harmonic_projected_frequency_waveform(jnp)
        data = fsdata(sum_harmonics(case["wf"]))
        model = FDNoiseModel(
            SpectralDensity(case["frequencies"], dense_kernel_2ch(jnp), ["X", "Y"])
        )
        likelihood = FDWhittleLikelihood(data, model)

        got = np.asarray(likelihood.get_cross_product(case["wf"]))
        expected = np.asarray(
            model.reset().get_scalar_product(
                data,
                fsdata(sum_harmonics(case["wf"])),
            )
        )

        npt.assert_allclose(got, expected)

    def test_template_is_restricted_to_its_frequency_band(self):
        freqs = jnp.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        left = _build_fsdata(
            freqs,
            jnp.array(
                [1.0 + 0.0j, 0.5 + 0.1j, 2.0 - 0.2j, 1.0 + 0.5j, 0.1 + 0.0j],
                dtype=jnp.complex128,
            ),
            jnp.array(
                [0.25 + 0.0j, -0.5 + 0.25j, 1.0 + 0.0j, 0.5 - 0.25j, -0.1 + 0.0j],
                dtype=jnp.complex128,
            ),
        )
        template = left.get_subset(interval=(1.0, 3.0))
        kernel = jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (len(freqs), 2, 2))
        model = FDNoiseModel(SpectralDensity(freqs, kernel, ["X", "Y"]))
        likelihood = FDWhittleLikelihood(left, model)

        got = np.asarray(likelihood.get_cross_product(template))
        expected = np.asarray(
            FDNoiseModel(
                SpectralDensity(template.frequencies, kernel[1:4], ["X", "Y"])
            ).get_scalar_product(left.get_subset(interval=(1.0, 3.0)), template)
        )

        npt.assert_allclose(got, expected)
