"""Tests for likelihood computations with JAX arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import (
    fsdata,
    make_sdm,
    noise_model,
    sum_harmonics,
    whittle,
)
from typed_lisa_toolkit.types import (
    FDWhittleLikelihood,
)

if TYPE_CHECKING:
    from conftest import (
        build_fd_pair,
        build_fd_template_band_case,
        build_harmonic_projected_frequency_waveform,
        dense_kernel_3ch,
    )

jax.config.update("jax_enable_x64", val=True)


class TestFDWhittleLikelihoodJAX:
    def test_classmethod_formulas(self):
        assert FDWhittleLikelihood.log_likelihood_ratio(5.0, 2.0) == 4.0  # pyright: ignore[reportArgumentType]
        assert FDWhittleLikelihood.log_likelihood(4.0, 2.0) == 3.0  # pyright: ignore[reportArgumentType]

    def test_cross_product_and_template_square_match_noise_model(
        self,
    ):
        case = build_fd_pair(jnp)
        left = case["actual"]["left"]
        right = case["actual"]["right"]
        frequencies = case["expected"]["frequencies"]
        sdm = make_sdm(
            dense_kernel_3ch(jnp),
            frequencies=frequencies.asarray(jnp),
            channel_names=("X", "Y", "Z"),
        )
        model = noise_model(sdm)
        likelihood = whittle(left, model)

        cross = np.asarray(likelihood.get_cross_product(right))
        template_square = np.asarray(likelihood.get_template_square(right))

        npt.assert_allclose(
            cross,
            np.asarray(model.reset().get_scalar_product(left, right)),
        )
        npt.assert_allclose(
            template_square,
            np.asarray(model.reset().get_scalar_product(right, right)),
        )

    def test_log_likelihood_matches_closed_form(self):
        case = build_fd_pair(jnp)
        left = case["actual"]["left"]
        right = case["actual"]["right"]
        frequencies = case["expected"]["frequencies"]
        sdm = make_sdm(
            dense_kernel_3ch(jnp),
            frequencies=frequencies.asarray(jnp),
            channel_names=("X", "Y", "Z"),
        )
        model = noise_model(sdm)
        likelihood = whittle(left, model)

        got = np.asarray(likelihood.get_log_likelihood(right))
        expected = (
            np.asarray(model.reset().get_scalar_product(left, right))
            - 0.5 * np.asarray(model.reset().get_scalar_product(right, right))
            - 0.5 * np.asarray(model.reset().get_scalar_product(left, left))
        )

        npt.assert_allclose(got, expected)

    def test_harmonic_projected_template_is_summed_before_evaluation(
        self,
    ):
        case = build_harmonic_projected_frequency_waveform(jnp)
        data = fsdata(sum_harmonics(case["actual"]["wf"]))
        sdm = make_sdm(
            dense_kernel_3ch(jnp),
            frequencies=case["expected"]["frequencies"].ax,
            channel_names=("X", "Y", "Z"),
        )
        model = noise_model(sdm)
        likelihood = whittle(data, model)

        got = np.asarray(likelihood.get_cross_product(case["actual"]["wf"]))
        expected = np.asarray(
            model.reset().get_scalar_product(
                data,
                fsdata(sum_harmonics(case["actual"]["wf"])),
            ),
        )

        npt.assert_allclose(got, expected)

    def test_template_is_restricted_to_its_frequency_band(self):
        case = build_fd_template_band_case(jnp)
        left = case["actual"]["data"]
        freqs = case["expected"]["frequencies"].asarray(jnp)
        template = left.get_subset(interval=(1.0, 3.0))
        kernel = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (len(freqs), 3, 3))
        model = noise_model(
            make_sdm(kernel, frequencies=freqs, channel_names=("X", "Y", "Z")),
        )
        likelihood = whittle(left, model)

        got = np.asarray(likelihood.get_cross_product(template))
        expected = np.asarray(
            noise_model(
                make_sdm(
                    kernel[1:4],
                    frequencies=template.frequencies.asarray(jnp),
                    channel_names=("X", "Y", "Z"),
                ),
            ).get_scalar_product(left.get_subset(interval=(1.0, 3.0)), template),
        )

        npt.assert_allclose(got, expected)

    def test_whittle_factory_returns_fd_whittle_likelihood(
        self,
    ):
        case = build_fd_pair(jnp)
        left = case["actual"]["left"]
        frequencies = case["expected"]["frequencies"]
        model = noise_model(
            make_sdm(
                dense_kernel_3ch(jnp),
                frequencies=frequencies.asarray(jnp),
                channel_names=("X", "Y", "Z"),
            ),
        )

        likelihood = whittle(left, model)

        assert isinstance(likelihood, FDWhittleLikelihood)
