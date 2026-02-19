"""Tests for noise models with JAX arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit.consumers.noisemodel import (
    DiagonalSpectralDensity,
    EvolutionarySpectralDensity,
    FDNoiseModel,
    SpectralDensity,
    TFNoiseModel,
)
from tests._shared.noisemodel_helpers import (
    build_fd_pair,
    build_wdm_pair,
    build_wdm_pair_batched_2x2,
    dense_esdm_2ch,
    dense_kernel_2ch,
    diagonal_kernel_2ch,
)


class TestSpectralDensityJAX(unittest.TestCase):
    def test_to_subband_slices_frequency_axis(self):
        case = build_fd_pair(jnp)
        psd = SpectralDensity(
            case["frequencies"], dense_kernel_2ch(jnp), channel_order=["X", "Y"]
        )

        sub = psd.to_subband((1.5, 3.0))

        npt.assert_allclose(
            np.asarray(sub.get_kernel()), np.asarray(dense_kernel_2ch(jnp)[1:2])
        )

    def test_whitening_matrix_reconstructs_inverse_sdm(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        psd = SpectralDensity(case["frequencies"], kernel, channel_order=["X", "Y"])

        w = jnp.asarray(psd.get_whitening_matrix())
        reconstructed = jnp.einsum("fji,fjk->fik", jnp.conj(w), w)

        npt.assert_allclose(np.asarray(reconstructed), np.asarray(kernel), rtol=1e-12, atol=1e-12)

    def test_diagonal_whitening_default_kind(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)
        psd = DiagonalSpectralDensity(case["frequencies"], kernel, channel_order=["X", "Y"])

        w = jnp.asarray(psd.get_whitening_matrix())

        npt.assert_allclose(np.asarray(w[:, 0, 0]), np.sqrt(np.asarray(kernel[:, 0, 0])))
        npt.assert_allclose(np.asarray(w[:, 1, 1]), np.sqrt(np.asarray(kernel[:, 1, 1])))


class TestFDNoiseModelJAX(unittest.TestCase):
    def test_fd_model_init_is_not_supported_for_jax_yet(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)

        with self.assertRaises(NotImplementedError):
            FDNoiseModel(
                DiagonalSpectralDensity(case["frequencies"], kernel, ["X", "Y"])
            )


class TestEvolutionarySpectralDensityJAX(unittest.TestCase):
    def test_whitening_matrix_reconstructs_inverse_esdm(self):
        invevsdm = jnp.array(
            [
                [
                    [[2.0, 0.3], [0.3, 1.0]],
                    [[1.5, 0.2], [0.2, 0.8]],
                ],
                [
                    [[2.5, 0.1], [0.1, 1.2]],
                    [[1.8, -0.2], [-0.2, 1.0]],
                ],
            ],
            dtype=jnp.float64,
        )
        esd = EvolutionarySpectralDensity(
            frequencies=jnp.array([0.25, 0.5]),
            times=jnp.array([0.0, 1.0]),
            inverse_esdm=invevsdm,
            channel_order=["X", "Y"],
        )

        w = jnp.asarray(esd.get_whitening_matrix())
        reconstructed = jnp.einsum("ftji,ftjk->ftik", jnp.conj(w), w)

        npt.assert_allclose(np.asarray(reconstructed), np.asarray(invevsdm), rtol=1e-12, atol=1e-12)


class TestTFNoiseModelJAX(unittest.TestCase):
    def test_scalar_product_with_identity_esdm(self):
        case = build_wdm_pair(jnp)
        invevsdm = jnp.broadcast_to(
            jnp.eye(2, dtype=jnp.float64),
            (len(case["frequencies"]), len(case["times"]), 2, 2),
        )
        model = TFNoiseModel(
            EvolutionarySpectralDensity(
                case["frequencies"], case["times"], invevsdm, ["X", "Y"]
            )
        )

        got = model.get_scalar_product(case["left"], case["right"])
        expected = jnp.sum(
            case["left_x"] * case["right_x"] + case["left_y"] * case["right_y"]
        )

        npt.assert_allclose(np.asarray(got), np.asarray(expected))

    def test_whiten_identity_keeps_entries(self):
        case = build_wdm_pair(jnp)
        invevsdm = jnp.broadcast_to(
            jnp.eye(2, dtype=jnp.float64),
            (len(case["frequencies"]), len(case["times"]), 2, 2),
        )
        model = TFNoiseModel(
            EvolutionarySpectralDensity(
                case["frequencies"], case["times"], invevsdm, ["X", "Y"]
            )
        )

        whitened = model.whiten(case["left"])

        npt.assert_allclose(
            np.asarray(whitened.get_kernel()), np.asarray(case["left"].get_kernel())
        )

    def test_scalar_product_dense_esdm_batched_matches_manual_contraction(self):
        case = build_wdm_pair_batched_2x2(jnp)
        invevsdm = dense_esdm_2ch(jnp)
        model = TFNoiseModel(
            EvolutionarySpectralDensity(
                case["frequencies"], case["times"], invevsdm, ["X", "Y"]
            )
        )

        got = model.get_scalar_product(case["left"], case["right"])
        left = jnp.stack([case["left_x"], case["left_y"]], axis=-1)
        right = jnp.stack([case["right_x"], case["right_y"]], axis=-1)
        expected = jnp.einsum("bfti,ftij,bftj->", left, invevsdm, right).real

        npt.assert_allclose(np.asarray(got), np.asarray(expected))

    def test_whiten_dense_esdm_batched_matches_manual_channel_mixing(self):
        case = build_wdm_pair_batched_2x2(jnp)
        invevsdm = dense_esdm_2ch(jnp)
        esd = EvolutionarySpectralDensity(
            case["frequencies"], case["times"], invevsdm, ["X", "Y"]
        )
        model = TFNoiseModel(esd)

        whitened = model.whiten(case["left"])
        got = jnp.asarray(whitened.get_kernel())[:, :, 0, 0, :, :]

        left = jnp.asarray(case["left"].get_kernel())[:, :, 0, 0, :, :]
        left_e = jnp.moveaxis(left, 1, -1)
        w = jnp.asarray(esd.get_whitening_matrix())
        expected_e = jnp.einsum("ftij,bftj->bfti", w, left_e)
        expected = jnp.moveaxis(expected_e, -1, 1)

        self.assertEqual(got.shape, (2, 2, 2, 3))
        npt.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12, atol=1e-12)
