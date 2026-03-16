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
    build_fd_pair_batched_2x2,
    build_wdm_pair,
    build_wdm_pair_batched_2x2,
    dense_esdm_2ch,
    dense_kernel_2ch,
    diagonal_kernel_2ch,
)


class _FlatFDNoiseJAX:
    def psd(self, frequencies, option):
        del option
        return jnp.ones_like(frequencies)


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

    def test_get_kernel_backend_argument_is_not_supported(self):
        case = build_fd_pair(jnp)
        psd = SpectralDensity(
            case["frequencies"], dense_kernel_2ch(jnp), channel_order=["X", "Y"]
        )

        with self.assertRaises(NotImplementedError):
            psd.get_kernel(backend="numpy")

    def test_whitening_matrix_reconstructs_inverse_sdm(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        psd = SpectralDensity(case["frequencies"], kernel, channel_order=["X", "Y"])

        w = jnp.asarray(psd.get_whitening_matrix())
        reconstructed = jnp.einsum("fji,fjk->fik", jnp.conj(w), w)

        npt.assert_allclose(
            np.asarray(reconstructed), np.asarray(kernel), rtol=1e-12, atol=1e-12
        )

    def test_whitening_matrix_invalid_kind_raises(self):
        case = build_fd_pair(jnp)
        psd = SpectralDensity(
            case["frequencies"], dense_kernel_2ch(jnp), channel_order=["X", "Y"]
        )

        with self.assertRaises(NotImplementedError):
            psd.get_whitening_matrix(kind="qr")

    def test_diagonal_whitening_default_kind(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)
        psd = DiagonalSpectralDensity(
            case["frequencies"], kernel, channel_order=["X", "Y"]
        )

        w = jnp.asarray(psd.get_whitening_matrix())

        npt.assert_allclose(
            np.asarray(w[:, 0, 0]), np.sqrt(np.asarray(kernel[:, 0, 0]))
        )
        npt.assert_allclose(
            np.asarray(w[:, 1, 1]), np.sqrt(np.asarray(kernel[:, 1, 1]))
        )

    def test_diagonal_from_fd_noise(self):
        case = build_fd_pair(jnp)
        psd = DiagonalSpectralDensity.from_fd_noise(
            _FlatFDNoiseJAX(), case["frequencies"], ["X", "Y"]
        )

        kernel = np.asarray(psd.get_kernel())
        self.assertEqual(kernel.shape, (3, 2, 2))
        npt.assert_allclose(kernel[:, 0, 0], np.ones(3))
        npt.assert_allclose(kernel[:, 1, 1], np.ones(3))


class TestFDNoiseModelJAX(unittest.TestCase):
    def test_fd_model_init_supports_jax_namespace(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)

        model = FDNoiseModel(
            DiagonalSpectralDensity(case["frequencies"], kernel, ["X", "Y"])
        )

        got = model.get_scalar_product(case["left"], case["right"])

        self.assertIs(model.reset(), model)
        self.assertTrue(np.isfinite(np.asarray(got)).all())

    def test_get_integrand_diagonal_shape_and_value(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)
        model = FDNoiseModel(
            DiagonalSpectralDensity(case["frequencies"], kernel, ["X", "Y"])
        )

        integrand = np.asarray(model.get_integrand(case["left"], case["right"]))
        expected_x = 4.0 * case["left_x"].conj() * case["right_x"] * kernel[:, 0, 0]
        expected_y = 4.0 * case["left_y"].conj() * case["right_y"] * kernel[:, 1, 1]

        npt.assert_allclose(integrand[:, 0, 0, 0, :], np.asarray(expected_x)[None, :])
        npt.assert_allclose(integrand[:, 1, 0, 0, :], np.asarray(expected_y)[None, :])

    def test_get_scalar_product_dense_matches_manual_contraction(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = FDNoiseModel(SpectralDensity(case["frequencies"], kernel, ["X", "Y"]))

        got = np.asarray(model.get_scalar_product(case["left"], case["right"]))
        left = jnp.stack([case["left_x"], case["left_y"]], axis=-1)
        right = jnp.stack([case["right_x"], case["right_y"]], axis=-1)
        integrand = 4.0 * jnp.einsum("fi,fij,fj->f", jnp.conj(left), kernel, right)
        expected = jnp.trapezoid(integrand, x=jnp.asarray(case["frequencies"]))

        npt.assert_allclose(got.squeeze(), np.asarray(expected.real))

    def test_get_scalar_product_dense_batched_matches_manual_contraction(self):
        case = build_fd_pair_batched_2x2(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = FDNoiseModel(SpectralDensity(case["frequencies"], kernel, ["X", "Y"]))

        got = np.asarray(model.get_scalar_product(case["left"], case["right"]))
        left = jnp.stack([case["left_x"], case["left_y"]], axis=-1)
        right = jnp.stack([case["right_x"], case["right_y"]], axis=-1)
        integrand = 4.0 * jnp.einsum("bfi,fij,bfj->bf", jnp.conj(left), kernel, right)
        expected = jnp.trapezoid(
            integrand,
            x=jnp.asarray(case["frequencies"]),
            axis=-1,
        ).real

        self.assertEqual(got.shape, (2, 1, 1))
        npt.assert_allclose(got[..., 0, 0], np.asarray(expected))

    def test_cumulative_scalar_product_matches_final_scalar_product(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = FDNoiseModel(SpectralDensity(case["frequencies"], kernel, ["X", "Y"]))

        cumulative = np.asarray(
            model.get_cumulative_scalar_product(case["left"], case["right"])
        )
        scalar = np.asarray(model.get_scalar_product(case["left"], case["right"]))

        npt.assert_allclose(cumulative[..., -1], scalar)

    def test_whiten_diagonal_scales_each_channel(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)
        model = FDNoiseModel(
            DiagonalSpectralDensity(case["frequencies"], kernel, ["X", "Y"])
        )

        whitened = model.whiten(case["left"])
        got = np.asarray(whitened.get_kernel())
        expected_x = case["left_x"] * jnp.sqrt(kernel[:, 0, 0])
        expected_y = case["left_y"] * jnp.sqrt(kernel[:, 1, 1])

        npt.assert_allclose(got[:, 0, 0, 0, :], np.asarray(expected_x)[None, :])
        npt.assert_allclose(got[:, 1, 0, 0, :], np.asarray(expected_y)[None, :])

    def test_whiten_dense_batched_matches_manual_channel_mixing(self):
        case = build_fd_pair_batched_2x2(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = FDNoiseModel(SpectralDensity(case["frequencies"], kernel, ["X", "Y"]))

        whitened = model.whiten(case["left"])
        got = np.asarray(whitened.get_kernel())[:, :, 0, 0, :]

        left = np.asarray(case["left"].get_kernel())[:, :, 0, 0, :]
        left_e = np.moveaxis(left, 1, -1)
        w = np.asarray(model.psd.get_whitening_matrix())
        expected_e = np.einsum("fij,bfj->bfi", w, left_e)
        expected = np.moveaxis(expected_e, -1, 1)

        self.assertEqual(got.shape, (2, 2, 3))
        npt.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_overlap_self_is_unity(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = FDNoiseModel(SpectralDensity(case["frequencies"], kernel, ["X", "Y"]))

        overlap = np.asarray(model.get_overlap(case["left"], case["left"]))

        npt.assert_allclose(overlap.squeeze(), 1.0)

    def test_cross_correlation_currently_raises_for_linspace_grid(self):
        times = np.linspace(0.0, 7.0, 8)
        frequencies = np.fft.rfftfreq(len(times), d=times[1] - times[0])
        x = np.array([1.0 + 0.0j, 0.5 + 0.25j, -0.25 + 0.5j, 0.1 - 0.2j, 0.05 + 0.0j])
        y = np.array([0.5 + 0.0j, -0.2 + 0.1j, 0.3 - 0.4j, -0.1 + 0.2j, 0.01 + 0.0j])
        from typed_lisa_toolkit.containers.data import FSData
        from typed_lisa_toolkit.containers.representations import FrequencySeries

        fs = FSData.from_dict(
            {
                "X": FrequencySeries((frequencies,), x[None, None, None, None, :]),
                "Y": FrequencySeries((frequencies,), y[None, None, None, None, :]),
            }
        ).set_times(times)
        freqs = np.asarray(fs.frequencies)
        kernel = np.zeros((len(freqs), 2, 2), dtype=float)
        kernel[:, 0, 0] = 1.0
        kernel[:, 1, 1] = 1.0
        model = FDNoiseModel(DiagonalSpectralDensity(freqs, kernel, ["X", "Y"]))

        with self.assertRaises((TypeError, ValueError)):
            model.get_cross_correlation(fs, fs)


class TestEvolutionarySpectralDensityJAX(unittest.TestCase):
    def test_invalid_shape_returns_false_without_raising(self):
        self.assertFalse(
            EvolutionarySpectralDensity.is_valid_sdm(
                jnp.eye(2),
                channel_order=["X", "Y"],
            )
        )

    def test_duplicate_channel_names_return_false_without_raising(self):
        self.assertFalse(
            EvolutionarySpectralDensity.is_valid_sdm(
                jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (2, 2, 2, 2)),
                channel_order=["X", "X"],
            )
        )

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

        npt.assert_allclose(
            np.asarray(reconstructed), np.asarray(invevsdm), rtol=1e-12, atol=1e-12
        )

    def test_get_kernel_backend_argument_is_not_supported(self):
        esd = EvolutionarySpectralDensity(
            frequencies=jnp.array([0.25, 0.5], dtype=jnp.float64),
            times=jnp.array([0.0, 1.0], dtype=jnp.float64),
            inverse_esdm=jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (2, 2, 2, 2)),
            channel_order=["X", "Y"],
        )

        with self.assertRaises(NotImplementedError):
            esd.get_kernel(backend="numpy")

    def test_whitening_matrix_invalid_kind_raises(self):
        esd = EvolutionarySpectralDensity(
            frequencies=jnp.array([0.25, 0.5], dtype=jnp.float64),
            times=jnp.array([0.0, 1.0], dtype=jnp.float64),
            inverse_esdm=jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (2, 2, 2, 2)),
            channel_order=["X", "Y"],
        )

        with self.assertRaises(NotImplementedError):
            esd.get_whitening_matrix(kind="qr")


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

        self.assertEqual(
            got.shape,
            (2, 2, len(case["frequencies"]), len(case["times"])),
        )
        npt.assert_allclose(
            np.asarray(got), np.asarray(expected), rtol=1e-12, atol=1e-12
        )
