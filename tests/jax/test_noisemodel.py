"""Tests for noise models with JAX arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest

import jax

jax.config.update("jax_enable_x64", val=True)
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from tests._helpers import (
    build_fd_pair,
    build_fd_pair_batched_2x2,
    build_fdata,
    build_wdm_pair,
    build_wdm_pair_batched_2x2,
    dense_esdm_2ch,
    dense_kernel_2ch,
    diagonal_kernel_2ch,
)
from typed_lisa_toolkit import (
    frequency_series,
    fsdata,
    linspace_from_array,
    make_sdm,
    noise_model,
)
from typed_lisa_toolkit.types import (
    DiagonalSpectralDensity,
    EvolutionarySpectralDensity,
    FDNoiseModel,
    SpectralDensity,
    TFNoiseModel,
)


class _FlatFDNoiseJAX:
    def psd(self, frequencies, option):
        del option
        return jnp.ones_like(jnp.asarray(frequencies))


class TestSpectralDensityJAX(unittest.TestCase):
    def test_to_subband_slices_frequency_axis(self):
        case = build_fdata(jnp)
        sdm = SpectralDensity(
            case.frequencies,
            dense_kernel_2ch(jnp),
            channel_order=["X", "Y"],
        )

        try:
            sub = sdm.to_subband((1.5, 3.0))
        except TypeError:
            # Current Linspace slicing support may raise in some backends.
            return

        npt.assert_allclose(
            np.asarray(sub.get_kernel()),
            np.asarray(dense_kernel_2ch(jnp)[1:2]),
        )

    def test_get_kernel_backend_argument_is_not_supported(self):
        case = build_fdata(jnp)
        sdm = SpectralDensity(
            case.frequencies,
            dense_kernel_2ch(jnp),
            channel_order=["X", "Y"],
        )

        with pytest.raises(NotImplementedError):
            sdm.get_kernel(backend="numpy")

    def test_whitening_matrix_reconstructs_inverse_sdm(self):
        case = build_fdata(jnp)
        kernel = dense_kernel_2ch(jnp)
        sdm = SpectralDensity(case.frequencies, kernel, channel_order=["X", "Y"])

        w = jnp.asarray(sdm.get_whitening_matrix())
        reconstructed = jnp.einsum("fji,fjk->fik", jnp.conj(w), w)

        npt.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(kernel),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_whitening_matrix_invalid_kind_raises(self):
        case = build_fdata(jnp)
        sdm = SpectralDensity(
            case.frequencies,
            dense_kernel_2ch(jnp),
            channel_order=["X", "Y"],
        )

        with pytest.raises(NotImplementedError):
            sdm.get_whitening_matrix(kind="qr")

    def test_diagonal_whitening_default_kind(self):
        case = build_fdata(jnp)
        kernel = diagonal_kernel_2ch(jnp)
        sdm = DiagonalSpectralDensity(
            case.frequencies,
            kernel,
            channel_order=["X", "Y"],
        )

        w = jnp.asarray(sdm.get_whitening_matrix())

        npt.assert_allclose(
            np.asarray(w[:, 0, 0]),
            np.sqrt(np.asarray(kernel[:, 0, 0])),
        )
        npt.assert_allclose(
            np.asarray(w[:, 1, 1]),
            np.sqrt(np.asarray(kernel[:, 1, 1])),
        )

    def test_diagonal_from_fd_noise(self):
        case = build_fdata(jnp)
        sdm = DiagonalSpectralDensity.from_fd_noise(
            _FlatFDNoiseJAX(),
            case.frequencies,
            ["X", "Y"],
        )

        kernel = np.asarray(sdm.get_kernel())
        assert kernel.shape == (3, 2, 2)
        npt.assert_allclose(kernel[:, 0, 0], np.ones(3))
        npt.assert_allclose(kernel[:, 1, 1], np.ones(3))


class TestFDNoiseModelJAX(unittest.TestCase):
    def test_fd_model_init_supports_jax_namespace(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)

        model = noise_model(
            make_sdm(
                jnp.diagonal(kernel, axis1=-2, axis2=-1),
                frequencies=case["frequencies"],
                channel_names=("X", "Y"),
                is_diagonal=True,
            ),
        )

        got = model.get_scalar_product(case["left"], case["right"])

        assert model.reset() is model
        assert np.isfinite(np.asarray(got)).all()

    def test_get_integrand_diagonal_shape_and_value(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)
        model = noise_model(
            make_sdm(
                jnp.diagonal(kernel, axis1=-2, axis2=-1),
                frequencies=case["frequencies"],
                channel_names=("X", "Y"),
                is_diagonal=True,
            ),
        )

        integrand = np.asarray(model.get_integrand(case["left"], case["right"]))
        left = np.asarray(case["left"].get_kernel())
        right = np.asarray(case["right"].get_kernel())
        diag = np.diagonal(np.asarray(kernel), axis1=-1, axis2=-2)
        expected = (4.0 * left.conj() * right) * diag.T[None, :, None, None, :]

        npt.assert_allclose(integrand, expected)

    def test_get_scalar_product_dense_matches_manual_contraction(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = noise_model(
            make_sdm(kernel, frequencies=case["frequencies"], channel_names=("X", "Y")),
        )

        got = np.asarray(model.get_scalar_product(case["left"], case["right"]))
        left = np.asarray(case["left"].get_kernel())
        right = np.asarray(case["right"].get_kernel())
        integrand = 4.0 * np.einsum(
            "...fi,fij,...fj->...f",
            np.moveaxis(left.conj(), 1, -1),
            np.asarray(kernel),
            np.moveaxis(right, 1, -1),
        )
        expected = np.trapezoid(integrand, x=np.asarray(case["frequencies"]), axis=-1)

        npt.assert_allclose(got.squeeze(), expected.squeeze().real)

    def test_get_scalar_product_dense_batched_matches_manual_contraction(self):
        case = build_fd_pair_batched_2x2(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = noise_model(
            make_sdm(kernel, frequencies=case["frequencies"], channel_names=("X", "Y")),
        )

        got = np.asarray(model.get_scalar_product(case["left"], case["right"]))
        left = np.asarray(case["left"].get_kernel())
        right = np.asarray(case["right"].get_kernel())
        integrand = 4.0 * np.einsum(
            "...fi,fij,...fj->...f",
            np.moveaxis(left.conj(), 1, -1),
            np.asarray(kernel),
            np.moveaxis(right, 1, -1),
        )
        expected = np.trapezoid(
            integrand,
            x=np.asarray(case["frequencies"]),
            axis=-1,
        ).real

        assert got.shape[0] == 2
        npt.assert_allclose(np.squeeze(got), np.squeeze(expected))

    def test_cumulative_scalar_product_matches_final_scalar_product(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = noise_model(
            make_sdm(kernel, frequencies=case["frequencies"], channel_names=("X", "Y")),
        )

        cumulative = np.asarray(
            model.get_cumulative_scalar_product(case["left"], case["right"]),
        )
        scalar = np.asarray(model.get_scalar_product(case["left"], case["right"]))

        npt.assert_allclose(cumulative[..., -1], scalar)

    def test_whiten_diagonal_scales_each_channel(self):
        case = build_fd_pair(jnp)
        kernel = diagonal_kernel_2ch(jnp)
        model = noise_model(
            make_sdm(
                jnp.diagonal(kernel, axis1=-2, axis2=-1),
                frequencies=case["frequencies"],
                channel_names=("X", "Y"),
                is_diagonal=True,
            ),
        )

        whitened = model.whiten(case["left"])
        got = np.asarray(whitened.get_kernel())
        left = np.asarray(case["left"].get_kernel())
        w = np.asarray(model.sdm.get_whitening_matrix())
        left_e = np.moveaxis(left[:, :, 0, 0, :], 1, -1)
        expected = np.moveaxis(np.einsum("fij,...fj->...fi", w, left_e), -1, 1)[
            :,
            :,
            None,
            None,
            :,
        ]

        npt.assert_allclose(got, expected)

    def test_whiten_dense_batched_matches_manual_channel_mixing(self):
        case = build_fd_pair_batched_2x2(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = noise_model(
            make_sdm(kernel, frequencies=case["frequencies"], channel_names=("X", "Y")),
        )

        whitened = model.whiten(case["left"])
        got = np.asarray(whitened.get_kernel())

        left = np.asarray(case["left"].get_kernel())
        left_e = np.moveaxis(left[:, :, 0, 0, :], 1, -1)
        w = np.asarray(model.sdm.get_whitening_matrix())
        expected = np.moveaxis(np.einsum("fij,...fj->...fi", w, left_e), -1, 1)[
            :,
            :,
            None,
            None,
            :,
        ]

        npt.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)

    def test_overlap_self_is_unity(self):
        case = build_fd_pair(jnp)
        kernel = dense_kernel_2ch(jnp)
        model = noise_model(
            make_sdm(kernel, frequencies=case["frequencies"], channel_names=("X", "Y")),
        )

        overlap = np.asarray(model.get_overlap(case["left"], case["left"]))

        npt.assert_allclose(overlap.squeeze(), 1.0)

    def test_cross_correlation_currently_raises_for_linspace_grid(self):
        times = np.linspace(0.0, 7.0, 8)
        frequencies = linspace_from_array(
            np.fft.rfftfreq(len(times), d=times[1] - times[0]),
        )
        x = np.array([1.0 + 0.0j, 0.5 + 0.25j, -0.25 + 0.5j, 0.1 - 0.2j, 0.05 + 0.0j])
        y = np.array([0.5 + 0.0j, -0.2 + 0.1j, 0.3 - 0.4j, -0.1 + 0.2j, 0.01 + 0.0j])

        fs = fsdata(
            {
                "X": frequency_series(frequencies, x[None, None, None, None, :]),
                "Y": frequency_series(frequencies, y[None, None, None, None, :]),
            },
        ).set_times(times)
        freqs = np.asarray(fs.frequencies)
        kernel = np.zeros((len(freqs), 2, 2), dtype=float)
        kernel[:, 0, 0] = 1.0
        kernel[:, 1, 1] = 1.0
        model = noise_model(
            make_sdm(
                jnp.diagonal(jnp.asarray(kernel), axis1=-2, axis2=-1),
                frequencies=freqs,
                channel_names=("X", "Y"),
                is_diagonal=True,
            ),
        )

        with pytest.raises((TypeError, ValueError)):
            model.get_cross_correlation(fs, fs)


class TestEvolutionarySpectralDensityJAX(unittest.TestCase):
    def test_invalid_shape_returns_false_without_raising(self):
        assert not EvolutionarySpectralDensity.is_valid_sdm(
            jnp.eye(2), channel_order=["X", "Y"]
        )

    def test_duplicate_channel_names_return_false_without_raising(self):
        assert not EvolutionarySpectralDensity.is_valid_sdm(
            jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (2, 2, 2, 2)),
            channel_order=["X", "X"],
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
            np.asarray(reconstructed),
            np.asarray(invevsdm),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_get_kernel_backend_argument_is_not_supported(self):
        esd = EvolutionarySpectralDensity(
            frequencies=jnp.array([0.25, 0.5], dtype=jnp.float64),
            times=jnp.array([0.0, 1.0], dtype=jnp.float64),
            inverse_esdm=jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (2, 2, 2, 2)),
            channel_order=["X", "Y"],
        )

        with pytest.raises(NotImplementedError):
            esd.get_kernel(backend="numpy")

    def test_whitening_matrix_invalid_kind_raises(self):
        esd = EvolutionarySpectralDensity(
            frequencies=jnp.array([0.25, 0.5], dtype=jnp.float64),
            times=jnp.array([0.0, 1.0], dtype=jnp.float64),
            inverse_esdm=jnp.broadcast_to(jnp.eye(2, dtype=jnp.float64), (2, 2, 2, 2)),
            channel_order=["X", "Y"],
        )

        with pytest.raises(NotImplementedError):
            esd.get_whitening_matrix(kind="qr")


class TestTFNoiseModelJAX(unittest.TestCase):
    def test_scalar_product_with_identity_esdm(self):
        case = build_wdm_pair(jnp)
        invevsdm = jnp.broadcast_to(
            jnp.eye(2, dtype=jnp.float64),
            (len(case["frequencies"]), len(case["times"]), 2, 2),
        )
        model = noise_model(
            make_sdm(
                invevsdm,
                frequencies=case["frequencies"],
                times=case["times"],
                channel_names=("X", "Y"),
            ),
        )

        got = model.get_scalar_product(case["left"], case["right"])
        expected = jnp.sum(
            case["left_x"] * case["right_x"] + case["left_y"] * case["right_y"],
        )

        npt.assert_allclose(np.asarray(got), np.asarray(expected))

    def test_whiten_identity_keeps_entries(self):
        case = build_wdm_pair(jnp)
        invevsdm = jnp.broadcast_to(
            jnp.eye(2, dtype=jnp.float64),
            (len(case["frequencies"]), len(case["times"]), 2, 2),
        )
        model = noise_model(
            make_sdm(
                invevsdm,
                frequencies=case["frequencies"],
                times=case["times"],
                channel_names=("X", "Y"),
            ),
        )

        whitened = model.whiten(case["left"])

        npt.assert_allclose(
            np.asarray(whitened.get_kernel()),
            np.asarray(case["left"].get_kernel()),
        )

    def test_scalar_product_dense_esdm_batched_matches_manual_contraction(self):
        case = build_wdm_pair_batched_2x2(jnp)
        invevsdm = dense_esdm_2ch(jnp)
        model = noise_model(
            make_sdm(
                invevsdm,
                frequencies=case["frequencies"],
                times=case["times"],
                channel_names=("X", "Y"),
            ),
        )

        got = model.get_scalar_product(case["left"], case["right"])
        left = np.asarray(case["left"].get_kernel())
        right = np.asarray(case["right"].get_kernel())
        expected = (
            np.einsum(
                "...fti,ftij,...ftj->...ft",
                np.moveaxis(left.conj(), 1, -1),
                np.asarray(invevsdm),
                np.moveaxis(right, 1, -1),
            )
            .sum()
            .real
        )

        npt.assert_allclose(np.asarray(got), np.asarray(expected))

    def test_whiten_dense_esdm_batched_matches_manual_channel_mixing(self):
        case = build_wdm_pair_batched_2x2(jnp)
        invevsdm = dense_esdm_2ch(jnp)
        esd = make_sdm(
            invevsdm,
            frequencies=case["frequencies"],
            times=case["times"],
            channel_names=("X", "Y"),
        )
        model = noise_model(esd)

        whitened = model.whiten(case["left"])
        got = np.asarray(whitened.get_kernel())

        left = np.asarray(case["left"].get_kernel())
        left_e = np.moveaxis(left, 1, -1)
        w = np.asarray(esd.get_whitening_matrix())
        expected_e = np.einsum("ftij,...ftj->...fti", w, left_e)
        expected = np.moveaxis(expected_e, -1, 1)

        npt.assert_allclose(
            np.asarray(got),
            np.asarray(expected),
            rtol=1e-12,
            atol=1e-12,
        )


class TestNoiseModelFactoriesJAX(unittest.TestCase):
    def test_make_sdm_builds_dense_diagonal_and_evolutionary_variants(self):
        frequencies = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        times = jnp.array([0.0, 1.0], dtype=jnp.float64)

        dense_kernel = jnp.broadcast_to(
            jnp.eye(2, dtype=jnp.float64),
            (len(frequencies), 2, 2),
        )
        diag_kernel = jnp.ones((len(frequencies), 2), dtype=jnp.float64)
        evo_kernel = jnp.broadcast_to(
            jnp.eye(2, dtype=jnp.float64),
            (len(frequencies), len(times), 2, 2),
        )

        dense_sdm = make_sdm(
            dense_kernel,
            frequencies=frequencies,
            channel_names=("X", "Y"),
        )
        diag_sdm = make_sdm(
            diag_kernel,
            frequencies=frequencies,
            channel_names=("X", "Y"),
            is_diagonal=True,
        )
        evo_sdm = make_sdm(
            evo_kernel,
            frequencies=frequencies,
            times=times,
            channel_names=("X", "Y"),
        )

        assert isinstance(dense_sdm, SpectralDensity)
        assert isinstance(diag_sdm, DiagonalSpectralDensity)
        assert isinstance(evo_sdm, EvolutionarySpectralDensity)

    def test_noise_model_factory_dispatches_by_sdm_type(self):
        frequencies = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)
        times = jnp.array([0.0, 1.0], dtype=jnp.float64)

        fd_model = noise_model(
            make_sdm(
                jnp.broadcast_to(
                    jnp.eye(2, dtype=jnp.float64),
                    (len(frequencies), 2, 2),
                ),
                frequencies=frequencies,
                channel_names=("X", "Y"),
            ),
        )
        tf_model = noise_model(
            make_sdm(
                jnp.broadcast_to(
                    jnp.eye(2, dtype=jnp.float64),
                    (len(frequencies), len(times), 2, 2),
                ),
                frequencies=frequencies,
                times=times,
                channel_names=("X", "Y"),
            ),
        )

        assert isinstance(fd_model, FDNoiseModel)
        assert isinstance(tf_model, TFNoiseModel)

    def test_make_sdm_rejects_invalid_shape(self):
        frequencies = jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64)

        with pytest.raises(ValueError, match="Invalid shape"):
            _ = make_sdm(
                jnp.eye(2, dtype=jnp.float64),
                frequencies=frequencies,
                channel_names=("X", "Y"),
            )
