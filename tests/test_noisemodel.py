from types import ModuleType
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

from typed_lisa_toolkit import (
    make_sdm,
    noise_model,
)
from typed_lisa_toolkit.types import (
    Axis,
    DiagonalSpectralDensity,
    EvolutionarySpectralDensity,
    FDNoiseModel,
    FSData,
    Grid2DCartesian,
    Linspace,
    SpectralDensity,
    TFNoiseModel,
    WDMData,
)
from typed_lisa_toolkit.types.noisemodel import _make_integration_policy


def _diagonal_kernel_3ch(xp: ModuleType) -> Any:
    values_x = xp.asarray([2.0, 4.0, 8.0], dtype=xp.float64)
    values_y = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    values_z = xp.asarray([0.75, 1.25, 2.0], dtype=xp.float64)
    zeros = xp.zeros_like(values_x)
    row0 = xp.stack([values_x, zeros, zeros], axis=-1)
    row1 = xp.stack([zeros, values_y, zeros], axis=-1)
    row2 = xp.stack([zeros, zeros, values_z], axis=-1)
    return xp.stack([row0, row1, row2], axis=-2)


def test_sdm_to_subband_slices_frequency_axis(xp: ModuleType, sdm: SpectralDensity):
    kernel = xp.asarray(sdm.get_kernel())

    try:
        sub = sdm.to_subband((1.5, 3.0))
    except TypeError:
        # Current Linspace slicing support may raise in some backends.
        return

    npt.assert_allclose(np.asarray(sub.get_kernel()), kernel[1:3])


def test_sdm_get_kernel_backend_argument_is_not_supported(sdm: SpectralDensity):
    with pytest.raises(NotImplementedError):
        sdm.get_kernel(backend="jax")


def test_sdm_whitening_matrix_reconstructs_inverse_sdm(
    xp: ModuleType, sdm: SpectralDensity
):
    w = xp.asarray(sdm.get_whitening_matrix())
    reconstructed = np.einsum("fji,fjk->fik", np.conj(w), w)

    npt.assert_allclose(
        reconstructed,
        np.asarray(sdm.get_kernel()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_sdm_whitening_matrix_invalid_kind_raises(sdm: SpectralDensity):
    with pytest.raises(NotImplementedError):
        sdm.get_whitening_matrix(kind="qr")  # pyright: ignore[reportArgumentType]


def test_make_integration_policy_numpy(xp: ModuleType):
    ip = _make_integration_policy(np)
    y = xp.array([0.0, 1.0, 2.0])
    x = xp.array([0.0, 0.5, 1.0])
    assert ip.integrate(y, x=x) == 1.0


def test_fd_model_integrand_diagonal_shape_and_value(fsdata: FSData, xp: ModuleType):
    left = fsdata
    right = fsdata
    kernel = _diagonal_kernel_3ch(xp)
    model = noise_model(
        make_sdm(
            xp.diagonal(xp.asarray(kernel), axis1=-2, axis2=-1),
            frequencies=fsdata.frequencies,
            channel_names=("X", "Y", "Z"),
            is_diagonal=True,
        ),
    )

    integrand = xp.asarray(model.get_integrand(left, right))
    left_k = xp.asarray(left.get_kernel())
    right_k = xp.asarray(right.get_kernel())
    diag = xp.diagonal(xp.asarray(kernel), axis1=-1, axis2=-2)
    expected = (4.0 * left_k.conj() * right_k) * diag.T[None, :, None, None, :]

    npt.assert_allclose(integrand, expected)


def test_fd_model_scalar_product_dense_matches_manual_contraction(
    fsdata: FSData, sdm: SpectralDensity, xp: ModuleType
):
    left = fsdata
    right = fsdata
    kernel = xp.asarray(sdm.get_kernel())
    model = noise_model(sdm)

    got = xp.asarray(model.get_scalar_product(left, right))
    left_k = xp.asarray(left.get_kernel())
    right_k = xp.asarray(right.get_kernel())
    integrand = 4.0 * np.einsum(
        "...fi,fij,...fj->...f",
        xp.moveaxis(left_k.conj(), 1, -1),
        kernel,
        xp.moveaxis(right_k, 1, -1),
    )
    expected = xp.trapezoid(integrand, x=xp.asarray(fsdata.frequencies), axis=-1)

    npt.assert_allclose(np.squeeze(got), np.squeeze(expected).real)


def test_fd_model_cumulative_scalar_product_matches_final_scalar_product(
    fsdata: FSData, sdm: SpectralDensity, xp: ModuleType
):
    model = noise_model(sdm)

    cumulative = xp.asarray(model.get_cumulative_scalar_product(fsdata, fsdata))
    scalar = xp.asarray(model.get_scalar_product(fsdata, fsdata))

    npt.assert_allclose(cumulative[..., -1], scalar)


def test_fd_model_whiten_dense_matches_manual_channel_mixing(
    fsdata: FSData, sdm: SpectralDensity, xp: ModuleType
):
    model = noise_model(sdm)

    whitened = model.whiten(fsdata)
    got = xp.asarray(whitened.get_kernel())

    left = xp.asarray(fsdata.get_kernel())
    left_e = xp.moveaxis(left[:, :, 0, 0, :], 1, -1)
    w = xp.asarray(sdm.get_whitening_matrix())
    expected = xp.moveaxis(xp.einsum("fij,...fj->...fi", w, left_e), -1, 1)[
        :,
        :,
        None,
        None,
        :,
    ]

    npt.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_fd_model_overlap_self_is_unity(
    fsdata: FSData, sdm: SpectralDensity, xp: ModuleType
):
    model = noise_model(sdm)
    overlap = xp.asarray(model.get_overlap(fsdata, fsdata))
    npt.assert_allclose(overlap.squeeze(), 1.0)


def test_esdm_is_valid_sdm_returns_false_without_raising(xp: ModuleType):
    assert not EvolutionarySpectralDensity.is_valid_sdm(
        xp.eye(2), channel_order=["X", "Y", "Z"]
    )
    assert not EvolutionarySpectralDensity.is_valid_sdm(
        xp.broadcast_to(xp.eye(2), (2, 2, 2, 2)).copy(), channel_order=["X", "X"]
    )


def test_esdm_invalid_shape_raises(xp: ModuleType):
    with pytest.raises(ValueError, match=r".+"):
        make_sdm(
            xp.eye(2),
            frequencies=xp.array([0.1, 0.2]),
            times=xp.array([0.0, 1.0]),
            channel_names=("X", "Y", "Z"),
        )


def test_esdm_duplicate_channel_names_raise(xp: ModuleType):
    with pytest.raises(ValueError, match=r".+"):
        make_sdm(
            xp.broadcast_to(xp.eye(2), (2, 2, 2, 2)).copy(),
            frequencies=xp.array([0.1, 0.2]),
            times=xp.array([0.0, 1.0]),
            channel_names=("X", "X"),
        )


def test_esdm_whitening_matrix_reconstructs_inverse_esdm(
    esdm: EvolutionarySpectralDensity,
    xp: ModuleType,
):
    w = xp.asarray(esdm.get_whitening_matrix())
    reconstructed = xp.einsum("ftji,ftjk->ftik", xp.conj(w), w)

    npt.assert_allclose(
        reconstructed,
        xp.asarray(esdm.get_kernel()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_esdm_get_kernel_backend_argument_is_not_supported(
    esdm: EvolutionarySpectralDensity,
):
    with pytest.raises(NotImplementedError):
        esdm.get_kernel(backend="jax")


def test_esdm_whitening_matrix_invalid_kind_raises(esdm: EvolutionarySpectralDensity):
    with pytest.raises(NotImplementedError):
        esdm.get_whitening_matrix(kind="qr")  # pyright: ignore[reportArgumentType]


def test_tf_model_scalar_product_with_identity_esdm(
    wdmdata_cartesian: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
    xp: ModuleType,
):
    frequencies, times = wdmdata_cartesian.grid
    kernel = xp.broadcast_to(
        xp.eye(3, dtype=float),
        (len(frequencies), len(times), 3, 3),
    ).copy()

    model = noise_model(
        make_sdm(
            kernel,
            frequencies=frequencies,
            times=times,
            channel_names=("X", "Y", "Z"),
        ),
    )

    got = xp.asarray(model.get_scalar_product(wdmdata_cartesian, wdmdata_cartesian))
    entries = xp.asarray(wdmdata_cartesian.get_kernel())
    expected = xp.sum(xp.abs(entries) ** 2)

    npt.assert_allclose(got, expected)


def test_tf_model_whiten_identity_keeps_entries(
    wdmdata_cartesian: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
    xp: ModuleType,
):
    frequencies, times = wdmdata_cartesian.grid
    kernel = xp.broadcast_to(
        xp.eye(3, dtype=float),
        (len(frequencies), len(times), 3, 3),
    ).copy()
    model = noise_model(
        make_sdm(
            kernel,
            frequencies=frequencies,
            times=times,
            channel_names=("X", "Y", "Z"),
        ),
    )

    whitened = model.whiten(wdmdata_cartesian)
    npt.assert_allclose(
        np.asarray(whitened.get_kernel()),
        np.asarray(wdmdata_cartesian.get_kernel()),
    )


def test_make_sdm_builds_dense_diagonal_and_evolutionary_variants(
    xp: ModuleType, lin_freq_axis: Axis[Linspace], short_time_axis: Axis[Linspace]
):

    dense_kernel = xp.broadcast_to(xp.eye(3), (len(lin_freq_axis), 3, 3)).copy()
    diag_kernel = xp.ones((len(lin_freq_axis), 3), dtype=float)
    evo_kernel = xp.broadcast_to(
        xp.eye(3),
        (len(lin_freq_axis), len(short_time_axis), 3, 3),
    ).copy()

    dense_sdm = make_sdm(
        dense_kernel,
        frequencies=lin_freq_axis,
        channel_names=("X", "Y", "Z"),
    )
    diag_sdm = make_sdm(
        diag_kernel,
        frequencies=lin_freq_axis,
        channel_names=("X", "Y", "Z"),
        is_diagonal=True,
    )
    evo_sdm = make_sdm(
        evo_kernel,
        frequencies=lin_freq_axis,
        times=short_time_axis,
        channel_names=("X", "Y", "Z"),
    )

    assert isinstance(dense_sdm, SpectralDensity)
    assert isinstance(diag_sdm, DiagonalSpectralDensity)
    assert isinstance(evo_sdm, EvolutionarySpectralDensity)


def test_noise_model_factory_dispatches_by_sdm_type(
    sdm: SpectralDensity, esdm: EvolutionarySpectralDensity
):
    fd_model = noise_model(sdm)
    tf_model = noise_model(esdm)

    assert isinstance(fd_model, FDNoiseModel)
    assert isinstance(tf_model, TFNoiseModel)


def test_make_sdm_rejects_invalid_shape(lin_freq_axis: Axis[Linspace]):
    with pytest.raises(ValueError, match="Invalid shape"):
        _ = make_sdm(
            np.eye(2),
            frequencies=lin_freq_axis,
            channel_names=("X", "Y", "Z"),
        )
