import numpy as np
import numpy.testing as npt
import pytest

from typed_lisa_toolkit import make_sdm, shop
from typed_lisa_toolkit.types import (
    EvolutionarySpectralDensity,
    SpectralDensity,
    TSData,
)


def test_xyz_aet_roundtrip_tsdata(tsdata: TSData):
    xyz = tsdata
    aet = shop.xyz2aet(xyz)
    recovered = shop.aet2xyz(aet)

    assert aet.channel_names == ("A", "E", "T")
    assert recovered.channel_names == xyz.channel_names
    npt.assert_allclose(np.asarray(recovered.times), np.asarray(xyz.times))
    npt.assert_allclose(
        np.asarray(recovered.get_kernel()),
        np.asarray(xyz.get_kernel()),
        atol=1e-12,
    )


def test_xyz_aet_roundtrip_spectral_density(sdm: SpectralDensity):
    xyz_sdm = sdm

    aet_sdm = shop.xyz2aet(xyz_sdm)
    recovered = shop.aet2xyz(aet_sdm)

    assert aet_sdm.channel_order == ("A", "E", "T")
    assert recovered.channel_order == xyz_sdm.channel_order
    npt.assert_allclose(
        np.asarray(recovered.get_kernel()),
        np.asarray(xyz_sdm.get_kernel()),
        atol=1e-12,
    )


def test_xyz_aet_roundtrip_evolutionary_spectral_density(
    esdm: EvolutionarySpectralDensity,
):
    xyz_esdm = esdm

    aet_esdm = shop.xyz2aet(xyz_esdm)
    recovered = shop.aet2xyz(aet_esdm)

    assert aet_esdm.channel_order == ("A", "E", "T")
    assert recovered.channel_order == xyz_esdm.channel_order
    npt.assert_allclose(
        np.asarray(recovered.get_kernel()),
        np.asarray(xyz_esdm.get_kernel()),
        atol=1e-12,
    )


def test_xyz2aet_with_xyz_and_xyz_components_raises(tsdata: TSData):
    xyz = tsdata
    with pytest.raises(ValueError, match="Cannot specify both xyz and X, Y, Z"):
        shop.xyz2aet(xyz, X=np.array([1.0]), Y=np.array([2.0]), Z=np.array([3.0]))  # pyright: ignore[reportCallIssue]


def test_aet2xyz_with_aet_and_aet_components_raises(tsdata: TSData):
    aet = shop.xyz2aet(tsdata)
    with pytest.raises(ValueError, match="Cannot specify both aet and A, E, T"):
        shop.aet2xyz(aet, A=np.array([1.0]), E=np.array([2.0]), T=np.array([3.0]))  # pyright: ignore[reportCallIssue]


def test_xyz2aet_without_inputs_raises():
    with pytest.raises(ValueError, match="Must specify either xyz or all of X, Y, Z"):
        shop.xyz2aet()  # pyright: ignore[reportCallIssue]


def test_aet2xyz_without_inputs_raises():
    with pytest.raises(ValueError, match="Must specify either aet or all of A, E, T"):
        shop.aet2xyz()  # pyright: ignore[reportCallIssue]


def test_xyz2aet_keyword_components_path():
    x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    y = np.array([1.0, 0.5, -1.0], dtype=np.float64)
    z = np.array([-0.5, 0.0, 1.5], dtype=np.float64)

    a, e, t = shop.xyz2aet(X=x, Y=y, Z=z)
    recovered_x, recovered_y, recovered_z = shop.aet2xyz(A=a, E=e, T=t)

    npt.assert_allclose(recovered_x, x, atol=1e-12)
    npt.assert_allclose(recovered_y, y, atol=1e-12)
    npt.assert_allclose(recovered_z, z, atol=1e-12)


def test_mapping_not_channel_mapping_raises_type_error():
    bad_mapping = {
        "X": np.array([1.0, 2.0], dtype=np.float64),
        "Y": np.array([0.0, 3.0], dtype=np.float64),
        "Z": np.array([-1.0, 4.0], dtype=np.float64),
    }
    with pytest.raises(TypeError, match="got dict"):
        shop.xyz2aet(bad_mapping)  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_spectral_density_channel_order_assertions():
    freqs = np.array([0.25, 0.5], dtype=np.float64)
    kernel = np.broadcast_to(np.eye(3, dtype=np.float64), (2, 3, 3)).copy()

    wrong_xyz_input = make_sdm(
        kernel,
        frequencies=freqs,
        channel_names=("A", "E", "T"),
    )
    with pytest.raises(ValueError, match="Expected original channel order"):
        shop.xyz2aet(wrong_xyz_input)

    wrong_aet_input = make_sdm(
        kernel,
        frequencies=freqs,
        channel_names=("X", "Y", "Z"),
    )
    with pytest.raises(ValueError, match="Expected original channel order"):
        shop.aet2xyz(wrong_aet_input)


def test_array_last_dimension_assertions():
    wrong_shape = np.ones((4, 2), dtype=np.float64)

    with pytest.raises(
        ValueError, match="Expected last dimension of input array to be 3"
    ):
        shop.xyz2aet(wrong_shape)

    with pytest.raises(
        ValueError, match="Expected last dimension of input array to be 3"
    ):
        shop.aet2xyz(wrong_shape)
