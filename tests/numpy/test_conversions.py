"""Unit tests for shop/conversions.py (NumPy backend)."""

import unittest

import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import shop, time_series, tsdata
from typed_lisa_toolkit.types import (
    EvolutionarySpectralDensity,
    SpectralDensity,
    TSData,
)


def _build_xyz_tsdata_numpy(n: int = 8) -> TSData:
    times = np.linspace(0.0, 3.5, n)
    x = np.asarray([0.0, 1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.1], dtype=np.float64)
    y = np.asarray([1.0, -0.5, 0.25, 0.0, 0.4, -0.2, 0.6, -0.8], dtype=np.float64)
    z = np.asarray([-0.2, 0.3, -0.1, 0.5, -0.7, 0.9, -0.4, 0.2], dtype=np.float64)
    return tsdata(
        {
            "X": time_series(times, x[None, None, None, None, :]),
            "Y": time_series(times, y[None, None, None, None, :]),
            "Z": time_series(times, z[None, None, None, None, :]),
        }
    )


def _build_xyz_spectral_density_numpy() -> SpectralDensity:
    frequencies = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    inverse_sdm = np.broadcast_to(
        np.array(
            [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]],
            dtype=np.float64,
        ),
        (len(frequencies), 3, 3),
    ).copy()
    return SpectralDensity(frequencies, inverse_sdm, ["X", "Y", "Z"])


def _build_xyz_evolutionary_spectral_density_numpy() -> EvolutionarySpectralDensity:
    frequencies = np.array([0.25, 0.5], dtype=np.float64)
    times = np.array([0.0, 1.0], dtype=np.float64)
    base = np.array(
        [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]],
        dtype=np.float64,
    )
    inverse_esdm = np.zeros((len(frequencies), len(times), 3, 3), dtype=np.float64)
    for i in range(len(frequencies)):
        for j in range(len(times)):
            inverse_esdm[i, j] = base * (1.0 + 0.1 * i + 0.05 * j)
    return EvolutionarySpectralDensity(
        frequencies, times, inverse_esdm, ["X", "Y", "Z"]
    )


class TestConversionsNumpy(unittest.TestCase):
    def test_xyz_aet_roundtrip_tsdata(self):
        xyz = _build_xyz_tsdata_numpy()

        aet = shop.xyz2aet(xyz)
        recovered = shop.aet2xyz(aet)

        self.assertEqual(aet.channel_names, ("A", "E", "T"))
        self.assertEqual(recovered.channel_names, xyz.channel_names)
        npt.assert_allclose(np.asarray(recovered.times), np.asarray(xyz.times))
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()), np.asarray(xyz.get_kernel()), atol=1e-12
        )

    def test_xyz_aet_roundtrip_spectral_density(self):
        xyz_sdm = _build_xyz_spectral_density_numpy()

        aet_sdm = shop.xyz2aet(xyz_sdm)
        recovered = shop.aet2xyz(aet_sdm)

        self.assertEqual(aet_sdm.channel_order, ("A", "E", "T"))
        self.assertEqual(recovered.channel_order, xyz_sdm.channel_order)
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()),
            np.asarray(xyz_sdm.get_kernel()),
            atol=1e-12,
        )

    def test_xyz_aet_roundtrip_evolutionary_spectral_density(self):
        xyz_esdm = _build_xyz_evolutionary_spectral_density_numpy()

        aet_esdm = shop.xyz2aet(xyz_esdm)
        recovered = shop.aet2xyz(aet_esdm)

        self.assertEqual(aet_esdm.channel_order, ("A", "E", "T"))
        self.assertEqual(recovered.channel_order, xyz_esdm.channel_order)
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()),
            np.asarray(xyz_esdm.get_kernel()),
            atol=1e-12,
        )

    def test_xyz2aet_with_xyz_and_xyz_components_raises(self):
        xyz = _build_xyz_tsdata_numpy()
        with self.assertRaisesRegex(ValueError, "Cannot specify both xyz and X, Y, Z"):
            shop.xyz2aet(xyz, X=np.array([1.0]), Y=np.array([2.0]), Z=np.array([3.0]))

    def test_aet2xyz_with_aet_and_aet_components_raises(self):
        aet = shop.xyz2aet(_build_xyz_tsdata_numpy())
        with self.assertRaisesRegex(ValueError, "Cannot specify both aet and A, E, T"):
            shop.aet2xyz(aet, A=np.array([1.0]), E=np.array([2.0]), T=np.array([3.0]))

    def test_xyz2aet_without_inputs_raises(self):
        with self.assertRaisesRegex(
            ValueError, "Must specify either xyz or all of X, Y, Z"
        ):
            shop.xyz2aet()

    def test_aet2xyz_without_inputs_raises(self):
        with self.assertRaisesRegex(
            ValueError, "Must specify either aet or all of A, E, T"
        ):
            shop.aet2xyz()

    def test_xyz2aet_keyword_components_path(self):
        x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        y = np.array([1.0, 0.5, -1.0], dtype=np.float64)
        z = np.array([-0.5, 0.0, 1.5], dtype=np.float64)

        a, e, t = shop.xyz2aet(X=x, Y=y, Z=z)
        recovered_x, recovered_y, recovered_z = shop.aet2xyz(A=a, E=e, T=t)

        npt.assert_allclose(recovered_x, x, atol=1e-12)
        npt.assert_allclose(recovered_y, y, atol=1e-12)
        npt.assert_allclose(recovered_z, z, atol=1e-12)

    def test_mapping_not_channel_mapping_raises_type_error(self):
        bad_mapping = {
            "X": np.array([1.0, 2.0], dtype=np.float64),
            "Y": np.array([0.0, 3.0], dtype=np.float64),
            "Z": np.array([-1.0, 4.0], dtype=np.float64),
        }
        with self.assertRaisesRegex(TypeError, "got dict"):
            shop.xyz2aet(bad_mapping)

    def test_spectral_density_channel_order_assertions(self):
        freqs = np.array([0.25, 0.5], dtype=np.float64)
        kernel = np.broadcast_to(np.eye(3, dtype=np.float64), (2, 3, 3)).copy()

        wrong_xyz_input = SpectralDensity(freqs, kernel, ["A", "E", "T"])
        with self.assertRaisesRegex(AssertionError, "Expected original channel order"):
            shop.xyz2aet(wrong_xyz_input)

        wrong_aet_input = SpectralDensity(freqs, kernel, ["X", "Y", "Z"])
        with self.assertRaisesRegex(AssertionError, "Expected original channel order"):
            shop.aet2xyz(wrong_aet_input)

    def test_array_last_dimension_assertions(self):
        wrong_shape = np.ones((4, 2), dtype=np.float64)

        with self.assertRaisesRegex(
            AssertionError, "Expected last dimension of input array to be 3"
        ):
            shop.xyz2aet(wrong_shape)

        with self.assertRaisesRegex(
            AssertionError, "Expected last dimension of input array to be 3"
        ):
            shop.aet2xyz(wrong_shape)


if __name__ == "__main__":
    unittest.main()
