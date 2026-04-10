"""Unit tests for shop/conversions.py (NumPy backend)."""

import unittest

import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import shop, time_series
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
    return TSData.from_dict(
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
    return EvolutionarySpectralDensity(frequencies, times, inverse_esdm, ["X", "Y", "Z"])


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


if __name__ == "__main__":
    unittest.main()
