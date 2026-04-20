"""Unit tests for shop/conversions.py (JAX backend)."""

import unittest

import jax

jax.config.update("jax_enable_x64", val=True)
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import linspace, shop, time_series, tsdata
from typed_lisa_toolkit.types import (
    EvolutionarySpectralDensity,
    SpectralDensity,
    TSData,
)


def _build_xyz_tsdata_jax(n: int = 8) -> TSData:
    times = linspace(0.0, 3.5, n)
    x = jnp.asarray([0.0, 1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.1], dtype=jnp.float64)
    y = jnp.asarray([1.0, -0.5, 0.25, 0.0, 0.4, -0.2, 0.6, -0.8], dtype=jnp.float64)
    z = jnp.asarray([-0.2, 0.3, -0.1, 0.5, -0.7, 0.9, -0.4, 0.2], dtype=jnp.float64)
    return tsdata(
        {
            "X": time_series(times, x[None, None, None, None, :]),
            "Y": time_series(times, y[None, None, None, None, :]),
            "Z": time_series(times, z[None, None, None, None, :]),
        },
    )


def _build_xyz_spectral_density_jax() -> SpectralDensity:
    frequencies = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float64)
    inverse_sdm = jnp.broadcast_to(
        jnp.array(
            [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]],
            dtype=jnp.float64,
        ),
        (len(frequencies), 3, 3),
    )
    return SpectralDensity(frequencies, inverse_sdm, ["X", "Y", "Z"])


def _build_xyz_evolutionary_spectral_density_jax() -> EvolutionarySpectralDensity:
    frequencies = jnp.array([0.25, 0.5], dtype=jnp.float64)
    times = jnp.array([0.0, 1.0], dtype=jnp.float64)
    base = jnp.array(
        [[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 1.2]],
        dtype=jnp.float64,
    )
    scales = jnp.array([[1.0, 1.05], [1.1, 1.15]], dtype=jnp.float64)
    inverse_esdm = scales[:, :, None, None] * base[None, None, :, :]
    return EvolutionarySpectralDensity(
        frequencies,
        times,
        inverse_esdm,
        ["X", "Y", "Z"],
    )


class TestConversionsJax(unittest.TestCase):
    def test_xyz_aet_roundtrip_tsdata(self):
        xyz = _build_xyz_tsdata_jax()

        aet = shop.xyz2aet(xyz)
        recovered = shop.aet2xyz(aet)

        self.assertEqual(aet.channel_names, ("A", "E", "T"))
        self.assertEqual(recovered.channel_names, xyz.channel_names)
        npt.assert_allclose(np.asarray(recovered.times), np.asarray(xyz.times))
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()),
            np.asarray(xyz.get_kernel()),
            atol=1e-12,
        )

    def test_xyz_aet_roundtrip_spectral_density(self):
        xyz_sdm = _build_xyz_spectral_density_jax()

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
        xyz_esdm = _build_xyz_evolutionary_spectral_density_jax()

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
        xyz = _build_xyz_tsdata_jax()
        with self.assertRaisesRegex(ValueError, "Cannot specify both xyz and X, Y, Z"):
            shop.xyz2aet(
                xyz,
                X=jnp.array([1.0], dtype=jnp.float64),
                Y=jnp.array([2.0], dtype=jnp.float64),
                Z=jnp.array([3.0], dtype=jnp.float64),
            )

    def test_aet2xyz_with_aet_and_aet_components_raises(self):
        aet = shop.xyz2aet(_build_xyz_tsdata_jax())
        with self.assertRaisesRegex(ValueError, "Cannot specify both aet and A, E, T"):
            shop.aet2xyz(
                aet,
                A=jnp.array([1.0], dtype=jnp.float64),
                E=jnp.array([2.0], dtype=jnp.float64),
                T=jnp.array([3.0], dtype=jnp.float64),
            )

    def test_xyz2aet_without_inputs_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "Must specify either xyz or all of X, Y, Z",
        ):
            shop.xyz2aet()

    def test_aet2xyz_without_inputs_raises(self):
        with self.assertRaisesRegex(
            ValueError,
            "Must specify either aet or all of A, E, T",
        ):
            shop.aet2xyz()

    def test_xyz2aet_keyword_components_path(self):
        x = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64)
        y = jnp.array([1.0, 0.5, -1.0], dtype=jnp.float64)
        z = jnp.array([-0.5, 0.0, 1.5], dtype=jnp.float64)

        a, e, t = shop.xyz2aet(X=x, Y=y, Z=z)
        recovered_x, recovered_y, recovered_z = shop.aet2xyz(A=a, E=e, T=t)

        npt.assert_allclose(np.asarray(recovered_x), np.asarray(x), atol=1e-12)
        npt.assert_allclose(np.asarray(recovered_y), np.asarray(y), atol=1e-12)
        npt.assert_allclose(np.asarray(recovered_z), np.asarray(z), atol=1e-12)

    def test_mapping_not_channel_mapping_raises_type_error(self):
        bad_mapping = {
            "X": jnp.array([1.0, 2.0], dtype=jnp.float64),
            "Y": jnp.array([0.0, 3.0], dtype=jnp.float64),
            "Z": jnp.array([-1.0, 4.0], dtype=jnp.float64),
        }
        with self.assertRaisesRegex(TypeError, "got dict"):
            shop.xyz2aet(bad_mapping)

    def test_spectral_density_channel_order_assertions(self):
        freqs = jnp.array([0.25, 0.5], dtype=jnp.float64)
        kernel = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float64), (2, 3, 3))

        wrong_xyz_input = SpectralDensity(freqs, kernel, ["A", "E", "T"])
        with self.assertRaisesRegex(ValueError, "Expected original channel order"):
            shop.xyz2aet(wrong_xyz_input)

        wrong_aet_input = SpectralDensity(freqs, kernel, ["X", "Y", "Z"])
        with self.assertRaisesRegex(ValueError, "Expected original channel order"):
            shop.aet2xyz(wrong_aet_input)

    def test_array_last_dimension_assertions(self):
        wrong_shape = jnp.ones((4, 2), dtype=jnp.float64)

        with self.assertRaisesRegex(
            ValueError,
            "Expected last dimension of input array to be",
        ):
            shop.xyz2aet(wrong_shape)

        with self.assertRaisesRegex(
            ValueError,
            "Expected last dimension of input array to be",
        ):
            shop.aet2xyz(wrong_shape)


if __name__ == "__main__":
    unittest.main()
