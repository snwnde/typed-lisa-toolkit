"""Unit tests for tapering.py (JAX backend)."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest

import jax

jax.config.update("jax_enable_x64", val=True)
import jax.numpy as jnp
import numpy as np

from typed_lisa_toolkit.types import tapering


class TestTaperingJax(unittest.TestCase):
    def test_ldc_window_basic(self):
        # kap=1.0 ensures tanh drives ends to ~0 and middle to ~1
        grid = jnp.linspace(0, 100, 100)
        win = tapering.ldc_window(margin=10, kap=1.0)(grid)
        self.assertAlmostEqual(np.asarray(win)[0], 0, places=2)
        self.assertAlmostEqual(np.asarray(win)[-1], 0, places=2)
        self.assertTrue((np.asarray(win)[20:80] > 0.9).all())

    def test_planck_window_basic(self):
        grid = jnp.linspace(0, 100, 100)
        win = tapering.planck_window(left_margin=10, right_margin=10)(grid)
        self.assertEqual(np.asarray(win)[0], 0)
        self.assertEqual(np.asarray(win)[-1], 0)
        self.assertTrue((np.asarray(win)[20:80] > 0.9).all())

    def test_get_tapering_func_str(self):
        grid = jnp.linspace(0, 1, 100)
        taper = tapering.get_tapering_func("hann")
        win = taper(grid)
        self.assertEqual(np.asarray(win).shape, np.asarray(grid).shape)
        self.assertTrue((np.asarray(win) >= 0).all() and (np.asarray(win) <= 1).all())

    def test_get_tapering_func_callable(self):
        grid = jnp.linspace(0, 1, 100)
        taper = tapering.get_tapering_func(np.hanning)
        win = taper(grid)
        self.assertEqual(np.asarray(win).shape, np.asarray(grid).shape)
        self.assertTrue((np.asarray(win) >= 0).all() and (np.asarray(win) <= 1).all())
