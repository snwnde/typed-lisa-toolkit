"""Unit tests for utils.py (JAX backend)."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typed_lisa_toolkit import utils


class TestUtilsJax(unittest.TestCase):
    def test_get_subset_slice_basic(self):
        arr = jnp.arange(10)
        s = utils.get_subset_slice(arr, 2, 6)
        self.assertEqual(np.asarray(arr[s]).tolist(), [2, 3, 4, 5, 6])

    def test_get_subset_slice_empty(self):
        arr = jnp.arange(10)
        s = utils.get_subset_slice(arr, 20, 30)
        self.assertEqual(np.asarray(arr[s]).tolist(), [])

    def test_get_support_slice(self):
        arr = jnp.array([0, 0, 1, 2, 0, 0])
        s = utils.get_support_slice(arr)
        self.assertEqual(s, slice(2, 4))
        self.assertEqual(np.asarray(arr[s]).tolist(), [1, 2])

    def test_get_support_slice_all_zeros(self):
        arr = jnp.zeros(5)
        s = utils.get_support_slice(arr)
        self.assertEqual(s, slice(0, 0))
        self.assertEqual(np.asarray(arr[s]).tolist(), [])

    def test_extend_to_1d(self):
        grid = jnp.arange(3, 8)
        entries = jnp.array([1, 2, 3, 4, 5]).reshape(1, 1, 1, 1, 5)
        target_grid = jnp.arange(10)
        extended = utils.extend_to(target_grid)(grid, entries)
        self.assertEqual(extended.shape, (1, 1, 1, 1, 10))
        self.assertTrue((np.asarray(extended[..., 3:8]) == np.asarray(entries)).all())
        self.assertTrue(
            (np.asarray(extended[..., :3]) == 0).all()
            and (np.asarray(extended[..., 8:]) == 0).all()
        )

    def test_trim_interp_decorator(self):
        # entries is 1D (support detection); interp must return canonical 5D output
        def interp(x, y):
            return lambda t: jnp.array(  # type: ignore
                np.interp(np.asarray(t), np.asarray(x), np.asarray(y), left=0, right=0)
            ).reshape(1, 1, 1, 1, -1)

        decorated = utils.trim_interp(interp)
        arr = jnp.array([0, 0, 1, 2, 0, 0])  # 1D
        grid = jnp.arange(6)
        f = decorated(grid, arr)
        out = np.asarray(f(jnp.arange(6)))  # shape (1,1,1,1,6)
        self.assertEqual(out.shape, (1, 1, 1, 1, 6))
        self.assertTrue((out[..., :2] == 0).all() and (out[..., 4:] == 0).all())
        self.assertTrue((out[..., 2:4] != 0).all())
