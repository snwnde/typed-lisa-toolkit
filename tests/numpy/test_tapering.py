"""Unit tests for tapering.py (NumPy backend)."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest

import numpy as np
import pytest

from typed_lisa_toolkit.types import tapering


class TestTaperingNumpy(unittest.TestCase):
    def test_ldc_window_basic(self):
        # kap=1.0 ensures tanh drives ends to ~0 and middle to ~1
        grid = np.linspace(0, 100, 100)
        win = tapering.ldc_window(margin=10, kap=1.0)(grid)
        assert win[0] == pytest.approx(0, abs=1e-2)
        assert win[-1] == pytest.approx(0, abs=1e-2)
        assert (win[20:80] > 0.9).all()

    def test_planck_window_basic(self):
        grid = np.linspace(0, 100, 100)
        win = tapering.planck_window(left_margin=10, right_margin=10)(grid)
        assert win[0] == 0
        assert win[-1] == 0
        assert (win[20:80] > 0.9).all()

    def test_get_tapering_func_str(self):
        grid = np.linspace(0, 1, 100)
        taper = tapering.get_tapering_func("hann")
        win = taper(grid)
        assert win.shape == grid.shape
        assert (win >= 0).all()
        assert (win <= 1).all()

    def test_get_tapering_func_callable(self):
        grid = np.linspace(0, 1, 100)
        taper = tapering.get_tapering_func(np.hanning)
        win = taper(grid)
        assert win.shape == grid.shape
        assert (win >= 0).all()
        assert (win <= 1).all()
