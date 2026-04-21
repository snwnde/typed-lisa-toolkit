# pyright: reportPrivateUsage=false

import unittest

import pytest

from typed_lisa_toolkit.types.modes import QNM, Harmonic, QuasiNormalMode, cast_mode


class TestModes(unittest.TestCase):
    def test_harmonic_properties_and_cast(self):
        mode = Harmonic.cast((2, 1))
        assert mode.degree == 2
        assert mode.order == 1

    def test_qnm_properties_and_cast_alias(self):
        mode = QNM.cast((3, 2, 0))
        assert isinstance(mode, QuasiNormalMode)
        assert mode.degree == 3
        assert mode.order == 2
        assert mode.overtone == 0

    def test_cast_mode_dispatch_and_error(self):
        assert cast_mode((2, 2)) == Harmonic(2, 2)
        assert cast_mode((2, 2, 1)) == QNM(2, 2, 1)
        with pytest.raises(ValueError, match=r".+"):
            cast_mode((2,))
