# pyright: reportPrivateUsage=false

import unittest

from typed_lisa_toolkit.types.modes import QNM, Harmonic, QuasiNormalMode, cast_mode


class TestModes(unittest.TestCase):
    def test_harmonic_properties_and_cast(self):
        mode = Harmonic.cast((2, 1))
        self.assertEqual(mode.degree, 2)
        self.assertEqual(mode.order, 1)

    def test_qnm_properties_and_cast_alias(self):
        mode = QNM.cast((3, 2, 0))
        self.assertIsInstance(mode, QuasiNormalMode)
        self.assertEqual(mode.degree, 3)
        self.assertEqual(mode.order, 2)
        self.assertEqual(mode.overtone, 0)

    def test_cast_mode_dispatch_and_error(self):
        self.assertEqual(cast_mode((2, 2)), Harmonic(2, 2))
        self.assertEqual(cast_mode((2, 2, 1)), QNM(2, 2, 1))
        with self.assertRaises(ValueError):
            cast_mode((2,))
