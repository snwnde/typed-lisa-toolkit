"""Tests for top-level public API re-exports."""

import unittest

import typed_lisa_toolkit as tlt
from typed_lisa_toolkit.containers.data import TSData
from typed_lisa_toolkit.containers.representations import FrequencySeries
from typed_lisa_toolkit.containers.waveforms import (
    harmonic_projected_waveform,
    harmonic_waveform,
    homogeneous_harmonic_projected_waveform,
    projected_waveform,
    sum_harmonics,
)


class TestPublicApi(unittest.TestCase):
    def test_module_reexports_are_available(self):
        self.assertIsNotNone(tlt.containers)
        self.assertIsNotNone(tlt.consumers)
        self.assertIsNotNone(tlt.viz)
        self.assertIsNotNone(tlt.utils)

    def test_common_symbols_are_reexported(self):
        self.assertIs(tlt.TSData, TSData)
        self.assertIs(tlt.FrequencySeries, FrequencySeries)
        self.assertIs(tlt.sum_harmonics, sum_harmonics)

    def test_submodule_aliases_are_available(self):
        self.assertIsNotNone(tlt.data)
        self.assertIsNotNone(tlt.representations)
        self.assertIsNotNone(tlt.waveforms)

    def test_waveform_aliases_are_reexported(self):
        self.assertIs(tlt.hw, harmonic_waveform)
        self.assertIs(tlt.pw, projected_waveform)
        self.assertIs(tlt.hpw, harmonic_projected_waveform)
        self.assertIs(tlt.hhpw, homogeneous_harmonic_projected_waveform)


if __name__ == "__main__":
    unittest.main()
