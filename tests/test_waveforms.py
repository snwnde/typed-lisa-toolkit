import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from typed_lisa_toolkit.containers.waveforms import format, to_fsdata
from typed_lisa_toolkit.containers.representations import FrequencySeries
from typed_lisa_toolkit.containers.arithdicts import ChnName, ChannelDict, ModeDict
from typed_lisa_toolkit.containers.data import FSData


class TestWaveformFunctions(unittest.TestCase):

    def setUp(self):
        self.freq_series = FrequencySeries(np.array([1.0, 2.0, 3.0]), np.array([1.0 + 0.0j, 0.5 + 0.5j, 0.2 + 0.2j]))
        self.waveform_in_channel = {(2,2): self.freq_series}
        self.waveform = {ChnName('channel1'): self.waveform_in_channel}

    def test_format(self):
        formatted_waveform = format(self.waveform)
        self.assertIsInstance(formatted_waveform, ChannelDict)
        self.assertIsInstance(formatted_waveform['channel1'], ModeDict)

    def test_to_fsdata(self):
        fsdata = to_fsdata(self.waveform)
        self.assertIsInstance(fsdata, FSData)
        expected_entries = self.freq_series.entries
        assert_array_almost_equal(fsdata['channel1'].entries, expected_entries)


if __name__ == '__main__':
    unittest.main()