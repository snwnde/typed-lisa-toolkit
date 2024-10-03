import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from typed_lisa_toolkit.containers.waveforms import PhasorSequence, format, to_fsdata
from typed_lisa_toolkit.containers.series import FrequencySeries
from typed_lisa_toolkit.containers.arithdicts import ChnName, ChannelDict, ModeDict
from typed_lisa_toolkit.containers.data import FSData

class TestPhasorSequence(unittest.TestCase):

    def setUp(self):
        self.frequencies = np.array([1.0, 2.0, 3.0])
        self.amplitudes = np.array([1.0, 0.5, 0.2])
        self.phases = np.array([0.0, np.pi/4, np.pi/2])
        self.phasor_sequence = PhasorSequence(self.frequencies, self.amplitudes, self.phases)

    def test_reim_to_cplx(self):
        real_parts = np.array([1.0, 0.0, -1.0])
        imag_parts = np.array([0.0, 1.0, 0.0])
        expected = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        result = PhasorSequence.reim_to_cplx(real_parts, imag_parts)
        assert_array_almost_equal(result, expected)

    def test_cplx_to_phasor(self):
        complex_numbers = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        expected_amplitudes = np.array([1.0, 1.0, 1.0])
        expected_phases = np.array([0.0, np.pi/2, np.pi])
        amplitudes, phases = PhasorSequence.cplx_to_phasor(complex_numbers)
        assert_array_almost_equal(amplitudes, expected_amplitudes)
        assert_array_almost_equal(phases, expected_phases)

    def test_phasor_to_cplx(self):
        amplitudes = np.array([1.0, 1.0, 1.0])
        phases = np.array([0.0, np.pi/2, np.pi])
        expected = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        result = PhasorSequence.phasor_to_cplx(amplitudes, phases)
        assert_array_almost_equal(result, expected)

    def test_reim_to_phasor(self):
        real_parts = np.array([1.0, 0.0, -1.0])
        imag_parts = np.array([0.0, 1.0, 0.0])
        expected_amplitudes = np.array([1.0, 1.0, 1.0])
        expected_phases = np.array([0.0, np.pi/2, np.pi])
        amplitudes, phases = PhasorSequence.reim_to_phasor(real_parts, imag_parts)
        assert_array_almost_equal(amplitudes, expected_amplitudes)
        assert_array_almost_equal(phases, expected_phases)

    def test_get_interpolated(self):
        def dummy_interpolator(x, y):
            return lambda z: np.interp(z, x, y)

        new_frequencies = np.array([1.5, 2.5])
        interpolated_sequence = self.phasor_sequence.get_interpolated(new_frequencies, dummy_interpolator)
        expected_amplitudes = np.array([0.75, 0.35])
        expected_phases = np.array([np.pi/8, 3*np.pi/8])
        assert_array_almost_equal(interpolated_sequence.amplitudes, expected_amplitudes)
        assert_array_almost_equal(interpolated_sequence.phases, expected_phases)

    def test_to_freq_series(self):
        freq_series = self.phasor_sequence.to_freq_series()
        expected_signal = np.array([1.0 + 0.0j, 0.5 * np.exp(1j * np.pi/4), 0.2 * np.exp(1j * np.pi/2)])
        assert_array_almost_equal(freq_series.signal, expected_signal)

    def test_from_freq_series(self):
        signal = np.array([1.0 + 0.0j, 0.5 * np.exp(1j * np.pi/4), 0.2 * np.exp(1j * np.pi/2)])
        freq_series = FrequencySeries(self.frequencies, signal)
        phasor_sequence = PhasorSequence.from_freq_series(freq_series)
        assert_array_almost_equal(phasor_sequence.amplitudes, self.amplitudes)
        assert_array_almost_equal(phasor_sequence.phases, self.phases)


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
        expected_signal = self.freq_series.signal
        assert_array_almost_equal(fsdata['channel1'].signal, expected_signal)


if __name__ == '__main__':
    unittest.main()