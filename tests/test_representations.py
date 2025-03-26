import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from typed_lisa_toolkit.containers.representations import TimeSeries, FrequencySeries, Phasor

class TestSeries(unittest.TestCase):

    def setUp(self):
        self.time_grid = np.linspace(0, 1, 100)
        self.entries = np.sin(2 * np.pi * self.time_grid) + 2
        self.time_series = TimeSeries(grid=self.time_grid, entries=self.entries)

        self.freq_grid = np.fft.rfftfreq(len(self.time_grid), d=self.time_grid[1] - self.time_grid[0])
        self.freq_entries = np.fft.rfft(self.entries * (self.time_grid[1] - self.time_grid[0]))
        self.freq_series = FrequencySeries(grid=self.freq_grid, entries=self.freq_entries)

    def test_is_consistent(self):
        self.assertTrue(self.time_series.is_consistent)
        self.assertTrue(self.freq_series.is_consistent)

        inconsistent_time_series = TimeSeries(grid=self.time_grid[:-1], entries=self.entries)
        self.assertFalse(inconsistent_time_series.is_consistent)

        inconsistent_freq_series = FrequencySeries(grid=self.freq_grid[:-1], entries=self.freq_entries)
        self.assertFalse(inconsistent_freq_series.is_consistent)

    def test_has_even_spacing(self):
        self.assertTrue(self.time_series.has_even_spacing)
        self.assertTrue(self.freq_series.has_even_spacing)

    def test_create_like(self):
        new_entries = self.entries * 2
        new_time_series = self.time_series.create_like(new_entries)
        self.assertTrue(np.array_equal(new_time_series.entries, new_entries))
        self.assertTrue(np.array_equal(new_time_series.grid, self.time_series.grid))

        new_freq_entries = self.freq_entries * 2
        new_freq_series = self.freq_series.create_like(new_freq_entries)
        self.assertTrue(np.array_equal(new_freq_series.entries, new_freq_entries))
        self.assertTrue(np.array_equal(new_freq_series.grid, self.freq_series.grid))

    def test_arithmetic_operations(self):
        result = self.time_series + self.time_series
        self.assertTrue(np.array_equal(result.entries, self.entries + self.entries))

        result = self.time_series - self.time_series
        self.assertTrue(np.array_equal(result.entries, self.entries - self.entries))

        result = self.time_series * 2
        self.assertTrue(np.array_equal(result.entries, self.entries * 2))

        result = self.time_series / 2
        self.assertTrue(np.array_equal(result.entries, self.entries / 2))

        result = 2 * self.time_series
        self.assertTrue(np.array_equal(result.entries, 2 * self.entries))

        result = 2 / self.time_series
        self.assertTrue(np.array_equal(result.entries, 2 / self.entries))

    def test_unary_operations(self):
        result = -self.time_series
        self.assertTrue(np.array_equal(result.entries, -self.entries))

        result = self.time_series.exp()
        self.assertTrue(np.array_equal(result.entries, np.exp(self.entries)))

        result = self.time_series.sqrt()
        self.assertTrue(np.array_equal(result.entries, np.sqrt(self.entries)))

    def test_frequency_series_methods(self):
        result = self.freq_series.conj()
        self.assertTrue(np.array_equal(result.entries, np.conj(self.freq_entries)))

        result = self.freq_series.angle()
        self.assertTrue(np.array_equal(result.entries, np.unwrap(np.angle(self.freq_entries))))

        result = self.freq_series.abs()
        self.assertTrue(np.array_equal(result.entries, np.abs(self.freq_entries)))

        result = self.freq_series.real
        self.assertTrue(np.array_equal(result.entries, np.real(self.freq_entries)))

        result = self.freq_series.imag
        self.assertTrue(np.array_equal(result.entries, np.imag(self.freq_entries)))

    def test_fft_methods(self):
        result = self.time_series.rfft()
        self.assertTrue(np.allclose(result.entries, self.freq_entries))

        result = self.freq_series.irfft(self.time_grid)
        self.assertTrue(np.allclose(result.entries, self.entries))

    def test_time_shift(self):
        shift = 0.1
        result = self.freq_series.get_time_shifted(shift)
        expected_entries = self.freq_entries * np.exp(-2j * np.pi * self.freq_grid * shift)
        self.assertTrue(np.allclose(result.entries, expected_entries))


class TestPhasor(unittest.TestCase):

    def setUp(self):
        self.frequencies = np.array([1.0, 2.0, 3.0])
        self.amplitudes = np.arange(1, 4) * np.exp(1j * np.pi/4)
        self.phases = np.array([0.0, np.pi/4, np.pi/2])
        self.phasor = Phasor.make(self.frequencies, self.amplitudes, self.phases)

    def test_reim_to_cplx(self):
        real_parts = np.array([1.0, 0.0, -1.0])
        imag_parts = np.array([0.0, 1.0, 0.0])
        expected = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        result = Phasor.reim_to_cplx(real_parts, imag_parts)
        assert_array_almost_equal(result, expected)

    def test_cplx_to_reim(self):
        cplx = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        expected_real_parts = np.array([1.0, 0.0, -1.0])
        expected_imag_parts = np.array([0.0, 1.0, 0.0])
        result_real_parts, result_imag_parts = Phasor.cplx_to_reim(cplx)
        assert_array_almost_equal(result_real_parts, expected_real_parts)
        assert_array_almost_equal(result_imag_parts, expected_imag_parts)

    def test_phasor_to_cplx(self):
        amplitudes = np.array([1.0, 1.0, 1.0])
        phases = np.array([0.0, np.pi/2, np.pi])
        expected = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
        result = Phasor.phasor_to_cplx(amplitudes, phases)
        assert_array_almost_equal(result, expected)

    def test_get_interpolated(self):
        def dummy_interpolator(x, y):
            return lambda z: np.interp(z, x, y)

        new_frequencies = np.array([1.5, 2.5])
        interpolated_sequence = self.phasor.get_interpolated(new_frequencies, dummy_interpolator)
        expected_amplitudes = np.array([1.5, 2.5]) * np.exp(1j * np.pi/4)
        expected_phases = np.array([np.pi/8, 3*np.pi/8])
        assert_array_almost_equal(interpolated_sequence.amplitudes, expected_amplitudes)
        assert_array_almost_equal(interpolated_sequence.phases, expected_phases)

    def test_to_freq_series(self):
        freq_series = self.phasor.to_frequency_series()
        expected_entries = np.array([np.exp(1j * np.pi/4), 2 * np.exp(1j * np.pi/2), 3 * np.exp(3j * np.pi/4)])
        assert_array_almost_equal(freq_series.entries, expected_entries)

if __name__ == '__main__':
    unittest.main()