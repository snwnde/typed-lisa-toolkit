import unittest
import numpy as np
from typed_lisa_toolkit.containers.series import TimeSeries, FrequencySeries

class TestSeries(unittest.TestCase):

    def setUp(self):
        self.time_grid = np.linspace(0, 1, 100)
        self.signal = np.sin(2 * np.pi * self.time_grid) + 2
        self.time_series = TimeSeries(grid=self.time_grid, signal=self.signal)

        self.freq_grid = np.fft.rfftfreq(len(self.time_grid), d=self.time_grid[1] - self.time_grid[0])
        self.freq_signal = np.fft.rfft(self.signal)
        self.freq_series = FrequencySeries(grid=self.freq_grid, signal=self.freq_signal)

    def test_is_consistent(self):
        self.assertTrue(self.time_series.is_consistent)
        self.assertTrue(self.freq_series.is_consistent)

        inconsistent_time_series = TimeSeries(grid=self.time_grid[:-1], signal=self.signal)
        self.assertFalse(inconsistent_time_series.is_consistent)

        inconsistent_freq_series = FrequencySeries(grid=self.freq_grid[:-1], signal=self.freq_signal)
        self.assertFalse(inconsistent_freq_series.is_consistent)

    def test_has_even_spacing(self):
        self.assertTrue(self.time_series.has_even_spacing)
        self.assertTrue(self.freq_series.has_even_spacing)

    def test_create_like(self):
        new_signal = self.signal * 2
        new_time_series = self.time_series.create_like(new_signal)
        self.assertTrue(np.array_equal(new_time_series.signal, new_signal))
        self.assertTrue(np.array_equal(new_time_series.grid, self.time_series.grid))

        new_freq_signal = self.freq_signal * 2
        new_freq_series = self.freq_series.create_like(new_freq_signal)
        self.assertTrue(np.array_equal(new_freq_series.signal, new_freq_signal))
        self.assertTrue(np.array_equal(new_freq_series.grid, self.freq_series.grid))

    def test_arithmetic_operations(self):
        result = self.time_series + self.time_series
        self.assertTrue(np.array_equal(result.signal, self.signal + self.signal))

        result = self.time_series - self.time_series
        self.assertTrue(np.array_equal(result.signal, self.signal - self.signal))

        result = self.time_series * 2
        self.assertTrue(np.array_equal(result.signal, self.signal * 2))

        result = self.time_series / 2
        self.assertTrue(np.array_equal(result.signal, self.signal / 2))

        result = 2 * self.time_series
        self.assertTrue(np.array_equal(result.signal, 2 * self.signal))

        result = 2 / self.time_series
        self.assertTrue(np.array_equal(result.signal, 2 / self.signal))

    def test_unary_operations(self):
        result = -self.time_series
        self.assertTrue(np.array_equal(result.signal, -self.signal))

        result = self.time_series.exp()
        self.assertTrue(np.array_equal(result.signal, np.exp(self.signal)))

        result = self.time_series.sqrt()
        self.assertTrue(np.array_equal(result.signal, np.sqrt(self.signal)))

    def test_frequency_series_methods(self):
        result = self.freq_series.conj()
        self.assertTrue(np.array_equal(result.signal, np.conj(self.freq_signal)))

        result = self.freq_series.angle()
        self.assertTrue(np.array_equal(result.signal, np.angle(self.freq_signal)))

        result = self.freq_series.abs()
        self.assertTrue(np.array_equal(result.signal, np.abs(self.freq_signal)))

        result = self.freq_series.real()
        self.assertTrue(np.array_equal(result.signal, np.real(self.freq_signal)))

        result = self.freq_series.imag()
        self.assertTrue(np.array_equal(result.signal, np.imag(self.freq_signal)))

    def test_fft_methods(self):
        result = self.time_series.rfft()
        self.assertTrue(np.allclose(result.signal, self.freq_signal))

        result = self.freq_series.irfft(self.time_grid)
        self.assertTrue(np.allclose(result.signal, self.signal))

    def test_time_shift(self):
        shift = 0.1
        result = self.freq_series.get_time_shifted(shift)
        expected_signal = self.freq_signal * np.exp(-2j * np.pi * self.freq_grid * shift)
        self.assertTrue(np.allclose(result.signal, expected_signal))



if __name__ == '__main__':
    unittest.main()