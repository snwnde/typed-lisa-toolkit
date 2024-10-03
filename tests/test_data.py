import unittest
import numpy as np
from typed_lisa_toolkit.containers.series import TimeSeries, FrequencySeries

from typed_lisa_toolkit.containers.data import (
    get_subset_slice,
    TSData,
    FSData,
    TimedFSData,
)

class TestDataContainers(unittest.TestCase):

    def setUp(self):
        self.times = np.linspace(0, 10, 100)
        self.signal = np.sin(self.times)
        self.time_series = TimeSeries(grid=self.times, signal=self.signal)
        self.tsdata = TSData({"channel1": self.time_series})

        self.frequencies = np.fft.rfftfreq(len(self.times), d=self.times[1] - self.times[0])
        self.frequency_signal = np.fft.rfft(self.signal)
        self.frequency_series = FrequencySeries(grid=self.frequencies, signal=self.frequency_signal)
        self.fsdata = FSData({"channel1": self.frequency_series})

    def test_get_subset_slice(self):
        increasing_array = np.array([1, 2, 3, 4, 5])
        min_val, max_val = 2, 4
        result = get_subset_slice(increasing_array, min_val, max_val)
        expected = slice(1, 4)
        self.assertEqual(result, expected)

    def test_tsdata_times(self):
        self.assertTrue(np.array_equal(self.tsdata.times, self.times))

    def test_tsdata_dt(self):
        self.assertEqual(self.tsdata.dt, self.times[1] - self.times[0])

    def test_tsdata_get_frequencies(self):
        expected_frequencies = np.fft.rfftfreq(len(self.times), d=self.times[1] - self.times[0])
        self.assertTrue(np.array_equal(self.tsdata.get_frequencies(), expected_frequencies))

    def test_tsdata_get_subset(self):
        subset = self.tsdata.get_subset(duration=(2, 4))
        expected_times = self.times[20:40]
        self.assertTrue(np.array_equal(subset.times, expected_times))

    def test_tsdata_get_fsdata(self):
        fsdata = self.tsdata.get_fsdata()
        self.assertTrue(np.array_equal(fsdata.frequencies, self.frequencies))

    def test_tsdata_get_zero_padded(self):
        padded_data = self.tsdata.get_zero_padded(pad_time=(1, 1))
        expected_length = len(self.times) + 2 * int(np.rint(1 / self.tsdata.dt))
        self.assertEqual(len(padded_data.times), expected_length)

    def test_fsdata_frequencies(self):
        self.assertTrue(np.array_equal(self.fsdata.frequencies, self.frequencies))

    def test_fsdata_df(self):
        self.assertEqual(self.fsdata.df, self.frequencies[1] - self.frequencies[0])

    def test_fsdata_conj(self):
        conj_data = self.fsdata.conj()
        expected_signal = np.conj(self.frequency_signal)
        self.assertTrue(np.array_equal(conj_data["channel1"].signal, expected_signal))

    def test_fsdata_get_subset(self):
        subset = self.fsdata.get_subset(frequencies=(0.1, 0.5))
        mask = (self.frequencies >= 0.1) & (self.frequencies <= 0.5)
        expected_frequencies = self.frequencies[mask]
        self.assertTrue(np.array_equal(subset.frequencies, expected_frequencies))

    def test_timedfsdata_set_times(self):
        timed_fsdata = TimedFSData({"channel1": self.frequency_series}, self.times)
        new_times = np.linspace(0, 20, 200)
        timed_fsdata.set_times(new_times)
        self.assertTrue(np.array_equal(timed_fsdata.times, new_times))

    def test_timedfsdata_drop_times(self):
        timed_fsdata = TimedFSData({"channel1": self.frequency_series}, self.times)
        fsdata = timed_fsdata.drop_times()
        self.assertFalse(hasattr(fsdata, 'times'))

    def test_timedfsdata_get_tsdata(self):
        timed_fsdata = TimedFSData({"channel1": self.frequency_series}, self.times)
        tsdata = timed_fsdata.get_tsdata()
        self.assertTrue(np.array_equal(tsdata.times, self.times))

if __name__ == '__main__':
    unittest.main()