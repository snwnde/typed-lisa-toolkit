# import unittest
# import numpy as np
# from typed_lisa_toolkit.containers.representations import (
#     TimeSeries,
#     FrequencySeries,
# )
# import tempfile
# import os
# from typed_lisa_toolkit.containers.data import TSData, FSData, TimedFSData


# class TestDataContainers(unittest.TestCase):
#     def setUp(self):
#         self.times = np.linspace(0, 10, 100)
#         self.entries = np.sin(self.times)
#         self.time_series = TimeSeries(grid=self.times, entries=self.entries)
#         self.tsdata = TSData({"channel1": self.time_series})

#         self.frequencies = np.fft.rfftfreq(
#             len(self.times), d=self.times[1] - self.times[0]
#         )
#         self.frequency_entries = np.fft.rfft(self.entries)
#         self.frequency_series = FrequencySeries(
#             grid=self.frequencies, entries=self.frequency_entries
#         )
#         self.fsdata = FSData({"channel1": self.frequency_series})

#         # TODO test data containers with two channels, not just one
#         self.Nf, self.Nt = 10, 10
#         self.wdmdata = self.fsdata.to_WDMdata(Nf=self.Nf, Nt=self.Nt)
#         self.wdm_tgrid = self.wdmdata["channel1"].times
#         self.wdm_fgrid = self.wdmdata["channel1"].frequencies
#         self.wdmdata2 = self.wdmdata.copy()
#         self.wdmdata2["channel2"] = self.frequency_series.to_WDM(Nf=self.Nf, Nt=self.Nt)

#     def test_tsdata_times(self):
#         self.assertTrue(np.array_equal(self.tsdata.times, self.times))

#     def test_tsdata_dt(self):
#         self.assertEqual(self.tsdata.dt, self.times[1] - self.times[0])

#     def test_tsdata_get_frequencies(self):
#         expected_frequencies = np.fft.rfftfreq(
#             len(self.times), d=self.times[1] - self.times[0]
#         )
#         self.assertTrue(
#             np.array_equal(self.tsdata.get_frequencies(), expected_frequencies)
#         )

#     def test_tsdata_get_subset(self):
#         subset = self.tsdata.get_subset(interval=(2, 4))
#         expected_times = self.times[20:40]
#         self.assertTrue(np.array_equal(subset.times, expected_times))

#     def test_tsdata_get_fsdata(self):
#         fsdata = self.tsdata.to_fsdata()
#         self.assertTrue(np.array_equal(fsdata.frequencies, self.frequencies))

#     def test_tsdata_get_zero_padded(self):
#         padded_data = self.tsdata.get_zero_padded(pad_time=(1, 1))
#         expected_length = len(self.times) + 2 * int(np.rint(1 / self.tsdata.dt))
#         self.assertEqual(len(padded_data.times), expected_length)

#     def test_fsdata_frequencies(self):
#         self.assertTrue(np.array_equal(self.fsdata.frequencies, self.frequencies))

#     def test_fsdata_df(self):
#         self.assertEqual(self.fsdata.df, self.frequencies[1] - self.frequencies[0])

#     def test_fsdata_conj(self):
#         conj_data = self.fsdata.conj()
#         expected_entries = np.conj(self.frequency_entries)
#         self.assertTrue(np.array_equal(conj_data["channel1"].entries, expected_entries))

#     def test_fsdata_get_subset(self):
#         subset = self.fsdata.get_subset(interval=(0.1, 0.5))
#         mask = (self.frequencies >= 0.1) & (self.frequencies <= 0.5)
#         expected_frequencies = self.frequencies[mask]
#         self.assertTrue(np.array_equal(subset.frequencies, expected_frequencies))

#     def test_timedfsdata_set_times(self):
#         timed_fsdata = TimedFSData({"channel1": self.frequency_series}, self.times)
#         new_times = np.linspace(0, 20, 200)
#         timed_fsdata.set_times(new_times)
#         self.assertTrue(np.array_equal(timed_fsdata.times, new_times))

#     def test_timedfsdata_drop_times(self):
#         timed_fsdata = TimedFSData({"channel1": self.frequency_series}, self.times)
#         fsdata = timed_fsdata.drop_times()
#         self.assertFalse(hasattr(fsdata, "times"))

#     def test_timedfsdata_get_tsdata(self):
#         timed_fsdata = TimedFSData({"channel1": self.frequency_series}, self.times)
#         tsdata = timed_fsdata.to_tsdata()
#         self.assertTrue(np.array_equal(tsdata.times, self.times))

#     def test_save_and_load_tsdata(self):
#         with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
#             self.tsdata.save(tmpfile.name)
#             loaded_tsdata = TSData.load(tmpfile.name)
#             self.assertTrue(np.array_equal(loaded_tsdata.times, self.tsdata.times))
#             self.assertTrue(
#                 np.array_equal(
#                     loaded_tsdata["channel1"].entries, self.tsdata["channel1"].entries
#                 )
#             )
#         os.remove(tmpfile.name)

#     def test_save_and_load_fsdata(self):
#         with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
#             self.fsdata.save(tmpfile.name)
#             loaded_fsdata = FSData.load(tmpfile.name)
#             self.assertTrue(
#                 np.array_equal(loaded_fsdata.frequencies, self.fsdata.frequencies)
#             )
#             self.assertTrue(
#                 np.array_equal(
#                     loaded_fsdata["channel1"].entries, self.fsdata["channel1"].entries
#                 )
#             )
#         os.remove(tmpfile.name)

#     def test_save_and_load_timedfsdata(self):
#         timed_fsdata = TimedFSData({"channel1": self.frequency_series}, self.times)
#         with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
#             timed_fsdata.save(tmpfile.name)
#             loaded_timedfsdata = TimedFSData.load(tmpfile.name)
#             self.assertTrue(
#                 np.array_equal(loaded_timedfsdata.times, timed_fsdata.times)
#             )
#             self.assertTrue(
#                 np.array_equal(
#                     loaded_timedfsdata["channel1"].entries,
#                     timed_fsdata["channel1"].entries,
#                 )
#             )
#         os.remove(tmpfile.name)

#     def test_wdmdata_conversion(self):
#         # just test that these don't error out, and basic consistency
#         self.wdmdata.to_fsdata()
#         self.wdmdata2.to_fsdata()


# if __name__ == "__main__":
#     unittest.main()
