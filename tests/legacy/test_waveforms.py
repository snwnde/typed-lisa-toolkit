# import unittest
# import numpy as np
# from unittest.mock import MagicMock
# from numpy.testing import assert_array_almost_equal
# from typed_lisa_toolkit.containers.waveforms import format, to_fsdata, get_dense_maker
# from typed_lisa_toolkit.containers.representations import FrequencySeries, Interpolator
# from typed_lisa_toolkit.containers.arithdicts import ChnName, ChannelDict, ModeDict
# from typed_lisa_toolkit.containers.data import FSData


# class TestWaveformFunctions(unittest.TestCase):
#     def setUp(self):
#         self.freq_series = FrequencySeries(
#             np.array([1.0, 2.0, 3.0]), np.array([1.0 + 0.0j, 0.5 + 0.5j, 0.2 + 0.2j])
#         )
#         self.waveform_in_channel = {(2, 2): self.freq_series}
#         self.waveform = {ChnName("channel1"): self.waveform_in_channel}

#     def test_format(self):
#         formatted_waveform = format(self.waveform)
#         self.assertIsInstance(formatted_waveform, ChannelDict)
#         self.assertIsInstance(formatted_waveform["channel1"], ModeDict)

#     def test_to_fsdata(self):
#         fsdata = to_fsdata(self.waveform)
#         self.assertIsInstance(fsdata, FSData)
#         expected_entries = self.freq_series.entries
#         assert_array_almost_equal(fsdata["channel1"].entries, expected_entries)


# class TestGetDenseMaker(unittest.TestCase):
#     def setUp(self):
#         # Mock interpolator
#         self.interpolator = MagicMock(spec=Interpolator)

#         # Mock _PhasorT
#         self.mock_phasor = MagicMock()
#         self.mock_phasor.frequencies = np.array([1.0, 2.0, 3.0])
#         self.mock_phasor.get_interpolated.return_value = self.mock_phasor
#         self.mock_phasor.get_embedded.return_value = self.mock_phasor

#         # Frequencies for testing
#         self.frequencies = np.array([0.5, 1.0, 2.0, 3.0, 4.0])

#     def test_make_with_embed_false(self):
#         dense_maker = get_dense_maker(self.interpolator)
#         make_function = dense_maker(self.frequencies, embed=False)

#         result = make_function(self.mock_phasor)

#         # Ensure get_interpolated is called with the correct subset of frequencies
#         self.mock_phasor.get_interpolated.assert_called_once()
#         self.assertEqual(result, self.mock_phasor)
#         self.mock_phasor.get_embedded.assert_not_called()

#     def test_make_with_embed_true(self):
#         dense_maker = get_dense_maker(self.interpolator)
#         make_function = dense_maker(self.frequencies, embed=True)

#         result = make_function(self.mock_phasor)

#         # Ensure get_interpolated is called with the correct subset of frequencies
#         self.mock_phasor.get_interpolated.assert_called_once()
#         # Ensure get_embedded is called
#         self.mock_phasor.get_embedded.assert_called_once_with(self.frequencies)
#         self.assertEqual(result, self.mock_phasor)


# if __name__ == "__main__":
#     unittest.main()
