import unittest
from unittest.mock import MagicMock
import numpy as np
from typed_lisa_toolkit.containers.likelihood import WhittleLikelihood
from typed_lisa_toolkit.containers import (
    arithdicts,
    waveforms as wf_,
    sensitivity as sens_,
    data as data_,
    modes,
)


class TestWhittleLikelihood(unittest.TestCase):
    def setUp(self):
        # Mock data and sensitivity objects
        self.mock_data = MagicMock(spec=data_.FSData)
        self.mock_sensitivity = MagicMock(spec=sens_.FDSensitivity)

        # Mock methods for data and sensitivity
        self.mock_data.frequencies = np.array([1.0, 2.0, 3.0])
        self.mock_sensitivity.get_cache.return_value = self.mock_sensitivity
        self.mock_sensitivity.get_noise_psd.return_value = np.array([1.0, 1.0, 1.0])
        self.mock_sensitivity.get_scalar_product.return_value = 1.0

        # Create instance of WhittleLikelihood
        self.whittle_likelihood = WhittleLikelihood(
            self.mock_data, self.mock_sensitivity
        )

        # Mock template
        self.mock_template = MagicMock(spec=wf_.FSWaveformGen)
        self.mock_template.get_frequencies.return_value = arithdicts.ModeDict(
            {modes.Harmonic(2, 2): np.array([1.0, 2.0, 3.0])}
        )

    def test_get_log_likelihood_ratio(self):
        # Test get_log_likelihood_ratio method
        result = self.whittle_likelihood.get_log_likelihood_ratio(self.mock_template)
        self.assertEqual(result, 0.5)

    def test_get_log_likelihood(self):
        # Test get_log_likelihood method
        result = self.whittle_likelihood.get_log_likelihood(self.mock_template)
        self.assertEqual(result, 0.0)

    def test_get_cross_product(self):
        # Test get_cross_product method
        result = self.whittle_likelihood.get_cross_product(self.mock_template)
        self.assertEqual(result, 1.0)

    def test_get_template_square(self):
        # Test get_template_square method
        result = self.whittle_likelihood.get_template_square(self.mock_template)
        self.assertEqual(result, 1.0)

    def test_process(self):
        # Test _process method
        template_waveform, f_interval = self.whittle_likelihood._process(
            self.mock_template
        )
        self.assertEqual(f_interval, (1.0, 3.0))
        self.assertTrue(isinstance(template_waveform, arithdicts.ChannelDict))


if __name__ == "__main__":
    unittest.main()
