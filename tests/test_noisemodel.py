import unittest
from unittest.mock import MagicMock, create_autospec
import numpy as np
import scipy.integrate # type: ignore[import]
from typed_lisa_toolkit.containers.data import FSData, TSData
from typed_lisa_toolkit.containers.representations import FrequencySeries, TimeSeries

from typed_lisa_toolkit.containers.noisemodel import (
    StationaryFDNoise,
    FDNoiseModel,
    _StationaryNoiseModel,
    _CacheNoiseModel,
    _collect_frequencies,
)


class TestFDSensitivity(unittest.TestCase):
    def setUp(self):
        self.noise_model = create_autospec(StationaryFDNoise, instance=True)
        self.noise_cache = MagicMock(spec=FSData)
        self.frequencies = np.array([0.1, 0.2, 0.3])
        self.entries = np.array([1.0, 2.0, 3.0])
        self.data = FSData(
            {"channel1": FrequencySeries(self.frequencies, self.entries)}
        )

    def test_noise_model_sensitivity_creation(self):
        sensitivity = FDNoiseModel.make(fd_noise=self.noise_model)
        self.assertIsInstance(sensitivity, _StationaryNoiseModel)

    def test_cache_sensitivity_creation(self):
        sensitivity = FDNoiseModel.make(noise_cache=self.noise_cache)
        self.assertIsInstance(sensitivity, _CacheNoiseModel)

    def test_invalid_sensitivity_creation(self):
        with self.assertRaises(ValueError):
            FDNoiseModel.make()

    def test_get_noise_psd_noise_model(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        noise_psd = sensitivity.get_noise_psd(_collect_frequencies(self.data))
        self.assertTrue(np.array_equal(noise_psd["channel1"].entries, self.entries))

    def test_get_noise_psd_cache(self):
        self.noise_cache.get_subset.return_value = self.data
        sensitivity = _CacheNoiseModel(self.noise_cache)
        noise_psd = sensitivity.get_noise_psd(_collect_frequencies(self.data))
        self.assertEqual(noise_psd, self.data)

    def test_get_integrand(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        integrand = sensitivity.get_integrand(self.data, self.data)
        expected_integrand = (
            4.0 * (1.0 / self.entries) * self.entries * self.entries.conj()
        )
        self.assertTrue(
            np.array_equal(integrand["channel1"].entries, expected_integrand)
        )

    def test_get_complex_scalar_product(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        product = sensitivity.get_complex_scalar_product(self.data, self.data)
        expected_product = scipy.integrate.trapezoid(
            4.0 * (1.0 / self.entries) * self.entries * self.entries.conj(),
            x=self.frequencies,
        )
        self.assertTrue(np.array_equal(product["channel1"], expected_product))

    def test_get_scalar_product(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        product = sensitivity.get_scalar_product(self.data, self.data)
        expected_product = np.real(
            scipy.integrate.trapezoid(
                4.0 * (1.0 / self.entries) * self.entries * self.entries.conj(),
                x=self.frequencies,
            )
        )
        self.assertTrue(np.array_equal(product["channel1"], expected_product))

    def test_get_cross_correlation(self):
        times = np.array([0.0, 1.0, 2.0])
        tsdata = TSData({"channel1": TimeSeries(times, np.sin(times))})
        timed_data = tsdata.to_fsdata()
        another_data = FSData(
            {
                "channel1": FrequencySeries(
                    timed_data.frequencies, np.cos(timed_data.frequencies)
                )
            }
        )
        self.noise_model.psd.return_value = np.ones_like(timed_data.frequencies)
        sensitivity = _StationaryNoiseModel(self.noise_model)
        cross_correlation = sensitivity.get_cross_correlation(timed_data, another_data)
        self.assertIsInstance(next(iter(cross_correlation.values())), TimeSeries)

    def test_get_whitened(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        whitened = sensitivity.get_whitened(self.data)
        expected_whitened = self.entries / np.sqrt(self.entries)
        self.assertTrue(np.array_equal(whitened["channel1"].entries, expected_whitened))

    def test_get_overlap(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        overlap = sensitivity.get_overlap(self.data, self.data)
        self.assertTrue(np.array_equal(overlap["channel1"], 1.0))


if __name__ == "__main__":
    unittest.main()
