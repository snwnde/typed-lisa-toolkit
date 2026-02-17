import unittest
from unittest.mock import MagicMock, create_autospec
import numpy as np
from numpy.typing import NDArray
import scipy.integrate  # type: ignore[import]
from typed_lisa_toolkit.containers.data import FSData, TSData, WDMData
from typed_lisa_toolkit.containers.representations import (
    FrequencySeries,
    TimeSeries,
    WDM,
    Linspace,
)

from typed_lisa_toolkit.containers.noisemodel import (
    StationaryFDNoise,
    FDNoiseModel,
    _StationaryNoiseModel,
    _CacheNoiseModel,
    _collect_frequencies,
    TFNoiseModel,
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
        noisemodel = FDNoiseModel.make(fd_noise=self.noise_model)
        self.assertIsInstance(noisemodel, _StationaryNoiseModel)

    def test_cache_sensitivity_creation(self):
        noisemodel = FDNoiseModel.make(noise_cache=self.noise_cache)
        self.assertIsInstance(noisemodel, _CacheNoiseModel)

    def test_invalid_sensitivity_creation(self):
        with self.assertRaises(ValueError):
            FDNoiseModel.make()

    def test_get_noise_psd_noise_model(self):
        self.noise_model.psd.return_value = self.entries
        noisemodel = _StationaryNoiseModel(self.noise_model)
        noise_psd = noisemodel.get_noise_psd(_collect_frequencies(self.data))
        self.assertTrue(np.array_equal(noise_psd["channel1"].entries, self.entries))

    def test_get_noise_psd_cache(self):
        self.noise_cache.get_subset.return_value = self.data
        noisemodel = _CacheNoiseModel(self.noise_cache)
        noise_psd = noisemodel.get_noise_psd(_collect_frequencies(self.data))
        self.assertEqual(noise_psd, self.data)

    def test_get_integrand(self):
        self.noise_model.psd.return_value = self.entries
        noisemodel = _StationaryNoiseModel(self.noise_model)
        integrand = noisemodel.get_integrand(self.data, self.data)
        expected_integrand = (
            4.0 * (1.0 / self.entries) * self.entries * self.entries.conj()
        )
        self.assertTrue(
            np.array_equal(integrand["channel1"].entries, expected_integrand)
        )

    def test_get_complex_scalar_product(self):
        self.noise_model.psd.return_value = self.entries
        noisemodel = _StationaryNoiseModel(self.noise_model)
        product = noisemodel.get_complex_scalar_product(self.data, self.data)
        expected_product = scipy.integrate.trapezoid(
            4.0 * (1.0 / self.entries) * self.entries * self.entries.conj(),
            x=self.frequencies,
        )
        self.assertTrue(np.array_equal(product["channel1"], expected_product))

    def test_get_scalar_product(self):
        self.noise_model.psd.return_value = self.entries
        noisemodel = _StationaryNoiseModel(self.noise_model)
        product = noisemodel.get_scalar_product(self.data, self.data)
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
        noisemodel = _StationaryNoiseModel(self.noise_model)
        cross_correlation = noisemodel.get_cross_correlation(timed_data, another_data)
        self.assertIsInstance(next(iter(cross_correlation.values())), TimeSeries)

    def test_get_whitened(self):
        self.noise_model.psd.return_value = self.entries
        noisemodel = _StationaryNoiseModel(self.noise_model)
        whitened = noisemodel.get_whitened(self.data)
        expected_whitened = self.entries / np.sqrt(self.entries)
        self.assertTrue(np.array_equal(whitened["channel1"].entries, expected_whitened))

    def test_get_overlap(self):
        self.noise_model.psd.return_value = self.entries
        noisemodel = _StationaryNoiseModel(self.noise_model)
        overlap = noisemodel.get_overlap(self.data, self.data)
        self.assertTrue(np.array_equal(overlap["channel1"], 1.0))


class TestTFNoiseModel(unittest.TestCase):
    def test_tfnoisemodel_init_valid(self):
        """Test TFNoiseModel initialization with valid inputs."""
        invevsdm = np.eye(3)[np.newaxis, np.newaxis, :, :]  # shape (1, 1, 3, 3)
        channel_order = ["X", "Y", "Z"]
        model = TFNoiseModel(invevsdm, channel_order=channel_order)
        self.assertTrue(np.array_equal(model.invevsdm, invevsdm))
        self.assertEqual(model.channel_order, channel_order)

    def test_tfnoisemodel_init_invert_sdm(self):
        """Test TFNoiseModel initialization with invert_sdm=True."""
        evsdm = np.array([[[[2.0, 0.0], [0.0, 2.0]]]])  # shape (1, 1, 2, 2)
        channel_order = ["X", "Y"]
        model = TFNoiseModel(evsdm, channel_order=channel_order, invert_sdm=True)
        expected_invevsdm = np.array([[[[0.5, 0.0], [0.0, 0.5]]]])
        self.assertTrue(np.allclose(model.invevsdm, expected_invevsdm))

    def test_tfnoisemodel_is_valid_sdm_valid(self):
        """Test is_valid_sdm returns True for valid input."""
        invevsdm = np.eye(2)[np.newaxis, np.newaxis, :, :]
        with self.assertRaises(ValueError):
            TFNoiseModel.is_valid_sdm(
                invevsdm, channel_order=["X", "Y", "Z"], raise_exception=True
            )
        with self.assertRaises(ValueError):
            TFNoiseModel.is_valid_sdm(
                invevsdm, channel_order=["X", "X"], raise_exception=True
            )

    def test_tfnoisemodel_invert_sdm(self):
        """Test invert_sdm method."""
        sdm = np.array([[[[2.0, 0.0], [0.0, 4.0]]]])
        inverted = TFNoiseModel.invert_sdm(sdm)
        expected = np.array([[[[0.5, 0.0], [0.0, 0.25]]]])
        self.assertTrue(np.allclose(inverted, expected))

    def test_tfnoisemodel_invevsdm(self):
        """Test evsdm method returns the inverse of invevsdm."""
        invevsdm = np.array([[[[4.0, 0.0], [0.0, 2.0]]]])
        model = TFNoiseModel(invevsdm, channel_order=["X", "Y"])
        evsdm = model.evsdm()
        expected = np.array([[[[0.25, 0.0], [0.0, 0.5]]]])
        self.assertTrue(np.allclose(evsdm, expected))

    def test_tfnoisemodel_get_whitening_matrix(self):
        """Test get_whitening_matrix method."""
        invevsdm = np.array([[[[2.0, 0.0], [0.0, 2.0]]]])
        model = TFNoiseModel(invevsdm, channel_order=["X", "Y"])
        W = model.get_whitening_matrix(kind="cholesky")
        # W^T @ W should equal invevsdm
        reconstructed = W.transpose(0, 1, 3, 2) @ W
        self.assertTrue(
            np.allclose(reconstructed, invevsdm),
            f"\nReconstructed:\n{reconstructed} of shape {reconstructed.shape}\nExpected:\n{invevsdm} of shape {invevsdm.shape}",
        )

    def test_inner_product(self):
        """Test inner_product method."""
        invevsdm = np.eye(2)[np.newaxis, np.newaxis, :, :]
        model = TFNoiseModel(invevsdm, channel_order=["X", "Y"])
        # data1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        data1 = {"X": np.array([1.0, 0.0]), "Y": np.array([0.0, 1.0])}
        data2 = {"X": np.array([2.0, 0.0]), "Y": np.array([0.0, 2.0])}
        product = model.get_scalar_product(data1, data2)
        expected = 4.0
        self.assertTrue(np.array_equal(product, expected), msg=f"Expected {expected} but got {product}")


if __name__ == "__main__":
    unittest.main()
