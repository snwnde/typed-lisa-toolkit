import unittest
from unittest.mock import MagicMock
from typed_lisa_toolkit.containers.likelihood import WhittleLikelihood
from typed_lisa_toolkit.containers import noisemodel, data


class TestWhittleLikelihood(unittest.TestCase):
    def setUp(self):
        # Mock data and noisemodel
        self.mock_data = MagicMock(spec=data.FSData)
        self.mock_noisemodel = MagicMock(spec=noisemodel.FDNoiseModel)

    def test_log_likelihood_ratio(self):
        cross_product = 5.0
        template_square = 2.0
        expected_result = 4.0  # 5.0 - 0.5 * 2.0
        result = WhittleLikelihood.log_likelihood_ratio(cross_product, template_square)
        self.assertEqual(result, expected_result)

    def test_log_likelihood(self):
        log_likelihood_ratio = 4.0
        data_square = 2.0
        expected_result = 3.0  # 4.0 - 0.5 * 2.0
        result = WhittleLikelihood.log_likelihood(log_likelihood_ratio, data_square)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
