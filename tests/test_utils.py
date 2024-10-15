import unittest
import numpy as np
from typed_lisa_toolkit.utils import get_subset_slice

class TestUtils(unittest.TestCase):

    def test_get_subset_slice(self):
        array = 0.5 * np.arange(10)

        # Test subset within the array range
        result = get_subset_slice(array, 0.9, 3.1)
        self.assertEqual(result, slice(2, 7, None))

        # Test subset outside the array range
        result = get_subset_slice(array, -1.0, 0.0)
        self.assertEqual(result, slice(0, 1, None))

        # Test subset partially within the array range
        result = get_subset_slice(array, 2.6, 7.4)
        self.assertEqual(result, slice(6, 10, None))

if __name__ == '__main__':
    unittest.main()