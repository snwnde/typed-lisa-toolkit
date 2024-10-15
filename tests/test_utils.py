import unittest
import numpy as np
from typed_lisa_toolkit.utils import get_subset_slice, get_support_slice, trim_interp


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

    def test_get_support_slice(self):
        # Test case 1: Non-zero array
        array = np.array([0, 0, 1, 2, 0, 0])
        result = get_support_slice(array)
        self.assertEqual(result, slice(2, 4, None))

        # Test case 2: All zeros array
        array = np.array([0, 0, 0, 0, 0])
        result = get_support_slice(array)
        self.assertEqual(result, slice(0, 0, None))

    def test_trim_interp(self):
        # Define a simple interpolator function
        def dummy_interpolator(x, y):
            return lambda z: np.interp(z, x, y)

        # Create a decorated interpolator
        decorated_interpolator = trim_interp(dummy_interpolator)

        # Test case 1: Empty array
        grid = np.array([])
        entries = np.array([])
        interpolated = decorated_interpolator(grid, entries)
        result = interpolated(np.array([1, 2, 3]))
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected)

        # Test case 2: Non-empty array
        grid = np.array([1, 2, 3, 4, 5])
        entries = np.array([1, 2, 3, 4, 5])
        interpolated = decorated_interpolator(grid, entries)
        result = interpolated(np.array([0.5, 1.5, 2.5, 3.5, 4.5]))
        expected = np.array([0, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
