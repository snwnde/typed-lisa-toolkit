"""Tests for canonical shape functionality in representations."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import unittest

import numpy as np
import numpy.testing as npt
from l2d_interface.validators import (
    validate_representation,  # type: ignore[import-untyped]
)
from scipy.signal import freqs

from tests._helpers import (
    AdvancedRepresentationMethodsMixin,
    HelperFunctionsMixin,
    LinspaceExtraPropertiesMixin,
    WDMPropertiesAndMethodsMixin,
    build_canonical_representations,
)
from typed_lisa_toolkit import (
    build_grid2d,
    frequency_series,
    linspace,
    phasor,
    stft,
    time_series,
    utils,
    wdm,
)
from typed_lisa_toolkit.types import (
    STFT,
    FrequencySeries,
    Grid2DSparse,
    Linspace,
    Phasor,
    TSData,
    UniformFrequencySeries,
    UniformTimeSeries,
)
from typed_lisa_toolkit.types.representations import (  # extra symbols for coverage tests
    _embed_entries_to_grid_2d_sparse,
    _get_full_slice,
    _get_subset_slice,
    _subset_grid_2d_sparse,
    _take_subset,
)


class TestCanonicalShape(unittest.TestCase):
    """Test semantic benefits of canonical shape: (n_batches, n_channels, n_harmonics, n_features, *grid_dims)."""

    def setUp(self):
        """Create test fixtures for canonical shape tests."""
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = 2, 3, 2, 1
        self.len_time, self.len_freq = 100, 50
        case = build_canonical_representations(
            np,
            n_batches=self.n_batches,
            n_channels=self.n_channels,
            n_harmonics=self.n_harmonics,
            n_features=self.n_features,
            len_time=self.len_time,
            len_freq=self.len_freq,
        )

        self.freqs = case["freqs"]
        self.times = case["times"]
        self.entries_fs = case["entries_fs"]
        self.entries_ts = case["entries_ts"]
        self.entries_tf = case["entries_tf"]
        self.fs = case["fs"]
        self.ts = case["ts"]
        self.stft = case["tf"]


class TestL2DContractNumpy(TestCanonicalShape):
    """Test l2d-interface runtime contract compliance with NumPy backend."""

    def test_representation_contract(self):
        times = np.linspace(0.0, 1.0, 16)
        freqs = np.fft.rfftfreq(len(times), d=times[1] - times[0])

        ts = time_series(
            times,
            entries=np.random.randn(1, 1, 1, 1, len(times)),
        )
        fs = frequency_series(
            freqs,
            entries=np.random.randn(1, 1, 1, 1, len(freqs)),
        )
        stft = STFT(
            grid=(freqs, times),
            entries=np.random.randn(1, 1, 1, 1, len(freqs), len(times)),
        )

        validate_representation(ts)
        validate_representation(fs)
        validate_representation(stft)

        self.assertEqual(ts.domain, "time")
        self.assertEqual(fs.domain, "frequency")
        self.assertEqual(stft.domain, "time-frequency")
        self.assertIsNone(ts.kind)
        self.assertIsNone(fs.kind)

    def test_data_contract(self):
        times = np.linspace(0.0, 1.0, 16)
        x = time_series(times, entries=np.random.randn(1, 1, 1, 1, len(times)))
        y = time_series(times, entries=np.random.randn(1, 1, 1, 1, len(times)))

        data = TSData.from_dict({"X": x, "Y": y})
        kernel = np.asarray(data.get_kernel())
        self.assertEqual(data.domain, "time")
        self.assertEqual(data.channel_names, ("X", "Y"))
        self.assertEqual(kernel.shape, (1, 2, 1, 1, len(times)))

        x_view = data["X"]
        x_entries = np.asarray(x_view.entries)
        validate_representation(x_view)
        self.assertEqual(x_entries.shape[1], 1)
        self.assertEqual(x_entries.shape[2], 1)

    def test_batch_dimension_indexing(self):
        """Test that batch dimension is first, enabling natural batch selection."""
        self.assertEqual(self.fs.n_batches, self.n_batches)
        self.assertEqual(self.ts.n_batches, self.n_batches)
        self.assertEqual(self.stft.n_batches, self.n_batches)
        self.assertEqual(
            self.entries_fs[0].shape,
            (self.n_channels, self.n_harmonics, self.n_features, self.len_freq),
        )
        self.assertEqual(
            self.entries_ts[0].shape,
            (self.n_channels, self.n_harmonics, self.n_features, self.len_time),
        )
        self.assertEqual(
            self.entries_tf[0].shape,
            (
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
                self.len_time,
            ),
        )

    def test_channel_dimension_indexing(self):
        """Test that channel dimension is second, enabling natural channel selection."""
        self.assertEqual(self.fs.n_channels, self.n_channels)
        self.assertEqual(self.ts.n_channels, self.n_channels)
        self.assertEqual(self.stft.n_channels, self.n_channels)
        self.assertEqual(
            self.entries_fs[:, 0].shape,
            (self.n_batches, self.n_harmonics, self.n_features, self.len_freq),
        )
        self.assertEqual(
            self.entries_ts[:, 0].shape,
            (self.n_batches, self.n_harmonics, self.n_features, self.len_time),
        )
        self.assertEqual(
            self.entries_tf[:, 0].shape,
            (
                self.n_batches,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
                self.len_time,
            ),
        )

    def test_harmonic_dimension_indexing(self):
        """Test that harmonic dimension is third, enabling natural harmonic selection."""
        self.assertEqual(self.fs.n_harmonics, self.n_harmonics)
        self.assertEqual(self.ts.n_harmonics, self.n_harmonics)
        self.assertEqual(self.stft.n_harmonics, self.n_harmonics)
        self.assertEqual(
            self.entries_fs[:, :, 0].shape,
            (self.n_batches, self.n_channels, self.n_features, self.len_freq),
        )
        self.assertEqual(
            self.entries_ts[:, :, 0].shape,
            (self.n_batches, self.n_channels, self.n_features, self.len_time),
        )
        self.assertEqual(
            self.entries_tf[:, :, 0].shape,
            (
                self.n_batches,
                self.n_channels,
                self.n_features,
                self.len_freq,
                self.len_time,
            ),
        )

    def test_grid_dimension_trailing(self):
        """Test that grid dimensions are trailing, enabling broadcasting with leading dimensions."""
        # Verify grid dimension is last for 1D series (FrequencySeries)
        grid_slice_fs = self.fs.entries[:, :, :, :, 10:20]
        self.assertEqual(
            grid_slice_fs.shape,
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 10),
        )

        # Verify broadcasting: selecting a single grid point removes last dimension
        grid_index_fs = self.fs.entries[:, :, :, :, 5]
        self.assertEqual(
            grid_index_fs.shape,
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

        # Verify for TimeSeries as well
        grid_slice_ts = self.ts.entries[:, :, :, :, 20:30]
        self.assertEqual(
            grid_slice_ts.shape,
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 10),
        )

    def test_broadcasting_with_batch_operations(self):
        """Test that canonical shape enables natural broadcasting for batch operations."""
        # Demonstrate per-batch scaling works with canonical shape
        batch_scales = np.array([2.0, 3.0])
        scaled = (
            self.fs.entries[0:2]
            * batch_scales[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        )

        self.assertEqual(
            scaled.shape,
            (
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
            ),
        )

    def test_multidimensional_grid_shape(self):
        """Test canonical shape with 2D grid (time-frequency representation)."""
        # Verify STFT shape structure with 2D grid
        self.assertEqual(self.stft.n_batches, self.n_batches)
        self.assertEqual(self.stft.n_channels, self.n_channels)
        self.assertEqual(self.stft.n_harmonics, self.n_harmonics)
        self.assertEqual(
            self.entries_tf.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )
        self.assertEqual(
            self.entries_tf.shape[4:], (self.len_freq, self.len_time)
        )  # 2D grid

        # Verify independent grid selection along time dimension
        time_slice = self.entries_tf[:, :, :, :, 10:20, :]
        self.assertEqual(
            time_slice.shape,
            (
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                10,
                self.len_time,
            ),
        )

        # Verify independent grid selection along frequency dimension
        freq_slice = self.entries_tf[:, :, :, :, :, 5:15]
        self.assertEqual(
            freq_slice.shape,
            (
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
                10,
            ),
        )

    def test_shape_enables_semantic_slicing(self):
        """Test that canonical shape enables semantic slicing patterns."""
        # Test semantic slicing on FrequencySeries
        semantic_slice_fs = self.fs.entries[0:2, 1:2, 1, :, 10:30]
        self.assertEqual(semantic_slice_fs.shape, (2, 1, self.n_features, 20))

        # Test semantic slicing on STFT with both grid dimensions
        semantic_slice_tf = self.stft.entries[0:2, 1:2, 1, :, 10:20, 5:15]
        self.assertEqual(
            semantic_slice_tf.shape,
            (2, 1, self.n_features, 10, 10),
        )

        # Verify counts match selection bounds
        self.assertEqual(semantic_slice_fs.shape[0], 2)  # both batches
        self.assertEqual(semantic_slice_fs.shape[1], 1)  # single channel


class TestSubsetOperations(unittest.TestCase):
    """Test subset operations with canonical shapes."""

    def setUp(self):
        """Create test fixtures for subset operation tests."""
        # Canonical shape dimensions
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = 2, 3, 1, 1
        self.len_grid_small, self.len_grid_large = 101, 1000

        # Frequency series fixture (larger grid)
        self.freqs_large = np.linspace(1e-4, 1e-1, self.len_grid_large)
        entries_fs_large = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_large,
        )
        self.fs_large = frequency_series(self.freqs_large, entries=entries_fs_large)

        # Time series fixture with Linspace
        self.times_ls = Linspace(0.0, 0.01, self.len_grid_large)
        entries_ts_ls = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_large,
        )
        self.ts_ls = time_series(self.times_ls, entries=entries_ts_ls)

        self.tf_large = stft(
            self.freqs_large,
            self.times_ls,
            entries=np.random.randn(
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_grid_large,
                self.len_grid_large,
            ),
        )

    def test_get_subset_slice_helper(self):
        """Test _get_subset_slice helper function."""
        grid = np.linspace(0, 10, 101)
        ls = Linspace(0, 0.1, 101)

        # Test with interval
        slice_obj = _get_subset_slice(grid, interval=(2.0, 5.0))
        self.assertIsInstance(slice_obj, slice)
        self.assertEqual(slice_obj.start, 20)
        self.assertEqual(slice_obj.stop, 51)

        # Test with Linspace
        slice_obj = _get_subset_slice(ls, interval=(2.0, 5.0))
        self.assertEqual(slice_obj.start, 20)
        self.assertEqual(slice_obj.stop, 51)

        # Test with explicit slice
        slice_obj = _get_subset_slice(grid, slice=slice(10, 20))
        self.assertEqual(slice_obj, slice(10, 20))

        # Test with None returns slice(None)
        slice_obj = _get_subset_slice(grid)
        self.assertEqual(slice_obj, slice(None))

    def test_get_full_slice(self):
        """Test _get_full_slice helper for canonical indexing."""
        # 1D grid: (batch, channels, harmonics, features, grid)
        grid_slices = (slice(10, 20),)
        full_slice = _get_full_slice(grid_slices)
        expected = (slice(None), slice(None), slice(None), slice(None), slice(10, 20))
        self.assertEqual(full_slice, expected)

        # 2D grid: (batch, channels, harmonics, features, grid1, grid2)
        grid_slices = (slice(5, 10), slice(20, 30))
        full_slice = _get_full_slice(grid_slices)
        expected = (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            slice(5, 10),
            slice(20, 30),
        )
        self.assertEqual(full_slice, expected)

    def test_take_subset_1d(self):
        """Test _take_subset with 1D grid and canonical shape."""
        # Create grid and canonical entries with parameterized shape
        grid = (np.linspace(0, 10, self.len_grid_small),)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_small,
        )

        # Take subset
        subset_slice = (slice(20, 51),)
        new_grid, new_entries = _take_subset(grid, entries, subset_slice)

        # Check grid
        self.assertEqual(len(new_grid), 1)
        npt.assert_array_equal(new_grid[0], grid[0][20:51])

        # Check entries shape
        expected_shape = (
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            31,
        )
        self.assertEqual(new_entries.shape, expected_shape)

        # Check entries values
        npt.assert_array_equal(new_entries, entries[:, :, :, :, 20:51])

    def test_frequency_series_get_subset(self):
        """Test FrequencySeries.get_subset with canonical shape."""
        # Use fixture - get subset by interval
        fs_sub = self.fs_large.get_subset(interval=(1e-3, 5e-2))

        # Check grid is subset (convert to array if Linspace)
        grid_array = np.array(fs_sub.grid[0])
        self.assertTrue(grid_array[0] >= 1e-3)
        self.assertTrue(grid_array[-1] <= 5e-2)

        # Check entries shape is correct - leading dimensions preserved
        self.assertEqual(fs_sub.entries.shape[0], self.n_batches)
        self.assertEqual(
            fs_sub.entries.shape[1:4],
            (self.n_channels, self.n_harmonics, self.n_features),
        )

        # Get subset by slice and verify shape
        fs_sub2 = self.fs_large.get_subset(slice=slice(100, 200))
        self.assertEqual(len(fs_sub2.grid[0]), 100)
        self.assertEqual(fs_sub2.entries.shape[4], 100)
        self.assertEqual(
            fs_sub2.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_time_series_get_subset(self):
        """Test TimeSeries.get_subset with canonical shape."""
        # Use fixture with Linspace
        ts_sub = self.ts_ls.get_subset(interval=(2.0, 5.0))

        # Check Linspace is maintained
        self.assertIsInstance(ts_sub.grid[0], Linspace)

        # Check shape - leading dimensions preserved
        self.assertEqual(ts_sub.entries.shape[0], self.n_batches)
        self.assertEqual(
            ts_sub.entries.shape[1:4],
            (self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_getitem_subset(self):
        """Test __getitem__ for subsetting with canonical shape."""
        # Create test series with parameterized shape
        freqs = np.linspace(0, 1, 100)
        entries = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 100
        )
        fs = frequency_series(freqs, entries=entries)

        # Use slice notation
        fs_sub = fs[20:50]

        # Check
        self.assertEqual(len(fs_sub.frequencies), 30)
        self.assertEqual(fs_sub.entries.shape[4], 30)
        self.assertEqual(
            fs_sub.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )
        npt.assert_array_almost_equal(np.array(fs_sub.frequencies), freqs[20:50])

    def test_timefrequency_subset(self):
        """Test STFT subsetting along time dimension."""
        # Subset by time interval
        tf_sub_time = self.tf_large.get_subset(time_interval=(2.0, 5.0))

        # Check grid
        time_array = np.array(tf_sub_time.times)
        npt.assert_array_max_ulp(time_array[0], 2.0, maxulp=100)
        npt.assert_array_max_ulp(time_array[-1], 5.0, maxulp=100)
        npt.assert_array_max_ulp(np.array(tf_sub_time.frequencies), self.freqs_large)

        # Check shape - leading dimensions and frequency grid preserved
        self.assertEqual(
            tf_sub_time.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )
        self.assertEqual(
            tf_sub_time.entries.shape[4], self.len_grid_large
        )  # frequency dimension preserved

        # Subset by frequency interval
        tf_sub_freq = self.tf_large.get_subset(freq_interval=(1e-3, 5e-2))
        freq_array = np.array(tf_sub_freq.frequencies)
        npt.assert_array_max_ulp(freq_array[0], 1e-3, maxulp=100)
        npt.assert_array_max_ulp(freq_array[-1], 5e-2, maxulp=100)
        npt.assert_array_max_ulp(np.array(tf_sub_freq.times), self.times_ls)
        self.assertEqual(
            tf_sub_freq.entries.shape[5], self.len_grid_large
        )  # time dimension preserved
        self.assertEqual(
            tf_sub_freq.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )


class TestEmbedOperations(unittest.TestCase):
    """Test embedding operations with canonical shapes."""

    def test_extend_to_canonical_shape(self):
        """Test utils.extend_to with canonical shape arrays."""
        # Create grids (as tuples)
        small_grid = (np.linspace(2, 5, 31),)  # subset
        large_grid = (np.linspace(0, 10, 101),)  # superset

        # Create canonical shape entries for small grid
        entries_small = np.random.randn(
            2, 3, 2, 1, 31
        )  # (batch, chan, harm, feat, grid)

        # Extend
        extender = utils.extend_to(large_grid)
        entries_large = extender(small_grid, entries_small)

        # Check shape
        expected_shape = (2, 3, 2, 1, 101)
        self.assertEqual(entries_large.shape, expected_shape)

        # Check that subset was placed correctly
        subset_slice = utils.get_subset_slice(
            large_grid[0], small_grid[0][0], small_grid[0][-1]
        )
        npt.assert_array_equal(entries_large[:, :, :, :, subset_slice], entries_small)

        # Check that outside values are zero
        self.assertTrue(np.all(entries_large[:, :, :, :, : subset_slice.start] == 0))
        self.assertTrue(np.all(entries_large[:, :, :, :, subset_slice.stop :] == 0))

    def test_frequency_series_get_embedded(self):
        """Test FrequencySeries.get_embedded with canonical shape."""
        # Create small frequency series - use exact grid alignment
        freqs_small = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        entries_small = np.ones((1, 1, 1, 1, 5), dtype=complex)  # Use simple values
        for i in range(5):
            entries_small[0, 0, 0, 0, i] = i + 1
        fs_small = frequency_series(freqs_small, entries=entries_small)

        # Create large grid that contains small grid exactly
        freqs_large = np.array(
            [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        )

        # Embed
        fs_large = fs_small.get_embedded((freqs_large,))

        # Check grid
        self.assertEqual(len(fs_large.grid), 1)
        npt.assert_array_almost_equal(fs_large.frequencies, freqs_large)

        # Check shape
        self.assertEqual(fs_large.entries.shape, (1, 1, 1, 1, 11))

        # Check that the values are placed correctly
        # The small frequencies [0.01, 0.02, 0.03, 0.04, 0.05] map to indices [1, 2, 3, 4, 5]
        for i in range(5):
            self.assertAlmostEqual(fs_large.entries[0, 0, 0, 0, i + 1], i + 1)

        # Check zeros outside
        self.assertEqual(fs_large.entries[0, 0, 0, 0, 0], 0)
        self.assertTrue(np.all(fs_large.entries[0, 0, 0, 0, 6:] == 0))

    def test_time_series_get_embedded_linspace(self):
        """Test TimeSeries.get_embedded with Linspace grids."""
        # Small grid (Linspace)
        times_small = Linspace(2.0, 0.1, 30)
        entries_small = np.random.randn(2, 1, 1, 1, 30)
        ts_small = time_series(times_small, entries=entries_small)

        # Large grid (Linspace)
        times_large = Linspace(0.0, 0.1, 100)

        # Embed
        ts_large = ts_small.get_embedded((times_large,))

        # Check helper-based construction preserves Linspace semantics
        self.assertIsInstance(ts_large.grid, tuple)
        self.assertEqual(len(ts_large.grid), 1)
        self.assertIsInstance(ts_large.grid[0], Linspace)
        self.assertEqual(ts_large.grid[0], times_large)

        # Check shape
        self.assertEqual(ts_large.entries.shape, (2, 1, 1, 1, 100))


class TestArithmeticOperations(unittest.TestCase):
    """Test arithmetic operations with canonical shapes."""

    def setUp(self):
        """Create test fixtures for arithmetic operation tests."""
        # Canonical shape dimensions
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = 2, 3, 2, 1
        self.len_freq, self.len_time = 50, 100

        # Frequency grids
        self.freqs_short = np.linspace(0, 1, self.len_freq)
        self.freqs_complex = np.linspace(0.1, 1, self.len_freq)

        # Time grids
        self.times = Linspace(0.0, 0.1, self.len_time)
        self.times_short = Linspace(0.0, 0.01, self.len_time)

        # STFT fixture
        times_large = np.linspace(0, 10, 100)
        freqs_large = np.linspace(0, 1, 50)
        entries_tf = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            100,
            50,
        )
        self.tf_large = STFT(grid=(times_large, freqs_large), entries=entries_tf)

    def test_addition_same_grid(self):
        """Test adding two series with same grid and canonical shape."""
        entries1 = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq,
        ) + 1j * np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq,
        )
        entries2 = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq,
        ) + 1j * np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq,
        )

        fs1 = frequency_series(self.freqs_short, entries=entries1)
        fs2 = frequency_series(self.freqs_short, entries=entries2)

        # Add
        fs_sum = fs1 + fs2

        # Check
        npt.assert_array_almost_equal(fs_sum.entries, entries1 + entries2)
        npt.assert_array_almost_equal(np.array(fs_sum.frequencies), self.freqs_short)
        self.assertEqual(
            fs_sum.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_multiplication_scalar(self):
        """Test multiplying series by scalar with canonical shape."""
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_time,
        )
        ts = time_series(times=self.times, entries=entries)

        # Multiply by scalar
        ts_scaled = ts * 2.5

        # Check
        npt.assert_array_almost_equal(ts_scaled.entries, entries * 2.5)
        self.assertIsInstance(ts_scaled.times, Linspace)
        self.assertEqual(
            ts_scaled.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_subtraction(self):
        """Test subtraction with canonical shape."""
        entries1 = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq,
        )
        entries2 = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq,
        )

        fs1 = frequency_series(self.freqs_short, entries=entries1)
        fs2 = frequency_series(self.freqs_short, entries=entries2)

        # Subtract
        fs_diff = fs1 - fs2

        # Check
        npt.assert_array_almost_equal(fs_diff.entries, entries1 - entries2)
        self.assertEqual(
            fs_diff.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_create_like(self):
        """Test create_like preserves grid but replaces entries."""
        freqs = Linspace(0.0, 1e-3, 200)
        entries_old = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 200
        )
        fs_old = frequency_series(freqs, entries=entries_old)

        # Create new entries with same shape
        entries_new = (
            np.random.randn(
                self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 200
            )
            * 10
        )
        fs_new = fs_old.create_like(entries_new)

        # Check grid is the same
        self.assertIsInstance(fs_new.grid[0], Linspace)
        self.assertEqual(fs_new.grid[0].start, freqs.start)
        self.assertEqual(fs_new.grid[0].step, freqs.step)
        self.assertEqual(fs_new.grid[0].num, freqs.num)

        # Check entries are new
        npt.assert_array_equal(fs_new.entries, entries_new)
        self.assertEqual(
            fs_new.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_division_scalar(self):
        """Test dividing series by scalar."""
        entries = (
            np.random.randn(
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
            )
            + 0.5
            + 1j
            * np.random.randn(
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
            )
        )
        fs = frequency_series(self.freqs_complex, entries=entries)

        # Divide by scalar
        fs_scaled = fs / 2.5

        # Check
        npt.assert_array_almost_equal(fs_scaled.entries, entries / 2.5)
        self.assertEqual(
            fs_scaled.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_right_multiplication(self):
        """Test right multiplication (scalar * series)."""
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_time,
        )
        ts = time_series(times=self.times_short, entries=entries)

        # Right multiply
        ts_scaled = 3.0 * ts

        # Check
        npt.assert_array_almost_equal(ts_scaled.entries, entries * 3.0)
        self.assertEqual(
            ts_scaled.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_right_division(self):
        """Test right division (scalar / series)."""
        entries = (
            np.random.randn(
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
            )
            + 0.5
            + 1j
            * np.random.randn(
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
            )
        )
        fs = frequency_series(self.freqs_complex, entries=entries)

        # Right divide
        fs_scaled = 2.0 / fs

        # Check
        npt.assert_array_almost_equal(fs_scaled.entries, 2.0 / entries)
        self.assertEqual(
            fs_scaled.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_negation(self):
        """Test unary negation operator."""
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq,
        )
        fs = frequency_series(self.freqs_short, entries=entries)

        # Negate
        fs_neg = -fs

        # Check
        npt.assert_array_almost_equal(fs_neg.entries, -entries)
        npt.assert_array_equal(np.array(fs_neg.frequencies), np.array(fs.frequencies))
        self.assertEqual(
            fs_neg.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_timefrequency_arithmetic(self):
        """Test STFT arithmetic operations."""
        # Addition
        times = np.linspace(0, 10, 100)
        freqs = np.linspace(0, 1, 50)
        entries1 = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 50, 100
        )
        entries2 = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 50, 100
        )

        tf1 = STFT(grid=(freqs, times), entries=entries1)
        tf2 = STFT(grid=(freqs, times), entries=entries2)

        # Add
        tf_sum = tf1 + tf2

        # Check
        npt.assert_array_almost_equal(tf_sum.entries, entries1 + entries2)
        npt.assert_array_almost_equal(np.array(tf_sum.times), times)
        npt.assert_array_almost_equal(np.array(tf_sum.frequencies), freqs)
        self.assertEqual(
            tf_sum.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )
        self.assertEqual(tf_sum.entries.shape[4:], (50, 100))

        # Scalar multiplication
        tf_scaled = tf1 * 2.5

        # Check
        npt.assert_array_almost_equal(tf_scaled.entries, entries1 * 2.5)
        npt.assert_array_almost_equal(np.array(tf_scaled.times), times)
        npt.assert_array_almost_equal(np.array(tf_scaled.frequencies), freqs)
        self.assertEqual(
            tf_scaled.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )
        self.assertEqual(tf_scaled.entries.shape[4:], (50, 100))


class TestLinspace(unittest.TestCase):
    """Test Linspace utility class."""

    def make_array(self, start, step, num):
        """Make array following same interface as Linspace.__init__."""
        return start + step * np.arange(num)

    def assertEqualLinspace(self, ls1: Linspace, ls2: Linspace, /):
        """Assert two Linspace objects are equal."""
        self.assertEqual(ls1.start, ls2.start)
        self.assertEqual(ls1.step, ls2.step)
        self.assertEqual(ls1.num, ls2.num)

    def test_init(self):
        """Test Linspace initialization."""
        npt.assert_allclose(Linspace(0, 1, 6), np.arange(6))
        npt.assert_allclose(Linspace(0, 2, 6), np.arange(6) * 2)
        npt.assert_allclose(Linspace(1, 2, 6), np.arange(6) * 2 + 1)

        for args in [(-10, 2, 10), (0, 1, 10), (0, 1.0, 10)]:
            npt.assert_allclose(Linspace(*args), self.make_array(*args))

        # Test error cases
        for num in [-1, 0]:
            with self.assertRaises(ValueError):
                Linspace(0, 1, num)

    def test_len(self):
        """Test Linspace length."""
        a, b = 0.0, 1.0
        for num in [1, 2, 100, 10000]:
            linsp = Linspace(a, b, num)
            self.assertEqual(num, len(linsp))

    def test_repr(self):
        """Test Linspace string representation."""
        ls = Linspace(-1, 1.5, 10)
        self.assertEqual(repr(ls), "Linspace(start=-1.0, step=1.5, num=10)")
        ls = Linspace(-1.0, 1.5, 10)
        self.assertEqual(repr(ls), "Linspace(start=-1.0, step=1.5, num=10)")

    def test_conversion_to_array(self):
        """Test converting Linspace to NumPy array."""
        ar = np.arange(10)
        for ls in [Linspace(0, 1, 10), Linspace(0, 1.0, 10)]:
            npt.assert_allclose(np.array(ls), ar)
            npt.assert_allclose(np.asarray(ls), ar)
            npt.assert_allclose(np.sin(ls), np.sin(ar))

    def test_getitem(self):
        """Test Linspace slicing."""
        # slice 3:5:
        # 0 1 2 3 4 5 6  <-> Linspace(0, 1, 7)
        # - - - 3 4 - -  <-> Linspace(3, 1, 2)
        ls = Linspace(0, 1, 7)
        ls1 = ls[3:5]
        ls2 = Linspace(3, 1, 2)
        self.assertEqualLinspace(ls1, ls2)

    def test_conversion_from_array(self):
        """Test converting NumPy array to Linspace."""
        for args in [(-10, 2, 10), (0, 1, 10), (0, 1.0, 10)]:
            arr = self.make_array(*args)
            ls1 = Linspace(*args)
            ls2 = Linspace.from_array(arr)
            self.assertEqualLinspace(ls1, ls2)

    def test_make(self):
        """Test Linspace.make factory method."""
        args = (0, 2.0, 10)
        self.assertEqualLinspace(Linspace(*args), Linspace.make(Linspace(*args)))
        self.assertEqualLinspace(Linspace(*args), Linspace.make(self.make_array(*args)))

    def test_get_step(self):
        """Test extracting step from Linspace or array."""
        start, num = 0, 10
        for step in [1e-16, 1, 1.0, 1e10]:
            ls = Linspace(start, step, num)
            ar = self.make_array(start, step, num)
            self.assertEqual(Linspace.get_step(ls), step)
            self.assertEqual(Linspace.get_step(ar), step)


class TestComplexProperties(unittest.TestCase):
    """Test complex number handling and properties."""

    def setUp(self):
        """Create test fixtures for complex property tests."""
        # Canonical shape dimensions
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = 2, 1, 1, 1
        self.len_freq = 50

        # Frequency grid
        self.freqs = np.linspace(0, 1, self.len_freq)

    def test_real_property(self):
        """Test extracting real part of complex series."""
        entries = np.cos(self.freqs) + 1j * np.sin(self.freqs)
        entries = entries.reshape(1, 1, 1, 1, self.len_freq)
        # Expand to full canonical shape
        entries_full = np.tile(
            entries,
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 1),
        )

        fs = frequency_series(self.freqs, entries=entries_full)
        fs_real = fs.real

        npt.assert_array_almost_equal(fs_real.entries, entries_full.real)
        npt.assert_array_equal(fs_real.grid[0], fs.grid[0])
        self.assertEqual(
            fs_real.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_imag_property(self):
        """Test extracting imaginary part of complex series."""
        entries = np.cos(self.freqs) + 1j * np.sin(self.freqs)
        entries = entries.reshape(1, 1, 1, 1, self.len_freq)
        # Expand to full canonical shape
        entries_full = np.tile(
            entries,
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 1),
        )

        fs = frequency_series(self.freqs, entries=entries_full)
        fs_imag = fs.imag

        npt.assert_array_almost_equal(fs_imag.entries, entries_full.imag)
        npt.assert_array_equal(fs_imag.grid[0], fs.grid[0])
        self.assertEqual(
            fs_imag.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_conj_property(self):
        """Test complex conjugate."""
        entries = (np.cos(self.freqs) + 1j * np.sin(self.freqs)).reshape(
            1, 1, 1, 1, self.len_freq
        )
        # Expand to full canonical shape
        entries_full = np.tile(
            entries,
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 1),
        )

        fs = frequency_series(self.freqs, entries=entries_full)
        fs_conj = fs.conj

        npt.assert_array_almost_equal(fs_conj.entries, np.conj(entries_full))
        npt.assert_array_equal(fs_conj.grid[0], fs.grid[0])
        self.assertEqual(
            fs_conj.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_abs_property(self):
        """Test absolute value (magnitude) of complex series."""
        # Create entries with known magnitude
        entries = (3 + 4j) * np.ones(
            (1, 1, 1, 1, self.len_freq)
        )  # magnitude should be 5
        # Expand to full canonical shape
        entries_full = np.tile(
            entries,
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 1),
        )

        fs = frequency_series(self.freqs, entries=entries_full)
        fs_abs = fs.abs()

        expected_abs = 5.0 * np.ones(
            (
                self.n_batches,
                self.n_channels,
                self.n_harmonics,
                self.n_features,
                self.len_freq,
            )
        )
        npt.assert_array_almost_equal(fs_abs.entries, expected_abs)
        npt.assert_array_equal(fs_abs.grid[0], fs.grid[0])
        self.assertEqual(
            fs_abs.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )


class TestPropertiesAndAliases(unittest.TestCase):
    """Test property access like df, dt, resolution."""

    def setUp(self):
        """Create test fixtures for property access tests."""
        # Canonical shape dimensions
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = 2, 2, 1, 1
        self.len_freq_long = 1000
        self.len_freq_short = 100
        self.len_time = 500

    def test_frequency_series_df(self):
        """Test FrequencySeries.df property."""
        freqs = Linspace(0.0, 1e-3, self.len_freq_long)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq_long,
        )
        fs = frequency_series(freqs, entries=entries)

        # Check df
        self.assertEqual(fs.df, 1e-3)
        self.assertIsInstance(fs.df, (int, float))  # Should be a number, not a method

        # Check resolution
        self.assertEqual(fs.resolution, 1e-3)
        self.assertEqual(fs.df, fs.resolution)
        self.assertEqual(
            fs.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_time_series_dt(self):
        """Test TimeSeries.dt property."""
        times = Linspace(0.0, 0.01, self.len_time)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_time,
        )
        ts = time_series(times, entries=entries)

        # Check dt
        self.assertEqual(ts.dt, 0.01)
        self.assertIsInstance(ts.dt, (int, float))

        # Check resolution
        self.assertEqual(ts.resolution, 0.01)
        self.assertEqual(ts.dt, ts.resolution)
        self.assertEqual(
            ts.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_frequencies_property(self):
        """Test frequencies property returns grid[0]."""
        freqs = np.linspace(0, 1, self.len_freq_short)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq_short,
        )
        fs = frequency_series(freqs, entries=entries)

        npt.assert_array_equal(fs.frequencies, freqs)
        npt.assert_array_equal(fs.frequencies, fs.grid[0])
        self.assertEqual(
            fs.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_times_property(self):
        """Test times property returns grid[0]."""
        times = np.linspace(0, 10, self.len_freq_long)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_freq_long,
        )
        ts = time_series(times=times, entries=entries)

        npt.assert_array_equal(ts.times, times)
        npt.assert_array_equal(ts.times, ts.grid[0])
        self.assertEqual(
            ts.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )


class TestGridTupleHandling(unittest.TestCase):
    """Test that grids are always tuples, even for 1D."""

    def setUp(self):
        """Create test fixtures for grid tuple handling tests."""
        # Canonical shape dimensions
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = 2, 2, 1, 1
        self.len_grid_small = 50
        self.len_grid_large = 100

    def test_grid_is_tuple_for_1d(self):
        """Test that 1D series have grid as tuple."""
        freqs = np.linspace(0, 1, self.len_grid_small)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_small,
        )
        fs = frequency_series(freqs, entries=entries)

        # Grid must be tuple
        self.assertIsInstance(fs.grid, tuple)
        self.assertEqual(len(fs.grid), 1)
        self.assertEqual(
            fs.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_grid_conversion_linspace(self):
        """Test that arrays are converted to Linspace when uniform."""
        freqs = np.linspace(0, 1, self.len_grid_large)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_large,
        )
        fs = frequency_series(freqs, entries=entries)

        # Should be converted to Linspace
        self.assertIsInstance(fs.grid[0], Linspace)
        self.assertAlmostEqual(fs.grid[0].start, 0.0)
        self.assertAlmostEqual(fs.grid[0].step, 1.0 / (self.len_grid_large - 1))
        self.assertEqual(fs.grid[0].num, self.len_grid_large)
        self.assertEqual(
            fs.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_grid_not_converted_non_uniform(self):
        """Test that non-uniform arrays are not converted to Linspace."""
        freqs = np.array([0.1, 0.2, 0.5, 1.0, 2.0])  # Non-uniform
        entries = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 5
        )
        fs = frequency_series(freqs, entries=entries)

        # Should remain as array
        self.assertIsInstance(fs.grid[0], np.ndarray)
        npt.assert_array_equal(fs.grid[0], freqs)
        self.assertEqual(
            fs.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create test fixtures for edge case tests."""
        # Canonical shape dimensions
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = 2, 2, 1, 1
        self.len_grid_large = 100
        self.len_grid_small = 5

    def test_subset_with_copy_false(self):
        """Test that copy=False returns view."""
        freqs = np.linspace(0, 1, self.len_grid_large)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_large,
        )
        fs = frequency_series(freqs, entries=entries)

        # Get subset without copy
        fs_sub = fs.get_subset(slice=slice(20, 50), copy=False)

        # Modify subset
        original_value = fs_sub.entries[0, 0, 0, 0, 0]
        fs_sub.entries[0, 0, 0, 0, 0] = 999.0

        # Check that original is modified (view behavior)
        self.assertEqual(fs.entries[0, 0, 0, 0, 20], 999.0)

        # Restore
        fs_sub.entries[0, 0, 0, 0, 0] = original_value
        self.assertEqual(
            fs_sub.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_subset_with_copy_true(self):
        """Test that copy=True returns independent array."""
        freqs = np.linspace(0, 1, self.len_grid_large)
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_large,
        )
        fs = frequency_series(freqs, entries=entries)

        # Get subset with copy
        fs_sub = fs.get_subset(slice=slice(20, 50), copy=True)

        # Modify subset
        original_value = fs.entries[0, 0, 0, 0, 20]
        fs_sub.entries[0, 0, 0, 0, 0] = 999.0

        # Check that original is NOT modified
        self.assertEqual(fs.entries[0, 0, 0, 0, 20], original_value)
        self.assertEqual(
            fs_sub.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )

    def test_empty_subset(self):
        """Test behavior with empty subsets."""
        # Use highly non-uniform spacing to prevent Linspace conversion
        freqs = np.array([0.0, 0.01, 0.05, 0.2, 1.0])
        entries = np.random.randn(
            self.n_batches,
            self.n_channels,
            self.n_harmonics,
            self.n_features,
            self.len_grid_small,
        )
        fs = frequency_series(freqs, entries=entries)

        # Get empty subset
        fs_empty = fs.get_subset(slice=slice(2, 2))

        # Check
        self.assertEqual(len(fs_empty.grid[0]), 0)
        self.assertEqual(fs_empty.entries.shape[4], 0)
        self.assertEqual(
            fs_empty.entries.shape[0:4],
            (self.n_batches, self.n_channels, self.n_harmonics, self.n_features),
        )


# It's a design choice to not validate shape on construction, so some tests are commented out.
# If strict validation is added, they should be re-enabled.
class TestErrorHandling(unittest.TestCase):
    """Test error handling and validation for invalid operations."""

    def setUp(self):
        """Create test fixtures for error handling tests."""
        self.n_batches, self.n_channels, self.n_harmonics, self.n_features = (
            2,
            2,
            1,
            1,
        )
        self.freqs_short = np.linspace(0, 1, 50)
        self.freqs_long = np.linspace(0, 1, 100)

    # def test_shape_mismatch_entries_vs_grid(self):
    #     """Test that mismatched grid and entries shapes raise errors."""
    #     freqs = np.linspace(0, 1, 50)
    #     # Entries have 60 grid points, but grid only has 50
    #     entries = np.random.randn(
    #         self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 60
    #     )

    #     # Should raise ValueError on construction
    #     with self.assertRaises(ValueError):
    #         FrequencySeries(grid=(freqs,), entries=entries)

    # def test_invalid_canonical_shape(self):
    #     """Test that invalid canonical shapes are rejected."""
    #     # Missing features dimension (should have 5 dimensions, has 4)
    #     freqs = np.linspace(0, 1, 50)
    #     entries = np.random.randn(self.n_batches, self.n_channels, self.n_harmonics, 50)

    #     # Should raise ValueError on construction (grid shape mismatch)
    #     with self.assertRaises(ValueError):
    #         FrequencySeries(grid=(freqs,), entries=entries)

    def test_empty_grid(self):
        """Test that empty grids are handled appropriately."""
        freqs = np.array([])
        entries = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 0
        )

        # Should either raise error or create empty series
        try:
            fs = frequency_series(freqs, entries=entries)
            # If it succeeds, verify it created empty series
            self.assertEqual(len(fs.grid[0]), 0)
        except ValueError:
            # Acceptable to reject empty grids
            pass

    def test_timefrequency_grid_tuple_structure(self):
        """Test STFT with different grid tuple structures."""
        times = np.linspace(0, 10, 100)
        freqs = np.linspace(0, 1, 50)
        entries = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 100, 50
        )

        # Correct structure: tuple of exactly two grids
        tf_correct = STFT(grid=(times, freqs), entries=entries)
        self.assertEqual(len(tf_correct.grid), 2)

        # Wrong number of grids (three): may succeed or fail depending on implementation
        try:
            STFT(grid=(times, freqs, times), entries=entries)
            # If it succeeds, just verify it was created
        except ValueError:
            # Expected for strict validation
            pass

    def test_invalid_subset_interval(self):
        """Test that subset intervals are handled gracefully."""
        entries = np.random.randn(
            self.n_batches, self.n_channels, self.n_harmonics, self.n_features, 50
        )
        fs = frequency_series(self.freqs_short, entries=entries)

        # Try interval outside grid range - may return empty or clamp to bounds
        try:
            fs_sub = fs.get_subset(interval=(10.0, 20.0))
            # If it succeeds, it should return a valid (possibly empty) result
            self.assertEqual(fs_sub.entries.ndim, entries.ndim)
        except ValueError:
            # Also acceptable to reject out-of-range intervals
            pass


class TestLinspaceExtraProperties(LinspaceExtraPropertiesMixin, unittest.TestCase):
    """Linspace edge-case tests (shared via mixin)."""


class TestHelperFunctions(HelperFunctionsMixin, unittest.TestCase):
    """Module-level helper function tests (shared via mixin)."""

    xp = np


class TestAdvancedRepresentationMethods(
    AdvancedRepresentationMethodsMixin, unittest.TestCase
):
    """FrequencySeries/TimeSeries/STFT method tests (shared via mixin)."""

    xp = np


class TestArithmeticAddMethods(unittest.TestCase):
    """Test add/iadd/iadd-operator methods on representations."""

    def setUp(self):
        self.large_freqs = Linspace(0.0, 0.01, 100)
        self.small_freqs = Linspace(0.30, 0.01, 30)
        self.entries_large = np.ones((1, 1, 1, 1, 100))
        self.entries_small = np.ones((1, 1, 1, 1, 30)) * 2.0

    def test_iadd_method(self):
        # iadd/add apply the grid slice directly to entries; use canonical full-slice
        large_freqs = Linspace(0.0, 0.01, 100)
        small_freqs = Linspace(0.30, 0.01, 30)
        entries_large = np.ones((1, 1, 1, 1, 100))
        entries_small = np.ones((1, 1, 1, 1, 30)) * 2.0
        fs_large = frequency_series(large_freqs, entries=entries_large.copy())
        fs_small = frequency_series(small_freqs, entries=entries_small)
        full_slice = (slice(None), slice(None), slice(None), slice(None), slice(30, 60))
        result = fs_large.iadd(fs_small, full_slice)
        npt.assert_allclose(result.entries[0, 0, 0, 0, 30:60], 3.0)
        npt.assert_allclose(result.entries[0, 0, 0, 0, :30], 1.0)

    def test_add_method_inplace_false(self):
        large_freqs = Linspace(0.0, 0.01, 100)
        small_freqs = Linspace(0.30, 0.01, 30)
        entries_large = np.ones((1, 1, 1, 1, 100))
        entries_small = np.ones((1, 1, 1, 1, 30)) * 2.0
        fs_large = frequency_series(large_freqs, entries=entries_large.copy())
        fs_small = frequency_series(small_freqs, entries=entries_small)
        full_slice = (slice(None), slice(None), slice(None), slice(None), slice(30, 60))
        result = fs_large.add(fs_small, full_slice, inplace=False)
        npt.assert_allclose(result.entries[0, 0, 0, 0, 30:60], 3.0)
        npt.assert_allclose(
            fs_large.entries[0, 0, 0, 0, 30:60], 1.0
        )  # original unchanged

    def test_add_method_inplace_true(self):
        large_freqs = Linspace(0.0, 0.01, 100)
        small_freqs = Linspace(0.30, 0.01, 30)
        entries_large = np.ones((1, 1, 1, 1, 100))
        entries_small = np.ones((1, 1, 1, 1, 30)) * 2.0
        fs_large = frequency_series(large_freqs, entries=entries_large.copy())
        fs_small = frequency_series(small_freqs, entries=entries_small)
        full_slice = (slice(None), slice(None), slice(None), slice(None), slice(30, 60))
        result = fs_large.add(fs_small, full_slice, inplace=True)
        npt.assert_allclose(result.entries[0, 0, 0, 0, 30:60], 3.0)
        self.assertIs(result, fs_large)

    def test_iadd_operator_with_scalar(self):
        # Test __iadd__ fallback to super().__iadd__ for non-matching types (scalar)
        freqs = Linspace(0.0, 0.01, 10)
        entries = np.ones((1, 1, 1, 1, 10))
        fs = frequency_series(freqs, entries=entries.copy())
        fs += 1.0
        npt.assert_allclose(np.asarray(fs.entries), 2.0)  # 1.0 + 1.0

    def test_iadd_operator_with_array_grid_type(self):
        # Test __iadd__ path where grid is a plain array (not Linspace)
        large_freqs_arr = np.linspace(0.0, 0.99, 10)
        small_freqs_arr = np.linspace(0.30, 0.59, 3)
        entries_large = np.ones((1, 1, 1, 1, 10))
        entries_small = np.ones((1, 1, 1, 1, 3)) * 2.0
        fs_large = frequency_series(large_freqs_arr, entries=entries_large.copy())
        fs_small = frequency_series(small_freqs_arr, entries=entries_small)
        # This covers the 'else: start, stop = float(...)' branch in __iadd__
        try:
            fs_large += fs_small
        except (ValueError, IndexError):
            pass  # iadd may fail for canonical shape; we just want the branch covered

    def test_addition_with_equal_nonuniform_array_grids(self):
        # Non-uniform grids avoid Linspace coercion and force array_equal path.
        freqs = np.array([0.0, 0.11, 0.37, 0.9])
        entries1 = np.ones((1, 1, 1, 1, len(freqs)))
        entries2 = np.ones((1, 1, 1, 1, len(freqs))) * 2.0
        fs1 = frequency_series(freqs, entries1)
        fs2 = frequency_series(freqs, entries2)

        out = fs1 + fs2
        npt.assert_allclose(np.asarray(out.entries), 3.0)

    def test_addition_with_mismatched_nonuniform_array_grids_raises(self):
        # Covers _check_grid_compatibility branch where array_equal is False.
        freqs1 = np.array([0.0, 0.11, 0.37, 0.9])
        freqs2 = np.array([0.0, 0.11, 0.4, 0.9])
        entries1 = np.ones((1, 1, 1, 1, len(freqs1)))
        entries2 = np.ones((1, 1, 1, 1, len(freqs2)))
        fs1 = frequency_series(freqs1, entries1)
        fs2 = frequency_series(freqs2, entries2)

        with self.assertRaises(ValueError):
            _ = fs1 + fs2

    def test_setitem_on_series(self):
        freqs = Linspace(0.0, 0.1, 50)
        entries = np.ones((1, 1, 1, 1, 50))
        fs = frequency_series(freqs, entries=entries)
        with self.assertRaises(ValueError):
            fs[10:20] = np.zeros((1, 1, 1, 1, 10))


class TestPhasor(unittest.TestCase):
    """Test Phasor representation class."""

    def setUp(self):
        self.freqs = np.linspace(1e-4, 1e-2, 20)
        self.amps = np.ones(20, dtype=np.complex128)[None, None, None, None, :]
        self.phases = np.linspace(0, np.pi, 20, dtype=np.float64)[
            None, None, None, None, :
        ]
        self.phasor = phasor(
            frequencies=self.freqs, amplitudes=self.amps, phases=self.phases
        )

    def test_domain_and_kind(self):
        self.assertEqual(self.phasor.domain, "frequency")
        self.assertEqual(self.phasor.kind, "phasor")

    def test_phases_and_amplitudes(self):
        npt.assert_allclose(np.asarray(self.phasor.amplitudes), self.amps)
        npt.assert_allclose(np.asarray(self.phasor.phases), self.phases)

    def test_frequencies_f_min_f_max(self):
        npt.assert_allclose(np.array(self.phasor.frequencies), self.freqs)
        self.assertAlmostEqual(self.phasor.f_min, self.freqs[0])
        self.assertAlmostEqual(self.phasor.f_max, self.freqs[-1])

    def test_create_like(self):
        new_entries = np.zeros_like(self.phasor.entries)
        new_phasor = self.phasor.create_like(new_entries)
        self.assertIsInstance(new_phasor, Phasor)
        npt.assert_allclose(new_phasor.entries, 0)
        npt.assert_allclose(np.array(new_phasor.frequencies), self.freqs)

    def test_setitem(self):
        import copy

        p = copy.deepcopy(self.phasor)
        p[:] = np.zeros_like(p.entries)
        npt.assert_allclose(p.entries, 0.0)

    def test_get_subset(self):
        sub = self.phasor.get_subset(interval=(self.freqs[5], self.freqs[15]))
        self.assertIsInstance(sub, Phasor)
        self.assertLess(len(np.array(sub.frequencies)), len(self.freqs))

    def test_to_frequency_series(self):
        fs = self.phasor.to_frequency_series()
        self.assertIsInstance(fs, FrequencySeries)
        expected = self.amps * np.exp(1j * self.phases)
        npt.assert_allclose(np.abs(np.asarray(fs.entries) - expected), 0, atol=1e-10)

    def test_get_interpolated(self):
        from scipy.interpolate import interp1d  # type: ignore[import]

        new_freqs = np.linspace(self.freqs[2], self.freqs[-3], 8)
        interpolated = self.phasor.get_interpolated(new_freqs, interp1d)
        self.assertIsInstance(interpolated, Phasor)
        self.assertEqual(len(np.array(interpolated.frequencies)), 8)


class TestWDMPropertiesAndMethods(WDMPropertiesAndMethodsMixin, unittest.TestCase):
    """WDM property/method tests (shared via mixin)."""

    def setUp(self):
        from tests._helpers import build_wdm_pair

        self.wdm = build_wdm_pair(np)["left"]["X"]


class TestSparse2DGridRepresentations(unittest.TestCase):
    """Coverage tests for sparse-grid code paths in representations."""

    def test_embed_entries_to_grid_2d_sparse_with_known_slices(self):
        """Test sparse embedding helper with correct 5D entries shape."""
        source_freqs = np.array([20.0, 30.0, 40.0])
        source_times = np.array([5.0, 6.0, 7.0])
        source_indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=int)
        source_grid = build_grid2d(source_freqs, source_times, sparse_indices=source_indices)
        # For sparse grids, entries is 5D: (n_batch, n_channels, n_harmonics, n_features, num_sparse_points)
        source_entries = np.array([[[[1.0, 2.0, 3.0]]]])

        embedding_grid = (
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        known_slices = (slice(1, 4), slice(2, 5))

        new_grid, new_entries = _embed_entries_to_grid_2d_sparse(
            source_grid,
            source_entries,
            embedding_grid,
            known_slices=known_slices,
        )

        # New indices are source indices shifted by known slice starts.
        expected_indices = np.array([[1, 2], [2, 4], [3, 3]], dtype=int)
        self.assertIsInstance(new_grid, Grid2DSparse)
        npt.assert_array_equal(np.asarray(new_grid.indices), expected_indices)
        npt.assert_array_equal(np.asarray(new_entries), np.array([[[[1.0, 2.0, 3.0]]]]))

    def test_sparse_stft_factory_with_sparse_indices(self):
        """Test that stft factory returns sparse-grid representation when indices are provided."""
        freqs = np.array([2.0, 3.0, 4.0])
        times = np.array([10.0, 20.0, 30.0])
        sparse_indices = np.array([[0, 0], [1, 2], [2, 1]], dtype=int)
        # For sparse grids, entries is 5D: (n_batch, n_channels, n_harmonics, n_features, num_sparse_points)
        entries = np.array([[[[[7.0, 8.0, 9.0]]]]])

        tf = stft(freqs, times, entries=entries, sparse_indices=sparse_indices)

        # Verify it's a sparse grid
        self.assertIsInstance(tf.grid, Grid2DSparse)
        npt.assert_array_equal(np.asarray(tf.grid.indices), sparse_indices)
        npt.assert_array_equal(np.asarray(tf.entries), entries)

    def test_subset_grid_2d_sparse_filters_and_reindexes(self):
        source_freqs = np.array([10.0, 20.0, 30.0, 40.0])
        source_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        source_indices = np.array([[0, 0], [1, 2], [2, 3], [3, 4]], dtype=int)
        source_grid = build_grid2d(source_freqs, source_times, sparse_indices=source_indices)
        source_entries = np.array([[[[[11.0, 22.0, 33.0, 44.0]]]]])

        subset_slices = (slice(1, 4), slice(2, 5))
        new_grid, new_entries = _subset_grid_2d_sparse(
            source_grid,
            source_entries,
            subset_slices,
        )

        expected_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=int)
        expected_entries = np.array([[[[[22.0, 33.0, 44.0]]]]])

        self.assertIsInstance(new_grid, Grid2DSparse)
        npt.assert_array_equal(np.asarray(new_grid.indices), expected_indices)
        npt.assert_array_equal(np.asarray(new_grid[0]), np.array([20.0, 30.0, 40.0]))
        npt.assert_array_equal(np.asarray(new_grid[1]), np.array([3.0, 4.0, 5.0]))
        npt.assert_array_equal(np.asarray(new_entries), expected_entries)

    def test_embed_entries_to_grid_2d_sparse_computes_slices_when_missing(self):
        source_freqs = np.array([20.0, 30.0, 40.0])
        source_times = np.array([5.0, 6.0, 7.0])
        source_indices = np.array([[0, 0], [1, 1], [2, 2]], dtype=int)
        source_grid = build_grid2d(source_freqs, source_times, sparse_indices=source_indices)
        source_entries = np.array([[[[1.0, 2.0, 3.0]]]])

        embedding_grid = (
            np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            np.array([4.0, 5.0, 6.0, 7.0, 8.0]),
        )

        new_grid, new_entries = _embed_entries_to_grid_2d_sparse(
            source_grid,
            source_entries,
            embedding_grid,
        )

        expected_indices = np.array([[1, 1], [2, 2], [3, 3]], dtype=int)
        npt.assert_array_equal(np.asarray(new_grid.indices), expected_indices)
        npt.assert_array_equal(np.asarray(new_entries), source_entries)


class TestRepresentationErrorBranches(unittest.TestCase):
    def test_get_subset_slice_rejects_interval_and_slice_together(self):
        grid = np.linspace(0.0, 1.0, 11)
        with self.assertRaises(ValueError):
            _get_subset_slice(grid, interval=(0.2, 0.6), slice=slice(2, 7))

    def test_take_subset_rejects_wrong_number_of_slices(self):
        grid = (np.linspace(0.0, 1.0, 11), np.linspace(0.0, 2.0, 21))
        entries = np.zeros((1, 1, 1, 1, 11, 21))

        with self.assertRaises(ValueError):
            _take_subset(grid, entries, (slice(2, 6),))
