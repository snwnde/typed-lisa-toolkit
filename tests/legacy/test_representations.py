import unittest
import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_array_almost_equal
from typed_lisa_toolkit.containers.representations import (
    TimeSeries,
    FrequencySeries,
    Phasor,
    Linspace,
    TimeFrequency,
    WDM,
)
from typed_lisa_toolkit.viz.plotters import FSPlotter, TSPlotter, TimeFrequencyPlotter
from typed_lisa_toolkit.containers.tapering import ldc_window, planck_window

TAPERING_WINDOWS = [ldc_window(), planck_window()]


class MixinTestUtils(unittest.TestCase):
    "common test utilities"

    def assertAllClose(self, x, y: NDArray | Linspace, /, **kwargs):
        "This won't work if the second argument is a _Series!"
        # that's because allclose will call asanyarray on y, and _Series is
        # not a subclass of ndarray, meaning it will get converted to an
        # array of dtype object, containing your series as the only element.
        # this will subsequently fail isfinite().
        self.assertTrue(np.allclose(x, y, **kwargs), msg=f"first = {x}, second = {y}")

    def assertArrayEq(self, x, y, /):
        self.assertTrue(np.array_equal(x, y), msg=f"first = {x}, second = {y}")


class TestLinspace(MixinTestUtils, unittest.TestCase):
    def make_array(self, start, step, num):
        "make array following same interface as Linspace.__init__"
        return start + step * np.arange(num)

    def assertEqualLinspace(self, ls1: Linspace, ls2: Linspace, /):
        self.assertEqual(ls1.start, ls2.start)
        self.assertEqual(ls1.step, ls2.step)
        self.assertEqual(ls1.stop, ls2.stop)
        self.assertEqual(ls1.num, ls2.num)
        self.assertEqual(ls1.shape, ls2.shape)

    def testinit(self):
        self.assertAllClose(Linspace(0, 1, 6), np.arange(6))
        self.assertAllClose(Linspace(0, 2, 6), np.arange(6) * 2)
        self.assertAllClose(Linspace(1, 2, 6), np.arange(6) * 2 + 1)

        for args in [(-10, 2, 10), (0, 1, 10), (0, 1.0, 10)]:
            self.assertAllClose(Linspace(*args), self.make_array(*args))

        for num in [-1, 0]:
            with self.assertRaises(ValueError):
                Linspace(0, 1, num)

    def testlen(self):
        a, b = 0.0, 1.0
        for num in [1, 2, 100, np.iinfo(np.int64).max]:
            linsp = Linspace(a, b, num)
            self.assertEqual(num, len(linsp))

        with self.assertRaises(OverflowError):
            len(Linspace(a, b, np.iinfo(np.int64).max + 1))

    def test_repr(self):
        ls = Linspace(-1, 1.5, 10)
        self.assertEqual(repr(ls), "Linspace(start=-1, step=1.5, num=10)")
        ls = Linspace(-1.0, 1.5, 10)
        self.assertEqual(repr(ls), "Linspace(start=-1.0, step=1.5, num=10)")
        ls = Linspace(100, 1, 10)
        self.assertEqual(repr(ls), "Linspace(start=100, step=1, num=10)")

    def test_conversion_to_array(self):
        ar = np.arange(10)
        for ls in [Linspace(0, 1, 10), Linspace(0, 1.0, 10)]:
            self.assertAllClose(np.array(ls), ar)
            self.assertAllClose(np.asarray(ls), ar)
            self.assertAllClose(np.sin(ls), np.sin(ar))

    def test_getitem(self):
        # slice 3:5:
        # 0 1 2 3 4 5 6  <-> Linspace(0, 1, 7)
        # - - - 3 4 - -  <-> Linspace(3, 1, 2)
        ls = Linspace(0, 1, 7)
        ls1 = ls[3:5]
        ls2 = Linspace(3, 1, 2)
        self.assertEqualLinspace(ls1, ls2)

    def test_conversion_from_array(self):
        for args in [(-10, 2, 10), (0, 1, 10), (0, 1.0, 10)]:
            arr = self.make_array(*args)
            ls1 = Linspace(*args)
            ls2 = Linspace.from_array(arr)
            self.assertEqualLinspace(ls1, ls2)

    def test_make(self):
        args = (0, 2.0, 10)
        self.assertEqualLinspace(Linspace(*args), Linspace.make(Linspace(*args)))
        self.assertEqualLinspace(Linspace(*args), Linspace.make(self.make_array(*args)))

    def test_get_step(self):
        start, num = 0, 10
        for step in [1e-16, 1, 1.0, 1e10, 10000, np.iinfo(np.int64).max // 10]:
            ls = Linspace(start, step, num)
            ar = self.make_array(start, step, num)
            self.assertEqual(Linspace.get_step(ls), step)
            self.assertEqual(Linspace.get_step(ar), step)








# class TestSeriesSuperclass(MixinTestUtils, unittest.TestCase):
#     "Test that some behaviors of TimeSeries, FrequencySeries and their superclass match"

#     def rep3(self, func):
#         "do the same test for TimeSeries, FrequencySeries and _Series"
#         func(_Series)
#         func(TimeSeries)
#         func(FrequencySeries)

#     def test_init(self):
#         grid = np.linspace(0, 1, 100)
#         entries_real = np.sin(grid)
#         entries_complex = entries_real * (1 + 1j)

#         def f(c, entries):
#             # must accept grid and entries as positional or kw arguments
#             series1 = c(grid=grid, entries=entries)
#             series2 = c(grid, entries)

#             # the results should match
#             self.assertArrayEq(series1.grid, series2.grid)
#             self.assertArrayEq(series1.entries, series2.entries)

#         self.rep3(lambda c: f(c, entries_real))
#         self.rep3(lambda c: f(c, entries_complex))

#     def test_is_consistent(self):
#         grid = np.linspace(0, 1, 100)
#         entries = np.sin(grid)

#         def f(c):
#             series = c(grid=grid, entries=entries)
#             self.assertTrue(series.is_consistent())
#             series = c(grid=grid[:-1], entries=entries)
#             self.assertFalse(series.is_consistent())

#         self.rep3(f)

#     def test_create_like(self):
#         grid = np.linspace(0, 1, 100)
#         entries_old = np.sin(grid) * (1 + 1j)
#         entries_new = entries_old * 2

#         def f(c):
#             series_old = c(grid, entries_old)
#             series_new = series_old.create_like(entries_new)
#             self.assertArrayEq(series_old.grid, series_new.grid)
#             self.assertArrayEq(entries_new, series_new.entries)

#         self.rep3(f)

#     def test_resolution(self):
#         def f(c):
#             for step in [1e-16, 1, 1.0]:
#                 grid = Linspace(0, step, 10)
#                 entries = np.sin(grid)
#                 series = c(grid, entries)
#                 self.assertEqual(step, series.resolution)

#         self.rep3(f)

#     def test_arithmetic_ops(self):
#         grid = np.linspace(0, 1, 100)
#         entries1 = np.sin(grid)
#         entries2 = np.cos(grid)

#         def f(c):
#             series1 = c(grid, entries1)
#             series2 = c(grid, entries2)
#             self.assertArrayEq((series1 + series2).entries, entries1 + entries2)
#             self.assertArrayEq((series1 - series2).entries, entries1 - entries2)
#             self.assertArrayEq((series1 * series2).entries, entries1 * entries2)
#             self.assertArrayEq((series1 / series2).entries, entries1 / entries2)
#             self.assertArrayEq((series1 * 2).entries, entries1 * 2)
#             self.assertArrayEq((2 * series1).entries, entries1 * 2)
#             self.assertArrayEq((series1 / 2).entries, entries1 / 2)
#             self.assertArrayEq((2 / series2).entries, 2 / entries2)

#         self.rep3(f)

#     def test_unary_ops(self):
#         grid = np.linspace(0, 1, 100)
#         entries_positive = np.cos(grid)
#         entries = np.sin(grid) * 1j + np.cos(grid)

#         def f(c):
#             series = c(grid, entries)
#             series_positive = c(grid, entries_positive)
#             # the following all come from lib.mixins.NDArrayMixin
#             self.assertArrayEq((-series).entries, -entries)
#             self.assertArrayEq((series.square()).entries, np.square(entries))
#             self.assertArrayEq((series.exp()).entries, np.exp(entries))
#             self.assertArrayEq(
#                 (series_positive.sqrt()).entries, np.sqrt(entries_positive)
#             )
#             self.assertArrayEq((series.conj()).entries, np.conj(entries))
#             self.assertArrayEq((series.abs()).entries, np.abs(entries))

#         self.rep3(f)

#     def test_ufunc_call(self):
#         grid = np.linspace(0, 1, 10)
#         entries = np.sin(grid)
#         entries2 = np.cos(grid)
#         entries3 = np.exp(1j * grid)

#         def f(c):
#             series = c(grid, entries)
#             series2 = c(grid, entries2)
#             series3 = c(grid, entries3)

#             # binary ops
#             for ufunc in [np.fmod, np.greater, np.maximum, np.copysign]:
#                 self.assertArrayEq(
#                     (ufunc(series, series2)).entries, ufunc(entries, entries2)
#                 )

#             # unary ops
#             for ufunc in [np.exp, np.isfinite]:
#                 self.assertArrayEq(ufunc(series).entries, ufunc(entries))
#                 self.assertArrayEq(ufunc(series3).entries, ufunc(entries3))

#         self.rep3(f)

#     def test_ufunc_at(self):
#         grid = (np.linspace(0, 1, 5), )
#         entries = np.sin(grid)

#         def do_one_ufunc(c, ufunc, *args):
#             # the .at method works in-place, so we need a bunch of copies here
#             e = entries.copy()
#             s = c(grid, e)
#             ufunc.at(s, [0, 1], *args)
#             ufunc.at(e, [0, 1], *args)
#             self.assertArrayEq(s.entries, e)

#         def f(c):
#             do_one_ufunc(c, np.exp)
#             do_one_ufunc(c, np.add, 1)
#             do_one_ufunc(c, np.add, [3, 3])

#         self.rep3(f)

#     def test_getitem(self):
#         grid = (np.linspace(0, 1, 5), )
#         entries = np.sin(grid)

#         def f(c):
#             series = c(grid, entries)
#             subseries = series[2:4]
#             self.assertArrayEq(subseries.grid, grid[2:4])
#             self.assertArrayEq(subseries.entries, entries[2:4])

#         self.rep3(f)

#     def test_setitem(self):
#         grid = (np.linspace(0, 1, 5), )
#         entries = np.sin(grid)

#         def f(c):
#             series = c(grid.copy(), entries.copy())

#             # you can set a slice of a series to another series
#             subseries = c(grid[2:4], -np.ones(2))
#             series[2:4] = subseries
#             e = entries.copy()
#             e[2:4] = -1.0
#             self.assertArrayEq(series.entries, e)

#             # this works also if the grids don't match (!)
#             series = c(grid.copy(), entries.copy())
#             subseries = c(np.arange(2), -np.ones(2))
#             series[2:4] = subseries
#             self.assertArrayEq(series.entries, e)

#             # NOTE commenting the following block out because I
#             # removed this behavior for compatibility with Phasor.
#             # This change may be reversed later.
#             # # you can also set a slice to a float or to an array of
#             # # appropriate size, like with numpy arrays
#             # series = c(grid.copy(), entries.copy())
#             # series[2:4] = -1.0
#             # self.assertArrayEq(series.entries, e)

#             # series = c(grid.copy(), entries.copy())
#             # series[2:4] = -np.ones(2)
#             # self.assertArrayEq(series.entries, e)

#             # wrong size -> error
#             with self.assertRaises(ValueError):
#                 subseries = c(grid[2:5], -np.ones(3))
#                 series[2:4] = subseries

#         self.rep3(f)

#     def test_add(self):
#         # test .add() and .iadd()
#         grid = (np.linspace(0, 1, 5), )
#         entries = np.sin(grid)

#         def f(c):
#             series = c(grid.copy(), entries.copy())
#             subseries = c(grid[1:3].copy(), np.ones(2))

#             # add one to two contiguous elements of the series
#             added = series.add(subseries, slice(1, 3))
#             e = entries.copy()
#             e[1:3] += 1.0
#             self.assertArrayEq(added.entries, e)

#             # that should not have changed either series in-place
#             self.assertArrayEq(series.grid, grid)
#             self.assertArrayEq(series.entries, entries)
#             self.assertArrayEq(subseries.grid, grid[1:3])
#             self.assertArrayEq(subseries.entries, np.ones(2))

#             # *now* do it in-place
#             added = series.add(subseries, slice(1, 3), inplace=True)
#             self.assertArrayEq(series.entries, e)
#             self.assertIs(added, series)

#             # again check that subseries wasn't changed
#             self.assertArrayEq(subseries.grid, grid[1:3])
#             self.assertArrayEq(subseries.entries, np.ones(2))

#             # result should be the same as calling .iadd()
#             series2 = c(grid.copy(), entries.copy())
#             series2.iadd(subseries, slice(1, 3))
#             self.assertArrayEq(series2.grid, series.grid)
#             self.assertArrayEq(series2.entries, series.entries)
#             self.assertArrayEq(subseries.grid, grid[1:3])
#             self.assertArrayEq(subseries.entries, np.ones(2))

#         self.rep3(f)

#     def test_automatic_iadd(self):
#         # test .__iadd__(), not the same as .iadd() because it also
#         # finds the subgrid automatically
#         grid = (np.linspace(0, 1, 5), )
#         entries = np.sin(grid)

#         def f(c):
#             series = c(grid.copy(), entries.copy())
#             subseries = c(grid[1:3].copy(), np.ones(2))
#             e = entries.copy()
#             e[1:3] += 1.0

#             series += subseries
#             self.assertArrayEq(series.entries, e)

#             # should not have changed subseries
#             self.assertArrayEq(subseries.grid, grid[1:3])
#             self.assertArrayEq(subseries.entries, np.ones(2))

#         self.rep3(f)

#     def test_get_subset(self):
#         grid = np.arange(5)
#         entries = np.sin(grid)

#         def f(c):
#             # we can generate a subset using (min, max) interval in the grid
#             series = c(grid.copy(), entries.copy())
#             sub1 = series.get_subset(interval=(0.9, 2.1))  # copy=True
#             self.assertArrayEq(sub1.grid, grid[1:3])
#             self.assertArrayEq(sub1.entries, entries[1:3])
#             series.entries[1] = 0.0
#             self.assertNotAlmostEqual(sub1.entries[0], 0.0)  # detect copying

#             # we can also use a slice directly
#             series = c(grid.copy(), entries.copy())
#             sub2 = series.get_subset(slice=slice(1, 3))  # copy=True
#             self.assertArrayEq(sub2.grid, grid[1:3])
#             self.assertArrayEq(sub2.entries, entries[1:3])
#             series.entries[1] = 0.0
#             self.assertNotAlmostEqual(sub2.entries[0], 0.0)  # detect copying

#             # cannot use both
#             with self.assertRaises(ValueError):
#                 series.get_subset(interval=(0.9, 2.1), slice=slice(1, 3))

#             # now the same tests, but with copy=False
#             series = c(grid.copy(), entries.copy())
#             sub3 = series.get_subset(interval=(0.9, 2.1), copy=False)
#             self.assertArrayEq(sub3.grid, grid[1:3])
#             self.assertArrayEq(sub3.entries, entries[1:3])
#             series.entries[1] = 0.0
#             self.assertEqual(sub3.entries[0], 0.0)  # detect copying

#             series = c(grid.copy(), entries.copy())
#             sub4 = series.get_subset(slice=slice(1, 3), copy=False)
#             self.assertArrayEq(sub4.grid, grid[1:3])
#             self.assertArrayEq(sub4.entries, entries[1:3])
#             series.entries[1] = 0.0
#             self.assertEqual(sub4.entries[0], 0.0)  # detect copying

#             with self.assertRaises(ValueError):
#                 series.get_subset(interval=(0.9, 2.1), slice=slice(1, 3), copy=False)

#         self.rep3(f)

#     def test_get_embedded(self):
#         grid_large = np.arange(10, dtype=float)  # 0..9
#         grid_small = np.arange(3, 7, dtype=float)  # 3..6
#         entries = np.sin(grid_small)

#         def f(c):
#             small_series = c(grid_small, entries)

#             # can embed into large grid as an array
#             embedded = small_series.get_embedded(grid_large)
#             self.assertEqual(type(small_series), type(embedded))
#             self.assertArrayEq(embedded.grid, grid_large)
#             self.assertArrayEq(embedded.entries[:3], np.zeros(3))
#             self.assertArrayEq(embedded.entries[3:7], entries)
#             self.assertArrayEq(embedded.entries[7:], np.zeros(3))

#             # same, but the large grid is a linspace
#             embedded = small_series.get_embedded(Linspace.make(grid_large))
#             self.assertEqual(type(small_series), type(embedded))
#             self.assertArrayEq(embedded.grid, grid_large)
#             self.assertArrayEq(embedded.entries[:3], np.zeros(3))
#             self.assertArrayEq(embedded.entries[3:7], entries)
#             self.assertArrayEq(embedded.entries[7:], np.zeros(3))

#         self.rep3(f)


# class TestFS(MixinTestUtils, unittest.TestCase):
#     "Test FrequencySeries-specific behaviors"

#     def test_angle(self):
#         grid = (np.linspace(0, 20, 50),)  # angles go above 2pi: need unwrap
#         freq = grid[0]
#         entries = np.exp(1j * freq).reshape(
#             1, len(freq), 1, 1, 1
#         )  # use the grid itself as angles

#         series = FrequencySeries(grid, entries)
#         angle = series.angle()
#         self.assertIsInstance(angle, FrequencySeries)
#         self.assertArrayEq(angle.grid, grid)
#         self.assertAllClose(angle.entries, grid)

#         # take a subseries using parts of the grid above 2pi but below 4pi
#         idx = np.searchsorted(grid, 2 * np.pi, side="right")
#         subseries = series[idx:]
#         self.assertGreater(len(subseries.grid), 0)  # TODO add __len__ to _Series

#         # the angle now should be off by a 2pi
#         subangle = subseries.angle()
#         self.assertArrayEq(subangle.grid, subseries.grid)
#         self.assertAllClose(subangle.entries, grid[idx:] - 2 * np.pi)

#     def test_real_imag(self):
#         grid = (np.linspace(0, 1, 5),)
#         entries = np.exp(1j * grid)

#         series = FrequencySeries(grid, entries)
#         real = series.real  # NOTE this is a property, it executes code
#         imag = series.imag  # same

#         self.assertArrayEq(grid, real.grid)
#         self.assertArrayEq(grid, imag.grid)

#         self.assertAllClose(entries.real, real.entries)  # type: ignore
#         self.assertAllClose(entries.imag, imag.entries)  # type: ignore

#     def test_aliases(self):
#         grid = (np.linspace(0, 1, 5),)
#         freq = grid[0]
#         entries = np.sin(grid)

#         series = FrequencySeries(grid, entries)

#         self.assertEqual(series.resolution, series.df)
#         self.assertEqual(series.df, freq[1] - freq[0])
#         self.assertIs(series.frequencies, series.grid[0])

#     def test_irfft(self):
#         # NOTE this test is not very deep, more thorough consistency
#         # checks in TestTSFSConsistency

#         # NOTE odd N would yield spurious warning
#         N, dt = 6, 1.5869
#         tgrid = np.arange(N) * dt
#         fgrid = np.fft.rfftfreq(N, dt)
#         entries = np.zeros_like(fgrid)
#         entries[0] = 1.0
#         expected_tseries = 1 / (N * dt) * np.ones(N)

#         fseries = FrequencySeries(fgrid, entries)
#         tseries = fseries.irfft(tgrid)
#         self.assertIsInstance(tseries, TimeSeries)
#         self.assertArrayEq(tseries.grid, tgrid)
#         self.assertAllClose(tseries, expected_tseries)

#         # does tapering at least seem to work?
#         for win in TAPERING_WINDOWS:
#             tseries2 = fseries.irfft(tgrid, win)
#             self.assertArrayEq(tseries2.grid, tgrid)
#             self.assertFalse(np.allclose(tseries2.entries, expected_tseries))

#     def test_time_shift(self):
#         grid = (np.linspace(0, 1, 5),)
#         entries = np.sin(grid)
#         series = FrequencySeries(grid, entries)
#         shifted = series.get_time_shifted(1.5)

#         # shift should not change freq grid
#         self.assertArrayEq(series.grid, shifted.grid)

#         # shift should be invertible
#         shiftedback = shifted.get_time_shifted(-1.5)
#         self.assertAllClose(series.entries, shiftedback.entries)

#     def test_get_plotter(self):
#         grid = (np.linspace(0, 1, 5),)
#         entries = np.sin(grid)
#         series = FrequencySeries(grid, entries)
#         self.assertIsInstance(series.get_plotter(), FSPlotter)


# class TestTS(MixinTestUtils, unittest.TestCase):
#     def test_aliases(self):
#         grid = (np.linspace(0, 1, 5),)
#         entries = np.sin(grid)

#         series = TimeSeries(grid, entries)

#         self.assertEqual(series.resolution, series.dt)
#         self.assertEqual(series.dt, grid[1] - grid[0])
#         self.assertIs(series.times, series.grid)

#     def test_rfft(self):
#         N, dt = 6, 1.576439
#         tgrid = dt * np.arange(N)
#         fgrid = np.fft.rfftfreq(N, dt)
#         entries = np.cos(2 * np.pi * fgrid[1] * tgrid)
#         expected_fseries = np.zeros_like(fgrid, dtype=complex)
#         expected_fseries[1] = (N * dt) / 2

#         tseries = TimeSeries(tgrid, entries)
#         fseries = tseries.rfft()
#         self.assertArrayEq(fseries.grid, fgrid)
#         self.assertAllClose(fseries.entries, expected_fseries)

#         # does tapering at least look like it works?
#         for win in TAPERING_WINDOWS:
#             fseries2 = tseries.rfft(win)
#             self.assertArrayEq(fseries2.grid, fgrid)
#             self.assertFalse(np.allclose(fseries2.entries, expected_fseries))

#     def test_get_plotter(self):
#         grid = (np.linspace(0, 1, 5),)
#         entries = np.sin(grid)
#         series = TimeSeries(grid, entries)
#         self.assertIsInstance(series.get_plotter(), TSPlotter)

#     def test_stfft(self):
#         N, dt = 20, 0.1
#         tgrid = dt * np.arange(N)
#         seglen, hop = 5, 4
#         nyquist = 1 / (2 * dt)
#         f0 = 2
#         assert f0 < nyquist
#         entries = np.sin(2 * np.pi * f0 * tgrid)
#         win = np.ones(seglen)  # square window

#         series = TimeSeries(tgrid, entries)
#         tfseries = series.stfft(win, hop)

#         self.assertIsInstance(tfseries, TimeFrequency)
#         # TODO test other properties. This is difficult because
#         # ordinarily STFTs are noninvertible


# class TestTSFSConsistency(MixinTestUtils, unittest.TestCase):
#     "Test that TimeSeries and FrequencySeries that should be the same really are the same"

#     def setUp(self):
#         self.time_grid = np.linspace(0, 1, 100)
#         dt = self.time_grid[1] - self.time_grid[0]
#         self.entries = np.sin(2 * np.pi * self.time_grid) + 2
#         self.time_series = TimeSeries(grid=self.time_grid, entries=self.entries)

#         self.freq_grid = np.fft.rfftfreq(len(self.time_grid), d=dt)
#         self.freq_entries = np.fft.rfft(self.entries * dt)
#         self.freq_series = FrequencySeries(
#             grid=self.freq_grid, entries=self.freq_entries
#         )

#     def test_basics(self):
#         self.assertEqual(self.time_series.entries.dtype, np.float64)
#         self.assertEqual(self.freq_series.entries.dtype, np.complex128)

#     def test_fft_methods(self):
#         result = self.time_series.rfft()
#         self.assertAllClose(result.entries, self.freq_entries)

#         result = self.freq_series.irfft(self.time_grid)
#         self.assertAllClose(result.entries, self.entries)

#     def test_round_trip(self):
#         tseries2 = self.time_series.rfft().irfft(self.time_grid)
#         self.assertAllClose(self.time_series.entries, tseries2.entries)

#         fseries2 = self.freq_series.irfft(self.time_grid).rfft()
#         self.assertAllClose(self.freq_series.entries, fseries2.entries)

#     def test_time_shift(self):
#         shift = 0.1
#         result = self.freq_series.get_time_shifted(shift)
#         expected_entries = self.freq_entries * np.exp(
#             -2j * np.pi * self.freq_grid * shift
#         )
#         self.assertAllClose(result.entries, expected_entries)


# class TestPhasor(unittest.TestCase):
#     def setUp(self):
#         self.frequencies = np.array([1.0, 2.0, 3.0])
#         self.amplitudes = np.arange(1, 4) * np.exp(1j * np.pi / 4)
#         self.phases = np.array([0.0, np.pi / 4, np.pi / 2])
#         self.phasor = Phasor.make(self.frequencies, self.amplitudes, self.phases)

#     def test_reim_to_cplx(self):
#         real_parts = np.array([1.0, 0.0, -1.0])
#         imag_parts = np.array([0.0, 1.0, 0.0])
#         expected = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
#         result = Phasor.reim_to_cplx(real_parts, imag_parts)
#         assert_array_almost_equal(result, expected)

#     def test_cplx_to_reim(self):
#         cplx = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
#         expected_real_parts = np.array([1.0, 0.0, -1.0])
#         expected_imag_parts = np.array([0.0, 1.0, 0.0])
#         result_real_parts, result_imag_parts = Phasor.cplx_to_reim(cplx)
#         assert_array_almost_equal(result_real_parts, expected_real_parts)
#         assert_array_almost_equal(result_imag_parts, expected_imag_parts)

#     def test_phasor_to_cplx(self):
#         amplitudes = np.array([1.0, 1.0, 1.0])
#         phases = np.array([0.0, np.pi / 2, np.pi])
#         expected = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j])
#         result = Phasor.phasor_to_cplx(amplitudes, phases)
#         assert_array_almost_equal(result, expected)

#     def test_get_interpolated(self):
#         def dummy_interpolator(x, y):
#             return lambda z: np.interp(z, x, y)

#         new_frequencies = np.array([1.5, 2.5])
#         interpolated_sequence = self.phasor.get_interpolated(
#             new_frequencies, dummy_interpolator
#         )
#         expected_amplitudes = np.array([1.5, 2.5]) * np.exp(1j * np.pi / 4)
#         expected_phases = np.array([np.pi / 8, 3 * np.pi / 8])
#         assert_array_almost_equal(interpolated_sequence.amplitudes, expected_amplitudes)
#         assert_array_almost_equal(interpolated_sequence.phases, expected_phases)

#     def test_to_freq_series(self):
#         freq_series = self.phasor.to_frequency_series()
#         expected_entries = np.array(
#             [
#                 np.exp(1j * np.pi / 4),
#                 2 * np.exp(1j * np.pi / 2),
#                 3 * np.exp(3j * np.pi / 4),
#             ]
#         )
#         assert_array_almost_equal(freq_series.entries, expected_entries)


# class TestTimeFrequency(MixinTestUtils, unittest.TestCase):
#     def setUp(self):
#         self.Nt, self.Nf = 5, 4
#         # TimeFrequency is not necessarily critically sampled, so the
#         # relation dF * dT = 1/2 does not necessarily hold
#         self.dT, self.dF = 0.13289, 0.5493
#         self.tgrid = self.dT * np.arange(self.Nt)
#         self.fgrid = self.dF * np.arange(self.Nf)

#         self.entries0 = np.zeros((self.Nf, self.Nt))
#         self.entries1 = np.ones((self.Nf, self.Nt))
#         self.entries2 = self.entries1 * 1j

#         self.series0 = TimeFrequency(self.tgrid, self.fgrid, self.entries0)
#         self.series1 = TimeFrequency(self.tgrid, self.fgrid, self.entries1)
#         self.series2 = TimeFrequency(self.tgrid, self.fgrid, self.entries2)

#     def test_init(self):
#         # must accept grid and entries as positional or kw arguments
#         series1 = TimeFrequency(
#             times=self.tgrid, frequencies=self.fgrid, entries=self.entries1
#         )
#         series2 = TimeFrequency(self.tgrid, self.fgrid, self.entries1)

#         # the results should match
#         self.assertArrayEq(series1.times, self.tgrid)
#         self.assertArrayEq(series1.times, series2.times)
#         self.assertArrayEq(series1.frequencies, self.fgrid)
#         self.assertArrayEq(series1.frequencies, series2.frequencies)
#         self.assertArrayEq(series1.entries, self.entries1)
#         self.assertArrayEq(series1.entries, series2.entries)

#         # there are no other kwargs
#         with self.assertRaises(TypeError):
#             # pylint: disable=unexpected-keyword-arg
#             TimeFrequency(self.tgrid, self.fgrid, self.entries1, other=None)  # type: ignore

#     def test_properties(self):
#         self.assertEqual(self.series2.dT, self.dT)
#         self.assertEqual(self.series2.dF, self.dF)

#     def test_get_subset(self):
#         big = self.series2

#         # we can generate a subset using (min, max) interval in the grid
#         assert self.Nt > 2 and self.Nf > 3
#         tmin, tmax = 0.99 * self.dT, 2.01 * self.dT
#         fmin, fmax = 0.99 * self.dF, 3.01 * self.dF
#         expected_subtgrid = self.tgrid[1:3]
#         expected_subfgrid = self.fgrid[1:4]
#         small = big.get_subset(time_interval=(tmin, tmax), freq_interval=(fmin, fmax))

#         self.assertEqual(small.entries.shape, (3, 2))
#         self.assertArrayEq(small.times, expected_subtgrid)
#         self.assertArrayEq(small.frequencies, expected_subfgrid)

#     def test_create_like(self):
#         old = self.series1
#         new_entries = 2 * self.entries1
#         new = old.create_like(new_entries)

#         self.assertArrayEq(old.times, new.times)
#         self.assertArrayEq(old.frequencies, new.frequencies)
#         self.assertArrayEq(new_entries, new.entries)

#     def test_get_plotter(self):
#         p = self.series2.get_plotter()
#         self.assertIsInstance(p, TimeFrequencyPlotter)

#     def test_arithmetic_ops(self):
#         entries1, entries2 = self.entries1, self.entries2
#         series1, series2 = self.series1, self.series2
#         self.assertArrayEq((series1 + series2).entries, entries1 + entries2)
#         self.assertArrayEq((series1 - series2).entries, entries1 - entries2)
#         self.assertArrayEq((series1 * series2).entries, entries1 * entries2)
#         self.assertArrayEq((series1 / series2).entries, entries1 / entries2)
#         self.assertArrayEq((series1 * 2).entries, entries1 * 2)
#         self.assertArrayEq((2 * series1).entries, entries1 * 2)
#         self.assertArrayEq((series1 / 2).entries, entries1 / 2)
#         self.assertArrayEq((2 / series2).entries, 2 / entries2)

#     def test_unary_ops(self):
#         entries_positive = self.entries1
#         series_positive = self.series1
#         entries = self.entries2  # complex
#         series = self.series2

#         # the following all come from lib.mixins.NDArrayMixin
#         self.assertArrayEq((-series).entries, -entries)
#         self.assertArrayEq((series.square()).entries, np.square(entries))
#         self.assertArrayEq((series.exp()).entries, np.exp(entries))
#         self.assertArrayEq((series_positive.sqrt()).entries, np.sqrt(entries_positive))
#         self.assertArrayEq((series.conj()).entries, np.conj(entries))
#         self.assertArrayEq((series.abs()).entries, np.abs(entries))

#     def test_ufunc_call(self):
#         entries0, entries1, entries2 = self.entries0, self.entries1, self.entries2
#         series0, series1, series2 = self.series0, self.series1, self.series2

#         # binary ops
#         for ufunc in [np.fmod, np.greater, np.maximum, np.copysign]:
#             self.assertArrayEq(
#                 (ufunc(series0, series1)).entries, ufunc(entries0, entries1)
#             )

#         # unary ops
#         for ufunc in [np.exp, np.isfinite]:
#             self.assertArrayEq(ufunc(series0).entries, ufunc(entries0))
#             self.assertArrayEq(ufunc(series2).entries, ufunc(entries2))


# class TestWDM(MixinTestUtils, unittest.TestCase):
#     def setUp(self):
#         # We use relatively big (and even) numbers because WDM breaks down
#         # at small values of Nt and Nf, and this implementation has issues
#         # with odd numbers
#         self.Nt, self.Nf = 20, 16

#         # WDM must be critically sampled: dF * dT = 1/2
#         self.dT = 0.12891289
#         self.dF = 1 / (2 * self.dT)
#         self.tgrid = self.dT * np.arange(self.Nt)  # t0 is zero!
#         self.fgrid = self.dF * np.arange(self.Nf)  # f0 is zero!

#         self.entries0 = np.zeros((self.Nf, self.Nt))
#         self.entries1 = np.ones((self.Nf, self.Nt))
#         self.entries2 = (
#             np.outer(
#                 np.ones_like(self.fgrid), np.where(self.tgrid == self.dT, 1.0, 0.0)
#             )
#             + 0.01
#         )
#         self.entries3 = np.outer(
#             np.where(self.fgrid == self.dF, 1.0, 0.0), np.ones_like(self.tgrid)
#         )
#         self.entries4 = np.outer(np.cos(self.fgrid), np.sin(self.tgrid))

#         self.series0 = WDM(self.tgrid, self.fgrid, self.entries0)
#         self.series1 = WDM(self.tgrid, self.fgrid, self.entries1)
#         self.series2 = WDM(self.tgrid, self.fgrid, self.entries2)
#         self.series3 = WDM(self.tgrid, self.fgrid, self.entries3)
#         self.series4 = WDM(self.tgrid, self.fgrid, self.entries4)
#         self.serieslist = [
#             self.series0,
#             self.series1,
#             self.series2,
#             self.series3,
#             self.series4,
#         ]

#     def test_init(self):
#         # must accept grid and entries as positional or kw arguments
#         series1 = WDM(times=self.tgrid, frequencies=self.fgrid, entries=self.entries1)
#         series2 = WDM(self.tgrid, self.fgrid, self.entries1)

#         # the results should match
#         self.assertArrayEq(series1.times, self.tgrid)
#         self.assertArrayEq(series1.times, series2.times)
#         self.assertArrayEq(series1.frequencies, self.fgrid)
#         self.assertArrayEq(series1.frequencies, series2.frequencies)
#         self.assertArrayEq(series1.entries, self.entries1)
#         self.assertArrayEq(series1.entries, series2.entries)

#         # there are no other kwargs
#         with self.assertRaises(TypeError):
#             # pylint: disable=unexpected-keyword-arg
#             WDM(self.tgrid, self.fgrid, self.entries1, other=None)  # type: ignore

#     # NOTE very slow test
#     def test_roundtrip(self):
#         # to frequencyseries and back
#         for wdm in self.serieslist:
#             fseries = wdm.to_freqseries()
#             new_wdm = fseries.to_WDM(Nf=self.Nf, Nt=self.Nt)
#             self.assertAllClose(wdm.times, new_wdm.times)
#             self.assertAllClose(wdm.frequencies, new_wdm.frequencies)
#             self.assertAllClose(wdm.entries, new_wdm.entries, atol=1e-7, rtol=1e-7)

#     def test_properties(self):
#         s, e, dF, dT, Nf, Nt = (
#             self.series2,
#             self.entries2,
#             self.dF,
#             self.dT,
#             self.Nf,
#             self.Nt,
#         )
#         self.assertAlmostEqual(s.dT, dT)
#         self.assertAlmostEqual(s.dF, dF)
#         self.assertAlmostEqual(s.Nf, e.shape[0])
#         self.assertAlmostEqual(s.Nt, e.shape[1])
#         self.assertAlmostEqual(s.ND, Nf * Nt)
#         self.assertAlmostEqual(s.duration, Nt * dT)
#         self.assertAlmostEqual(s.duration, Nt * Nf * s.dt)
#         self.assertAlmostEqual(s.df, 1 / s.duration)
#         self.assertAlmostEqual(s.sample_rate, s.fs)
#         self.assertAlmostEqual(s.fs, 1 / s.dt)
#         self.assertAlmostEqual(s.nyquist, s.fs / 2)
#         self.assertEqual(s.shape, e.shape)

#     def test_get_subset(self):
#         big = self.series2

#         # we can generate a subset using (min, max) interval in the grid
#         assert self.Nt > 2 and self.Nf > 3
#         tmin, tmax = 0.99 * self.dT, 2.01 * self.dT
#         fmin, fmax = 0.99 * self.dF, 3.01 * self.dF
#         expected_subtgrid = self.tgrid[1:3]
#         expected_subfgrid = self.fgrid[1:4]
#         small = big.get_subset(time_interval=(tmin, tmax), freq_interval=(fmin, fmax))

#         self.assertEqual(small.entries.shape, (3, 2))
#         self.assertArrayEq(small.times, expected_subtgrid)
#         self.assertArrayEq(small.frequencies, expected_subfgrid)

#     def test_create_like(self):
#         old = self.series1
#         new_entries = 2 * self.entries1
#         new = old.create_like(new_entries)

#         self.assertArrayEq(old.times, new.times)
#         self.assertArrayEq(old.frequencies, new.frequencies)
#         self.assertArrayEq(new_entries, new.entries)

#     def test_arithmetic_ops(self):
#         entries1, entries2 = self.entries1, self.entries2
#         series1, series2 = self.series1, self.series2
#         self.assertArrayEq((series1 + series2).entries, entries1 + entries2)
#         self.assertArrayEq((series1 - series2).entries, entries1 - entries2)
#         self.assertArrayEq((series1 * series2).entries, entries1 * entries2)
#         self.assertArrayEq((series1 / series2).entries, entries1 / entries2)
#         self.assertArrayEq((series1 * 2).entries, entries1 * 2)
#         self.assertArrayEq((2 * series1).entries, entries1 * 2)
#         self.assertArrayEq((series1 / 2).entries, entries1 / 2)
#         self.assertArrayEq((2 / series2).entries, 2 / entries2)

#     def test_unary_ops(self):
#         entries_positive = self.entries1
#         series_positive = self.series1
#         entries = self.entries2  # complex
#         series = self.series2

#         # the following all come from lib.mixins.NDArrayMixin
#         self.assertArrayEq((-series).entries, -entries)
#         self.assertArrayEq((series.square()).entries, np.square(entries))
#         self.assertArrayEq((series.exp()).entries, np.exp(entries))
#         self.assertArrayEq((series_positive.sqrt()).entries, np.sqrt(entries_positive))
#         self.assertArrayEq((series.conj()).entries, np.conj(entries))
#         self.assertArrayEq((series.abs()).entries, np.abs(entries))

#     def test_ufunc_call(self):
#         entries0, entries1, entries2 = self.entries0, self.entries1, self.entries2
#         series0, series1, series2 = self.series0, self.series1, self.series2

#         # binary ops
#         for ufunc in [np.fmod, np.greater, np.maximum, np.copysign]:
#             self.assertArrayEq(
#                 (ufunc(series0, series1)).entries, ufunc(entries0, entries1)
#             )

#         # unary ops
#         for ufunc in [np.exp, np.isfinite]:
#             self.assertArrayEq(ufunc(series0).entries, ufunc(entries0))
#             self.assertArrayEq(ufunc(series2).entries, ufunc(entries2))


if __name__ == "__main__":
    unittest.main()
