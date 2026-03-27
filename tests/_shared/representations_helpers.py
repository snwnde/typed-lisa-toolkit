"""Shared helper builders and test mixins for representation backend tests."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit.containers.representations import (
    STFT,
    WDM,
    UniformFrequencySeries,
    Linspace,
    UniformTimeSeries,
    _check_entry_grid_compatibility,
    _take_subset,
)


def _randn_array(xp, shape):
    return xp.asarray(np.random.randn(*shape))


def build_canonical_representations(
    xp,
    *,
    n_batches,
    n_channels,
    n_harmonics,
    n_features,
    len_time,
    len_freq,
    tf_grid_order,
):
    freqs = xp.linspace(0, 1, len_freq)
    times = xp.linspace(0, 10, len_time)

    entries_fs = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_freq),
    )
    fs = UniformFrequencySeries(grid=(freqs,), entries=entries_fs)

    entries_ts = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_time),
    )
    ts = UniformTimeSeries(grid=(times,), entries=entries_ts)

    if tf_grid_order == "freq_time":
        entries_tf = _randn_array(
            xp,
            (n_batches, n_channels, n_harmonics, n_features, len_freq, len_time),
        )
        tf = STFT(grid=(freqs, times), entries=entries_tf)
    elif tf_grid_order == "time_freq":
        entries_tf = _randn_array(
            xp,
            (n_batches, n_channels, n_harmonics, n_features, len_time, len_freq),
        )
        tf = STFT(grid=(times, freqs), entries=entries_tf)
    else:
        raise ValueError(f"Unknown tf_grid_order: {tf_grid_order}")

    return {
        "freqs": freqs,
        "times": times,
        "entries_fs": entries_fs,
        "entries_ts": entries_ts,
        "entries_tf": entries_tf,
        "fs": fs,
        "ts": ts,
        "tf": tf,
    }


# ---------------------------------------------------------------------------
# Test mixins (no unittest.TestCase dependency; combine with it in concrete
# test classes via multiple inheritance).
# ---------------------------------------------------------------------------


class LinspaceExtraPropertiesMixin:
    """Backend-agnostic Linspace edge-case tests (no array operations)."""

    def test_shape_property(self):
        ls = Linspace(1.0, 0.5, 4)
        self.assertEqual(ls.shape, (4,))  # type: ignore[attr-defined]

    def test_stop_property(self):
        ls = Linspace(1.0, 0.5, 4)
        self.assertAlmostEqual(ls.stop, 2.5)  # type: ignore[attr-defined]

    def test_eq_raises_for_non_linspacelike(self):
        ls = Linspace(0.0, 1.0, 5)
        with self.assertRaises(TypeError):  # type: ignore[attr-defined]
            ls.__eq__(42)

    def test_eq_returns_false_for_step_mismatch(self):
        ls1 = Linspace(0.0, 1.0, 5)
        ls2 = Linspace(0.0, 2.0, 5)
        self.assertFalse(ls1 == ls2)  # type: ignore[attr-defined]

    def test_array_with_copy_false(self):
        ls = Linspace(0.0, 1.0, 5)
        arr = np.array(ls, copy=False)
        npt.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_getitem_invalid_type_raises(self):
        ls = Linspace(0.0, 1.0, 10)
        with self.assertRaises(TypeError):  # type: ignore[attr-defined]
            ls["bad"]  # type: ignore[index]

    def test_make_from_linspace_like(self):
        class _MockLinspaceLike:
            def __init__(self, start, step, num):
                self._start, self._step, self._num = start, step, num

            @property
            def start(self):
                return self._start

            @property
            def step(self):
                return self._step

            def __len__(self):
                return self._num

        mock = _MockLinspaceLike(2.0, 0.5, 8)
        ls = Linspace.make(mock)  # type: ignore[arg-type]
        self.assertIsInstance(ls, Linspace)  # type: ignore[attr-defined]
        self.assertAlmostEqual(ls.start, 2.0)  # type: ignore[attr-defined]
        self.assertAlmostEqual(ls.step, 0.5)  # type: ignore[attr-defined]
        self.assertEqual(ls.num, 8)  # type: ignore[attr-defined]


class HelperFunctionsMixin:
    """Mix-in for module-level helper function tests.

    Subclass must provide a class-level ``xp`` attribute (numpy or jax.numpy).
    """

    xp = None  # subclasses set: np or jnp

    def test_check_entry_grid_compatibility_raises_on_mismatch(self):
        xp = self.xp  # type: ignore[attr-defined]
        grid = (xp.asarray(np.linspace(0, 1, 10)),)
        entries = xp.asarray(np.ones((1, 1, 1, 1, 20)))
        with self.assertRaises(ValueError):  # type: ignore[attr-defined]
            _check_entry_grid_compatibility(grid, entries)

    def test_take_subset_slice_dimension_mismatch_raises(self):
        xp = self.xp  # type: ignore[attr-defined]
        grid = (xp.asarray(np.linspace(0, 1, 10)),)
        entries = xp.asarray(np.ones((1, 1, 1, 1, 10)))
        with self.assertRaises(ValueError):  # type: ignore[attr-defined]
            _take_subset(grid, entries, (slice(0, 5), slice(0, 3)))

    def test_take_subset_with_array_grid(self):
        xp = self.xp  # type: ignore[attr-defined]
        grid = (xp.asarray(np.linspace(0.0, 1.0, 20)),)
        entries = xp.asarray(np.arange(20, dtype=float))[None, None, None, None, :]
        new_grid, new_entries = _take_subset(grid, entries, (slice(5, 10),))
        npt.assert_allclose(np.asarray(new_grid[0]), np.asarray(grid[0][5:10]))
        npt.assert_allclose(
            np.asarray(new_entries[0, 0, 0, 0, :]),
            np.asarray(entries[0, 0, 0, 0, 5:10]),
        )

    def test_non_uniform_grid_stays_array(self):
        xp = self.xp  # type: ignore[attr-defined]
        non_uniform = xp.asarray(np.array([0.0, 1.0, 3.0, 7.0]))
        ts = UniformTimeSeries(
            grid=(non_uniform,), entries=xp.asarray(np.ones((1, 1, 1, 1, 4)))
        )
        self.assertNotIsInstance(ts.grid[0], Linspace)  # type: ignore[attr-defined]

    def test_axis_onset_and_end_from_plain_arrays(self):
        xp = self.xp  # type: ignore[attr-defined]
        freqs = xp.asarray(np.array([0.01, 0.02, 0.03, 0.04]))
        fs = UniformFrequencySeries(
            grid=(freqs,), entries=xp.asarray(np.ones((1, 1, 1, 1, 4)))
        )
        self.assertAlmostEqual(fs.f_min, 0.01)  # type: ignore[attr-defined]
        self.assertAlmostEqual(fs.f_max, 0.04)  # type: ignore[attr-defined]


class AdvancedRepresentationMethodsMixin:
    """Mix-in for UniformFrequencySeries/UniformTimeSeries/STFT method tests.

    Subclass must provide ``xp`` (numpy or jax.numpy).
    Tests that use only plain numpy arrays need no ``xp``.
    """

    xp = None  # subclasses set: np or jnp

    def test_frequency_series_get_time_shifted(self):
        xp = self.xp  # type: ignore[attr-defined]
        n, dt = 32, 1.0 / 128
        freqs = xp.asarray(np.fft.rfftfreq(n, d=dt))
        entries_fs = xp.asarray(np.fft.rfft(np.sin(2 * np.pi * np.arange(n) * dt)))[
            None, None, None, None, :
        ]
        fs = UniformFrequencySeries(grid=(freqs,), entries=entries_fs)
        shifted = fs.get_time_shifted(2 * dt)
        self.assertIsInstance(shifted, UniformFrequencySeries)  # type: ignore[attr-defined]
        self.assertEqual(shifted.entries.shape, fs.entries.shape)  # type: ignore[attr-defined]

    def test_frequency_series_angle(self):
        xp = self.xp  # type: ignore[attr-defined]
        freqs = xp.asarray(np.linspace(1e-4, 1e-2, 10))
        z = xp.asarray(np.exp(1j * np.linspace(0, 4 * np.pi, 10)))[
            None, None, None, None, :
        ]
        fs = UniformFrequencySeries(grid=(freqs,), entries=z)
        angles = fs.angle()
        self.assertIsInstance(angles, UniformFrequencySeries)  # type: ignore[attr-defined]
        self.assertEqual(angles.entries.shape, fs.entries.shape)  # type: ignore[attr-defined]

    def test_time_series_stfft(self):
        xp = self.xp  # type: ignore[attr-defined]
        n = 64
        times = Linspace(0.0, 1.0 / 128, n)
        signal = xp.asarray(np.sin(2 * np.pi * np.arange(n) / 16).astype(float))
        entries = signal[None, None, None, None, :]
        ts = UniformTimeSeries(grid=(times,), entries=entries)
        win = np.hanning(16).astype(float)
        stft = ts.stfft(win, hop=8)
        self.assertIsInstance(stft, STFT)  # type: ignore[attr-defined]
        self.assertEqual(stft.entries.ndim, 6)  # type: ignore[attr-defined]

    def test_stft_make_classmethod(self):
        times = np.linspace(0, 10, 100)
        freqs = np.linspace(0, 1, 50)
        entries = np.random.randn(1, 1, 1, 1, 100, 50)
        stft = STFT.make(times, freqs, entries)
        self.assertIsInstance(stft, STFT)  # type: ignore[attr-defined]
        npt.assert_allclose(np.array(stft.grid[1]), times, rtol=1e-10)
        npt.assert_allclose(np.array(stft.grid[0]), freqs, rtol=1e-10)

    def test_stft_times_and_frequencies_properties(self):
        freqs = np.linspace(0, 1, 50)
        times = np.linspace(0, 10, 100)
        entries = np.random.randn(1, 1, 1, 1, 50, 100)
        stft = STFT(grid=(freqs, times), entries=entries)
        npt.assert_allclose(np.array(stft.times), times)
        npt.assert_allclose(np.array(stft.frequencies), freqs)

    def test_series_repr_and_grid_shape(self):
        xp = self.xp  # type: ignore[attr-defined]
        freqs = Linspace(0.0, 1e-3, 20)
        entries = xp.asarray(np.ones((1, 1, 1, 1, 20)))
        fs = UniformFrequencySeries(grid=(freqs,), entries=entries)
        r = repr(fs)
        self.assertIn("UniformFrequencySeries", r)  # type: ignore[attr-defined]
        self.assertEqual(fs.grid_shape, (20,))  # type: ignore[attr-defined]


class WDMPropertiesAndMethodsMixin:
    """Mix-in for WDM property/method tests.

    Subclass must set ``self.wdm`` in ``setUp`` using the appropriate backend.
    """

    def test_nd_duration_sample_interval(self):
        wdm = self.wdm  # type: ignore[attr-defined]
        self.assertEqual(wdm.ND, wdm.Nf * wdm.Nt)  # type: ignore[attr-defined]
        self.assertAlmostEqual(wdm.duration, wdm.Nt * wdm.times.step)  # type: ignore[attr-defined]
        self.assertAlmostEqual(wdm.sample_interval, wdm.duration / wdm.ND)  # type: ignore[attr-defined]
        self.assertAlmostEqual(wdm.dt, wdm.sample_interval)  # type: ignore[attr-defined]

    def test_df_shape_sample_rate_nyquist(self):
        wdm = self.wdm  # type: ignore[attr-defined]
        self.assertAlmostEqual(wdm.df, 1.0 / wdm.duration)  # type: ignore[attr-defined]
        self.assertEqual(wdm.shape, (wdm.Nf, wdm.Nt))  # type: ignore[attr-defined]
        self.assertAlmostEqual(wdm.sample_rate, 1.0 / wdm.sample_interval)  # type: ignore[attr-defined]
        self.assertAlmostEqual(wdm.nyquist, wdm.sample_rate / 2.0)  # type: ignore[attr-defined]

    def test_is_critically_sampled(self):
        wdm = self.wdm  # type: ignore[attr-defined]
        result = wdm.is_critically_sampled()
        expected = bool(np.isclose(wdm.dT * wdm.dF, 0.5))
        self.assertEqual(bool(result), expected)  # type: ignore[attr-defined]

    def test_get_subset_time(self):
        wdm = self.wdm  # type: ignore[attr-defined]
        times_arr = np.asarray(wdm.times)
        t_mid = float(times_arr[len(times_arr) // 2])
        sub = wdm.get_subset(time_interval=(float(times_arr[0]), t_mid))
        self.assertIsInstance(sub, WDM)  # type: ignore[attr-defined]
        self.assertLess(sub.Nt, wdm.Nt)  # type: ignore[attr-defined]

    def test_get_subset_freq(self):
        wdm = self.wdm  # type: ignore[attr-defined]
        freqs_arr = np.asarray(wdm.frequencies)
        f_mid = float(freqs_arr[len(freqs_arr) // 2])
        sub = wdm.get_subset(freq_interval=(float(freqs_arr[0]), f_mid))
        self.assertIsInstance(sub, WDM)  # type: ignore[attr-defined]
        self.assertLess(sub.Nf, wdm.Nf)  # type: ignore[attr-defined]
