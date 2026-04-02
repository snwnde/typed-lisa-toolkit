"""Unit tests for shop/conversions.py (JAX backend)."""

import unittest
import warnings

import jax
from jax.random import t

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import shop, time_series
from typed_lisa_toolkit.types import FSData, TSData
from typed_lisa_toolkit.types import representations as reps


def _build_timeseries_jax(n: int = 8):
    times = jnp.linspace(0.0, 3.5, n, dtype=jnp.float64)
    entries = jnp.asarray(
        [0.0, 1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.1],
        dtype=jnp.float64,
    )
    ts = time_series(times, entries[None, None, None, None, :])
    return times, entries, ts


def _build_tsdata_jax(n: int = 8):
    times = jnp.linspace(0.0, 3.5, n, dtype=jnp.float64)
    x = jnp.asarray([0.0, 1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.1], dtype=jnp.float64)
    y = jnp.asarray([1.0, -0.5, 0.25, 0.0, 0.4, -0.2, 0.6, -0.8], dtype=jnp.float64)
    tsd = TSData.from_dict(
        {
            "X": time_series(times, x[None, None, None, None, :]),
            "Y": time_series(times, y[None, None, None, None, :]),
        }
    )
    return times, tsd


class TestConversionsJax(unittest.TestCase):
    def test_time2freq_timeseries_returns_representation(self):
        _, entries, ts = _build_timeseries_jax()

        fs = shop.time2freq(ts)

        self.assertIsInstance(fs, reps.UniformFrequencySeries)
        expected = jnp.fft.rfft(entries * ts.dt)
        npt.assert_allclose(np.asarray(fs.entries).squeeze(), np.asarray(expected))

    def test_time2freq_tsdata_keep_time_switches_type(self):
        times, tsd = _build_tsdata_jax()

        fs_no_time = shop.time2freq(tsd, keep_time=False)
        fs_with_time = shop.time2freq(tsd, keep_time=True)

        self.assertIsInstance(fs_no_time, FSData)
        self.assertNotIn("times", dir(fs_no_time))
        self.assertIsInstance(fs_with_time, FSData)
        npt.assert_allclose(np.asarray(fs_with_time.times), np.asarray(times))
        self.assertEqual(fs_with_time.channel_names, tsd.channel_names)

    def test_freq2time_frequencyseries_roundtrip(self):
        _, entries, ts = _build_timeseries_jax()

        fs = shop.time2freq(ts)
        recovered = shop.freq2time(fs, times=np.asarray(ts.times))

        self.assertIsInstance(recovered, reps.UniformTimeSeries)
        npt.assert_allclose(
            np.asarray(recovered.entries).squeeze(), np.asarray(entries), atol=1e-12
        )

    def test_freq2time_fsdata_returns_tsdata(self):
        times, tsd = _build_tsdata_jax()

        fsd = shop.time2freq(tsd, keep_time=False)
        recovered = shop.freq2time(fsd, times=np.asarray(times))

        self.assertIsInstance(recovered, TSData)
        self.assertEqual(recovered.channel_names, tsd.channel_names)
        npt.assert_allclose(np.asarray(recovered.times), np.asarray(times))
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()), np.asarray(tsd.get_kernel()), atol=1e-12
        )

    def test_freq2time_warns_for_denser_than_nyquist_times(self):
        _, _, ts = _build_timeseries_jax()
        fs = shop.time2freq(ts)

        dense_times = np.linspace(
            float(ts.times.start), float(ts.times.stop), 2 * len(ts.times) - 1
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = shop.freq2time(fs, times=dense_times)

        self.assertTrue(
            any("denser than the Nyquist limit" in str(item.message) for item in caught)
        )


if __name__ == "__main__":
    unittest.main()
