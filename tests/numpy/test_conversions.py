"""Unit tests for shop/conversions.py (NumPy backend)."""

from numpy._typing._array_like import NDArray


from numpy import float64


from typed_lisa_toolkit.types.data import TSData


import unittest
import warnings

import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import shop, time_series, linspace
from typed_lisa_toolkit.types import FSData, TSData, WDMData
from typed_lisa_toolkit.types import representations as reps


def _require_wdm_transform():
    try:
        import wdm_transform  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(
            "wdm_transform is required for WDM conversion tests"
        ) from exc


def _build_timeseries_numpy(n: int = 8):
    times = linspace(0.0, 3.5, n)
    entries = np.asarray(
        [0.0, 1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.1], dtype=np.float64
    )
    ts = time_series(times, entries[None, None, None, None, :])
    return times, entries, ts


def _build_tsdata_numpy(n: int = 8) -> tuple[NDArray[float64], TSData]:
    times = np.linspace(0.0, 3.5, n)
    x = np.asarray([0.0, 1.0, -0.5, 0.75, -1.25, 0.5, 0.25, -0.1], dtype=np.float64)
    y = np.asarray([1.0, -0.5, 0.25, 0.0, 0.4, -0.2, 0.6, -0.8], dtype=np.float64)
    tsd = TSData.from_dict(
        {
            "X": time_series(times, x[None, None, None, None, :]),
            "Y": time_series(times, y[None, None, None, None, :]),
        }
    )
    return times, tsd


class TestConversionsNumpy(unittest.TestCase):
    def test_time2freq_timeseries_returns_representation(self):
        _, entries, ts = _build_timeseries_numpy()

        fs = shop.time2freq(ts)

        self.assertIsInstance(fs, reps.UniformFrequencySeries)
        expected = np.fft.rfft(entries * ts.times.step)
        npt.assert_allclose(np.asarray(fs.entries).squeeze(), expected)

    def test_time2freq_tsdata_keep_time_switches_type(self):
        times, tsd = _build_tsdata_numpy()

        fs_no_time = shop.time2freq(tsd, keep_time=False)
        fs_with_time = shop.time2freq(tsd, keep_time=True)

        self.assertIsInstance(fs_no_time, FSData)
        self.assertNotIn("times", dir(fs_no_time))
        self.assertIsInstance(fs_with_time, FSData)
        npt.assert_allclose(np.asarray(fs_with_time.times), times)
        self.assertEqual(fs_with_time.channel_names, tsd.channel_names)

    def test_freq2time_frequencyseries_roundtrip(self):
        _, entries, ts = _build_timeseries_numpy()

        fs = shop.time2freq(ts)
        recovered = shop.freq2time(fs, times=np.asarray(ts.times))

        self.assertIsInstance(recovered, reps.UniformTimeSeries)
        npt.assert_allclose(
            np.asarray(recovered.entries).squeeze(), entries, atol=1e-12
        )

    def test_freq2time_fsdata_returns_tsdata(self):
        times, tsd = _build_tsdata_numpy()

        fsd = shop.time2freq(tsd, keep_time=False)
        recovered = shop.freq2time(fsd, times=times)

        self.assertIsInstance(recovered, TSData)
        self.assertEqual(recovered.channel_names, tsd.channel_names)
        npt.assert_allclose(np.asarray(recovered.times), times)
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()), np.asarray(tsd.get_kernel()), atol=1e-12
        )

    def test_freq2time_warns_for_denser_than_nyquist_times(self):
        _, _, ts = _build_timeseries_numpy()
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

    def test_time2wdm_roundtrip_timeseries(self):
        _require_wdm_transform()
        _, entries, ts = _build_timeseries_numpy()

        wdm = shop.time2wdm(ts, Nt=2, Nf=4)
        recovered = shop.wdm2time(wdm)

        self.assertIsInstance(wdm, reps.WDM)
        self.assertIsInstance(recovered, reps.UniformTimeSeries)
        npt.assert_allclose(np.asarray(recovered.times), np.asarray(ts.times))
        npt.assert_allclose(
            np.asarray(recovered.entries).squeeze(), entries, atol=1e-12
        )

    def test_time2wdm_roundtrip_tsdata(self):
        _require_wdm_transform()
        times, tsd = _build_tsdata_numpy()

        wdmdata = shop.time2wdm(tsd, Nt=2, Nf=4)
        recovered = shop.wdm2time(wdmdata)

        self.assertIsInstance(wdmdata, WDMData)
        self.assertIsInstance(recovered, TSData)
        self.assertEqual(recovered.channel_names, tsd.channel_names)
        npt.assert_allclose(np.asarray(recovered.times), times)
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()), np.asarray(tsd.get_kernel()), atol=1e-12
        )

    def test_freq2wdm_roundtrip_frequencyseries(self):
        _require_wdm_transform()
        _, _, ts = _build_timeseries_numpy()

        fs = shop.time2freq(ts)
        wdm = shop.freq2wdm(fs, Nt=2, Nf=4)
        recovered = shop.wdm2freq(wdm)

        self.assertIsInstance(wdm, reps.WDM)
        self.assertIsInstance(recovered, reps.UniformFrequencySeries)
        npt.assert_allclose(
            np.asarray(recovered.frequencies),
            np.asarray(fs.frequencies),
            err_msg="Frequencies do not match after round-trip conversion.",
        )
        npt.assert_allclose(
            np.asarray(recovered.entries),
            np.asarray(fs.entries),
            err_msg="Kernel entries do not match after round-trip conversion.",
        )

    def test_freq2wdm_roundtrip_fsdata(self):
        _require_wdm_transform()
        _, tsd = _build_tsdata_numpy()

        fsd = shop.time2freq(tsd, keep_time=False)
        wdmdata = shop.freq2wdm(fsd, Nt=2, Nf=4)
        recovered = shop.wdm2freq(wdmdata)

        self.assertIsInstance(wdmdata, WDMData)
        self.assertIsInstance(recovered, FSData)
        self.assertEqual(recovered.channel_names, fsd.channel_names)
        npt.assert_allclose(
            np.asarray(recovered.get_kernel()),
            np.asarray(fsd.get_kernel()),
            err_msg="Kernel entries do not match after round-trip conversion.",
        )
        npt.assert_allclose(
            np.asarray(recovered.frequencies),
            np.asarray(fsd.frequencies),
            err_msg="Frequencies do not match after round-trip conversion.",
        )


if __name__ == "__main__":
    unittest.main()
