"""Tests for data containers with NumPy arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import tempfile
import unittest

import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit.containers.data import FSData, TSData, TimedFSData, WDMData, load_data
from typed_lisa_toolkit.containers.representations import TimeSeries
from tests._shared.noisemodel_helpers import build_fd_pair, build_wdm_pair
from tests._shared.waveforms_helpers import build_harmonic_projected_frequency_waveform


def _build_tsdata_numpy():
    times = np.linspace(0.0, 3.0, 8)
    x = np.array([0.0, 1.0, 0.5, -0.5, -1.0, -0.25, 0.75, 0.0])
    y = np.array([1.0, 0.0, -0.5, 0.25, 0.5, -0.75, 0.0, 1.0])
    data = TSData.from_dict(
        {
            "X": TimeSeries((times,), x[None, None, None, None, :]),
            "Y": TimeSeries((times,), y[None, None, None, None, :]),
        }
    )
    return times, x, y, data


class TestDataContainersNumpy(unittest.TestCase):
    def test_tsdata_times_dt_and_get_frequencies(self):
        times, _, _, tsdata = _build_tsdata_numpy()

        npt.assert_allclose(np.asarray(tsdata.times), times)
        self.assertAlmostEqual(tsdata.dt, times[1] - times[0])
        npt.assert_allclose(
            np.asarray(tsdata.get_frequencies()),
            np.fft.rfftfreq(len(times), d=tsdata.dt),
        )

    def test_fsdata_to_wdmdata_and_back_preserves_channels(self):
        _, _, _, tsdata = _build_tsdata_numpy()
        fsdata = tsdata.to_fsdata(keep_times=False)

        wdmdata = fsdata.to_WDMdata(Nf=2, Nt=2)
        recovered = wdmdata.to_fsdata()

        self.assertIsInstance(wdmdata, WDMData)
        self.assertIsInstance(recovered, FSData)
        self.assertEqual(wdmdata.channel_names, fsdata.channel_names)
        self.assertEqual(recovered.channel_names, fsdata.channel_names)
        self.assertEqual(np.asarray(wdmdata["X"].entries).shape[-2:], (2, 2))
        self.assertEqual(
            np.asarray(recovered.get_kernel()).shape[-1],
            np.asarray(recovered.frequencies).shape[0],
        )

    def test_wdm_fs_wdm_roundtrip_preserves_grid_and_entries(self):
        wdmdata = build_wdm_pair(np)["left"]
        nf, nt = wdmdata["X"].Nf, wdmdata["X"].Nt

        fsdata = wdmdata.to_fsdata()
        roundtrip = fsdata.to_WDMdata(Nf=nf, Nt=nt)

        self.assertIsInstance(roundtrip, WDMData)
        self.assertEqual(roundtrip.channel_names, wdmdata.channel_names)
        npt.assert_allclose(np.asarray(roundtrip["X"].times), np.asarray(wdmdata["X"].times))
        npt.assert_allclose(
            np.asarray(roundtrip["X"].frequencies),
            np.asarray(wdmdata["X"].frequencies),
        )
        npt.assert_allclose(
            np.asarray(roundtrip.get_kernel()),
            np.asarray(wdmdata.get_kernel()),
            rtol=1e-7,
            atol=1e-7,
        )

    def test_fsdata_frequencies_and_df(self):
        times, _, _, tsdata = _build_tsdata_numpy()
        fsdata = tsdata.to_fsdata(keep_times=False)

        frequencies = np.asarray(fsdata.frequencies)
        npt.assert_allclose(frequencies, np.fft.rfftfreq(len(times), d=tsdata.dt))
        self.assertAlmostEqual(float(fsdata.df), float(frequencies[1] - frequencies[0]))

    def test_pick_preserves_requested_order(self):
        case = build_fd_pair(np)

        picked = case["left"].pick(("Y", "X"))

        self.assertEqual(picked.channel_names, ("Y", "X"))
        self.assertEqual(np.asarray(picked["Y"].entries).shape[1], 1)
        npt.assert_allclose(
            np.asarray(picked["Y"].entries),
            np.asarray(case["left"]["Y"].entries),
        )

    def test_init_rejects_non_unit_harmonic_dimension(self):
        times = np.linspace(0.0, 1.0, 4)
        bad_repr = TimeSeries((times,), np.ones((1, 2, 2, 1, 4)))

        with self.assertRaises(ValueError):
            TSData(bad_repr, ("X", "Y"))

    def test_fsdata_set_times_drop_times_and_to_tsdata(self):
        case = build_fd_pair(np)
        times = np.linspace(0.0, 7.0, 8)

        timed = case["left"].set_times(times)
        recovered = timed.to_tsdata()

        self.assertIsInstance(timed, TimedFSData)
        self.assertIsInstance(timed.drop_times(), FSData)
        self.assertEqual(recovered.channel_names, case["left"].channel_names)
        npt.assert_allclose(np.asarray(recovered.times), times)
        self.assertEqual(np.asarray(recovered.get_kernel()).shape[-1], len(times))

    def test_fsdata_set_times_and_drop_times(self):
        times, _, _, tsdata = _build_tsdata_numpy()
        fsdata = tsdata.to_fsdata(keep_times=False)

        timed = fsdata.set_times(np.asarray(times))

        self.assertIsInstance(timed, TimedFSData)
        self.assertIsInstance(timed.drop_times(), FSData)
        npt.assert_allclose(np.asarray(timed.times), np.asarray(times))

    def test_tsdata_to_fsdata_uses_rfft_frequency_grid(self):
        times, _, _, tsdata = _build_tsdata_numpy()

        fsdata = tsdata.to_fsdata(keep_times=True)

        self.assertIsInstance(fsdata, TimedFSData)
        npt.assert_allclose(np.asarray(fsdata.times), times)
        npt.assert_allclose(np.asarray(fsdata.frequencies), np.fft.rfftfreq(len(times), tsdata.dt))

    def test_from_waveform_preserves_entries_and_channels(self):
        case = build_harmonic_projected_frequency_waveform(np)

        data = FSData.from_waveform(case["resp_22"])

        self.assertEqual(data.channel_names, case["resp_22"].channel_names)
        npt.assert_allclose(
            np.asarray(data.get_kernel()),
            np.asarray(case["resp_22"].get_kernel()),
        )

    def test_get_embedded_expands_frequency_grid(self):
        case = build_fd_pair(np)
        embedding = np.array([0.5, 1.0, 2.0, 4.0, 5.0])

        embedded = case["left"].get_embedded(embedding)
        got = np.asarray(embedded.get_kernel())
        source = np.asarray(case["left"].get_kernel())

        self.assertEqual(np.asarray(embedded.frequencies).shape[0], 5)
        npt.assert_allclose(got[..., 1:4], source)
        npt.assert_allclose(got[..., 0], 0.0)
        npt.assert_allclose(got[..., -1], 0.0)

    def test_save_and_load_dispatches_by_type(self):
        case = build_fd_pair(np)
        times = np.linspace(0.0, 7.0, 8)
        timed = case["left"].set_times(times)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            timed.save(handle.name)
            loaded = load_data(handle.name)

        self.assertIsInstance(loaded, TimedFSData)
        npt.assert_allclose(np.asarray(loaded.times), times)
        npt.assert_allclose(
            np.asarray(loaded.get_kernel()),
            np.asarray(timed.get_kernel()),
        )