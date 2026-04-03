"""Tests for data containers with JAX arrays."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import jax

jax.config.update("jax_enable_x64", True)
import h5py
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from tests._helpers import (
    DataAbstractBranchesMixin,
    build_fd_pair,
    build_fdata,
    build_harmonic_projected_frequency_waveform,
    build_wdm_pair,
)
from typed_lisa_toolkit import (
    load_data,
    load_ldc_data,
    shop,
)
from typed_lisa_toolkit.types import (
    FSData,
    STFTData,
    TimedFSData,
    TimeSeries,
    TSData,
    WDMData,
)


def _build_tsdata_jax():
    times = jnp.linspace(0.0, 3.0, 8, dtype=jnp.float64)
    x = jnp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25, 0.75, 0.0], dtype=jnp.float64)
    y = jnp.asarray([1.0, 0.0, -0.5, 0.25, 0.5, -0.75, 0.0, 1.0], dtype=jnp.float64)
    data = TSData.from_dict(
        {
            "X": TimeSeries((times,), x[None, None, None, None, :]),
            "Y": TimeSeries((times,), y[None, None, None, None, :]),
        }
    )
    return times, data


class TestDataContainersJAX(unittest.TestCase):
    def _assert_to_fsdata_deprecation(self, tsdata: TSData, *, keep_times: bool):
        with self.assertWarnsRegex(
            DeprecationWarning,
            r"UniformTimeSeries\.rfft",
        ):
            return tsdata.to_fsdata(keep_times=keep_times)

    def _assert_to_tsdata_deprecation(
        self,
        fs_like: FSData | TimedFSData,
        times: np.ndarray | None = None,
    ):
        with self.assertWarnsRegex(
            DeprecationWarning,
            r"UniformFrequencySeries\.irfft",
        ):
            if times is None:
                return fs_like.to_tsdata()
            return fs_like.to_tsdata(times)

    def test_tsdata_times_dt_and_get_frequencies(self):
        times, tsdata = _build_tsdata_jax()

        npt.assert_allclose(np.asarray(tsdata.times), np.asarray(times))
        self.assertAlmostEqual(tsdata.dt, float(times[1] - times[0]))
        npt.assert_allclose(
            np.asarray(tsdata.get_frequencies()),
            np.asarray(jnp.fft.rfftfreq(len(times), tsdata.dt)),
        )

    # def test_fsdata_to_wdmdata_and_back_preserves_channels(self):
    #     _, tsdata = _build_tsdata_jax()
    #     fsdata = tsdata.to_fsdata(keep_times=False)

    #     wdmdata = fsdata.to_wdm_data(Nf=2, Nt=2)
    #     recovered = wdmdata.to_fsdata()

    #     self.assertIsInstance(wdmdata, WDMData)
    #     self.assertIsInstance(recovered, FSData)
    #     self.assertEqual(wdmdata.channel_names, fsdata.channel_names)
    #     self.assertEqual(recovered.channel_names, fsdata.channel_names)
    #     self.assertEqual(np.asarray(wdmdata["X"].entries).shape[-2:], (2, 2))
    #     self.assertEqual(
    #         np.asarray(recovered.get_kernel()).shape[-1],
    #         np.asarray(recovered.frequencies).shape[0],
    #     )

    # def test_wdm_fs_wdm_roundtrip_preserves_grid_and_entries(self):
    #     wdmdata = build_wdm_pair(jnp)["left"]
    #     nf, nt = wdmdata["X"].Nf, wdmdata["X"].Nt
    #
    #     fsdata = wdmdata.to_fsdata()
    #     roundtrip = fsdata.to_wdm_data(Nf=nf, Nt=nt)
    #
    #     self.assertIsInstance(roundtrip, WDMData)
    #     self.assertEqual(roundtrip.channel_names, wdmdata.channel_names)
    #     npt.assert_allclose(
    #         np.asarray(roundtrip["X"].times), np.asarray(wdmdata["X"].times)
    #     )
    #     npt.assert_allclose(
    #         np.asarray(roundtrip["X"].frequencies),
    #         np.asarray(wdmdata["X"].frequencies),
    #     )
    #     npt.assert_allclose(
    #         np.asarray(roundtrip.get_kernel()),
    #         np.asarray(wdmdata.get_kernel()),
    #         rtol=1e-7,
    #         atol=1e-7,
    #     )

    def test_fsdata_frequencies_and_df(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)

        frequencies = np.asarray(fsdata.frequencies)
        npt.assert_allclose(
            frequencies,
            np.asarray(jnp.fft.rfftfreq(len(times), tsdata.dt)),
        )
        self.assertAlmostEqual(float(fsdata.df), float(frequencies[1] - frequencies[0]))

    def test_pick_preserves_requested_order(self):
        case = build_fdata(jnp)

        picked = case.pick(("Y", "X"))

        self.assertEqual(picked.channel_names, ("Y", "X"))
        self.assertEqual(np.asarray(picked["Y"].entries).shape[1], 1)
        npt.assert_allclose(
            np.asarray(picked["Y"].entries),
            np.asarray(case["Y"].entries),
        )

    def test_init_rejects_non_unit_harmonic_dimension(self):
        times = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float64)
        bad_repr = TimeSeries((times,), jnp.ones((1, 2, 2, 1, 4), dtype=jnp.float64))

        with self.assertRaises(ValueError):
            TSData(bad_repr, ("X", "Y"))

    def test_tsdata_to_fsdata_keep_times_returns_timedfsdata(self):
        times, tsdata = _build_tsdata_jax()

        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=True)

        self.assertIsInstance(fsdata, TimedFSData)
        npt.assert_allclose(np.asarray(fsdata.times), np.asarray(times))
        npt.assert_allclose(
            np.asarray(fsdata.frequencies),
            np.asarray(jnp.fft.rfftfreq(len(times), fsdata.dt)),
        )

    def test_phase3_positional_optional_args_emit_warnings(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fs_from_ts_pos = tsdata.representation.rfft(None)
        self.assertTrue(
            any(
                str(item.message)
                == "Passing `tapering` positionally to `rfft` is deprecated and will be removed in 0.7.0; pass it as a keyword argument instead."
                for item in caught
            )
        )
        self.assertTrue(
            any(
                str(item.message)
                == "The method `UniformTimeSeries.rfft` is deprecated and will be removed in 0.8.0; use `shop.time2freq` instead."
                for item in caught
            )
        )
        npt.assert_allclose(
            np.asarray(fs_from_ts_pos.entries),
            np.asarray(shop.time2freq(tsdata.representation).entries),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ts_from_fs_pos = fsdata.representation.irfft(np.asarray(times), None)
        self.assertTrue(
            any(
                str(item.message)
                == "Passing `tapering` positionally to `irfft` is deprecated and will be removed in 0.7.0; pass it as a keyword argument instead."
                for item in caught
            )
        )
        self.assertTrue(
            any(
                str(item.message)
                == "The method `UniformFrequencySeries.irfft` is deprecated and will be removed in 0.8.0; use `shop.freq2time` instead."
                for item in caught
            )
        )
        npt.assert_allclose(
            np.asarray(ts_from_fs_pos.entries),
            np.asarray(shop.freq2time(fsdata.representation, times=np.asarray(times)).entries),
        )

    def test_from_waveform_preserves_entries_and_channels(self):
        case = build_harmonic_projected_frequency_waveform(jnp)

        data = FSData.from_waveform(case["resp_22"])

        self.assertEqual(data.channel_names, case["resp_22"].channel_names)
        npt.assert_allclose(
            np.asarray(data.get_kernel()),
            np.asarray(case["resp_22"].get_kernel()),
        )

    def test_get_embedded_expands_frequency_grid(self):
        case = build_fdata(jnp)
        embedding = jnp.asarray([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
    
        embedded = case.get_embedded((embedding,))
        got = np.asarray(embedded.get_kernel())
        source = np.asarray(case.get_kernel())
    
        self.assertEqual(np.asarray(embedded.frequencies).shape[0], 6)
        npt.assert_allclose(got[..., 1:4], source)
        npt.assert_allclose(got[..., 0], 0.0)
        npt.assert_allclose(got[..., -1], 0.0)

    def test_fsdata_set_times_and_drop_times(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)

        timed = fsdata.set_times(np.asarray(times))

        self.assertIsInstance(timed, TimedFSData)
        self.assertIsInstance(timed.drop_times(), FSData)
        npt.assert_allclose(np.asarray(timed.times), np.asarray(times))

    def test_fsdata_set_times_drop_times_and_to_tsdata(self):
        case = build_fdata(jnp)
        times = np.linspace(0.0, 7.0, 8)

        timed = case.set_times(times)
        recovered = self._assert_to_tsdata_deprecation(timed)

        self.assertIsInstance(timed, TimedFSData)
        self.assertIsInstance(timed.drop_times(), FSData)
        self.assertEqual(recovered.channel_names, case.channel_names)
        npt.assert_allclose(np.asarray(recovered.times), times)

    def test_save_and_load_dispatches_by_type(self):
        case = build_fdata(jnp)
        times = np.linspace(0.0, 7.0, 8)
        timed = case.set_times(times)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            timed.save(handle.name)
            loaded = load_data(handle.name)

        self.assertIsInstance(loaded, TimedFSData)
        npt.assert_allclose(np.asarray(loaded.times), times)
        npt.assert_allclose(
            np.asarray(loaded.get_kernel()),
            np.asarray(timed.get_kernel()),
        )

    def test_tsdata_t_start_and_t_end(self):
        times, tsdata = _build_tsdata_jax()
        self.assertAlmostEqual(tsdata.t_start, float(times[0]))
        self.assertAlmostEqual(tsdata.t_end, float(times[-1]))

    def test_fsdata_f_min_and_f_max(self):
        _, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)
        freqs = np.asarray(fsdata.frequencies)
        self.assertAlmostEqual(float(fsdata.f_min), float(freqs[0]))
        self.assertAlmostEqual(float(fsdata.f_max), float(freqs[-1]))

    def test_fsdata_to_tsdata_explicit_times(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)
        recovered = self._assert_to_tsdata_deprecation(fsdata, np.asarray(times))
        self.assertIsInstance(recovered, TSData)
        self.assertEqual(recovered.channel_names, fsdata.channel_names)
        self.assertEqual(np.asarray(recovered.times).shape[0], len(times))

    # def test_tsdata_to_stftdata(self):
    #     _, tsdata = _build_tsdata_jax()
    #     win = np.hanning(4).astype(float)
    #     stftdata = tsdata.to_stftdata(win=win, hop=2)
    #     self.assertIsInstance(stftdata, STFTData)
    #     self.assertEqual(stftdata.channel_names, tsdata.channel_names)

    # def test_stftdata_get_subset(self):
    #     _, tsdata = _build_tsdata_jax()
    #     win = np.hanning(4).astype(float)
    #     stftdata = tsdata.to_stftdata(win=win, hop=2)

    #     times_arr = np.array(list(stftdata.values())[0].times)
    #     t_start, t_end = float(times_arr[0]), float(times_arr[-1])
    #     sub = stftdata.get_subset(
    #         time_interval=(t_start, t_start + (t_end - t_start) / 2)
    #     )
    #     self.assertIsInstance(sub, STFTData)
    #     self.assertEqual(sub.channel_names, stftdata.channel_names)

    def test_wdmdata_get_subset(self):
        wdmdata = build_wdm_pair(jnp)["left"]
        times_arr = np.array(wdmdata["X"].times)
        t_mid = float(times_arr[len(times_arr) // 2])
        sub = wdmdata.get_subset(time_interval=(float(times_arr[0]), t_mid))
        self.assertIsInstance(sub, WDMData)
        self.assertEqual(sub.channel_names, wdmdata.channel_names)

    def test_load_data_dispatches_tsdata_and_fsdata(self):
        _, tsdata = _build_tsdata_jax()
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            tsdata.save(handle.name)
            loaded = load_data(handle.name)
        self.assertIsInstance(loaded, TSData)
        npt.assert_allclose(
            np.asarray(loaded.get_kernel()), np.asarray(tsdata.get_kernel())
        )

    def test_channel_mapping_set_name(self):
        case = build_fdata(jnp)
        result = case.set_name("my_data")
        self.assertIs(result, case)
        self.assertEqual(case.name, "my_data")

    def test_channel_mapping_create_like(self):
        case = build_fdata(jnp)
        new_entries = jnp.zeros_like(case.entries)
        new_data = case.create_like(new_entries, case.channel_names)
        self.assertIsInstance(new_data, FSData)
        npt.assert_allclose(np.asarray(new_data.entries), 0.0)

    def test_channel_mapping_repr_includes_class_name(self):
        case = build_fdata(jnp)
        r = repr(case)
        self.assertIn("FSData", r)

    def test_channel_mapping_repr_with_name_includes_name(self):
        case = build_fdata(jnp)
        case.set_name("named")
        r = repr(case)
        self.assertIn("name='named'", r)

    def test_from_dict_empty_raises(self):
        with self.assertRaises(ValueError):
            FSData.from_dict({})

    def test_timedfsdata_set_times_updates_in_place(self):
        case = build_fdata(jnp)
        times_orig = np.linspace(0.0, 7.0, 8)
        times_new = np.linspace(0.0, 14.0, 8)
        timed = case.set_times(times_orig)
        result = timed.set_times(times_new)
        self.assertIs(result, timed)
        npt.assert_allclose(np.asarray(timed.times), times_new)

    def test_timedfsdata_get_subset_creates_new(self):
        fdata = build_fdata(jnp)
        times = np.linspace(0.0, 7.0, 8)
        timed = fdata.set_times(times)
        sub = timed.get_subset(
            interval=(float(fdata.frequencies.start), float(fdata.frequencies.stop))
        )
        self.assertIsInstance(sub, TimedFSData)
        npt.assert_allclose(np.asarray(sub.times), times)

    def test_data_arithmetic_inplace_and_reflected(self):
        case = build_fd_pair(jnp)
        left = case["left"]
        right = case["right"]

        left_copy = FSData.from_dict({chn: left[chn] for chn in left.channel_names})
        left_copy += right
        npt.assert_allclose(
            np.asarray(left_copy.entries),
            np.asarray(left.entries),
        )

        scaled = 2.0 * left
        npt.assert_allclose(
            np.asarray(scaled.entries),
            2.0 * np.asarray(left.entries),
        )

    def test_channel_mapping_properties_namespace_and_pick_string(self):
        case = build_fdata(jnp)
        left = case

        self.assertIs(left.xp, left.__xp__())
        self.assertEqual(left.grid, left.representation.grid)
        self.assertEqual(left.domain, left.representation.domain)
        self.assertEqual(left.kind, left.representation.kind)
        npt.assert_allclose(np.asarray(left.get_kernel()), np.asarray(left.entries))

        picked = left.pick("X")
        self.assertEqual(picked.channel_names, ("X",))
        self.assertEqual(np.asarray(picked.entries).shape[1], 1)

    def test_data_unary_op_and_mismatched_binary_op(self):
        case = build_fd_pair(jnp)
        left = case["left"]
        right = FSData(left.representation, ("A", "B"))

        absolute = abs(left)
        npt.assert_allclose(
            np.asarray(absolute.entries), np.abs(np.asarray(left.entries))
        )

        with self.assertRaises(ValueError):
            _ = left + right

    def test_timedfsdata_requires_times(self):
        case = build_fdata(jnp)
        with self.assertRaises(ValueError):
            TimedFSData(case.representation, case.channel_names)

    def test_tsdata_get_zero_padded(self):
        _, tsdata = _build_tsdata_jax()
        padded = tsdata.get_zero_padded((tsdata.dt, 2 * tsdata.dt))
    
        self.assertIsInstance(padded, TSData)
        self.assertEqual(
            np.asarray(padded.entries).shape[-1],
            np.asarray(tsdata.entries).shape[-1] + 3,
        )
        npt.assert_allclose(
            np.asarray(padded.entries)[..., 1:-2],
            np.asarray(tsdata.entries),
        )
        npt.assert_allclose(np.asarray(padded.entries)[..., :1], 0.0)
        npt.assert_allclose(np.asarray(padded.entries)[..., -2:], 0.0)

    def test_load_data_unknown_type_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.attrs["type"] = "UnknownData"
            with self.assertRaises(ValueError):
                load_data(handle.name)

    def test_load_data_dispatches_fsdata(self):
        _, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            fsdata.save(handle.name)
            loaded = load_data(handle.name)

        self.assertIsInstance(loaded, FSData)
        npt.assert_allclose(np.asarray(loaded.entries), np.asarray(fsdata.entries))

    def test_load_ldc_data_aet_to_xyz(self):
        n = 16
        t = np.linspace(0.0, 1.0, n)
        dataset = np.zeros(
            n, dtype=[("t", "f8"), ("A", "f8"), ("E", "f8"), ("T", "f8")]
        )
        dataset["t"] = t
        dataset["A"] = np.sin(2 * np.pi * t)
        dataset["E"] = np.cos(2 * np.pi * t)
        dataset["T"] = 0.5 * np.ones_like(t)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            loaded = load_ldc_data(handle.name, name="obs/tdi", channels="XYZ")

        self.assertIsInstance(loaded, TSData)
        self.assertEqual(loaded.channel_names, ("X", "Y", "Z"))
        self.assertEqual(np.asarray(loaded.entries).shape[-1], n)

    def test_load_ldc_data_xyz_to_ae(self):
        n = 16
        t = np.linspace(0.0, 1.0, n)
        dataset = np.zeros(
            n, dtype=[("t", "f8"), ("X", "f8"), ("Y", "f8"), ("Z", "f8")]
        )
        dataset["t"] = t
        dataset["X"] = np.sin(2 * np.pi * t)
        dataset["Y"] = np.cos(2 * np.pi * t)
        dataset["Z"] = t

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            loaded = load_ldc_data(handle.name, name="obs/tdi", channels="AE")

        self.assertIsInstance(loaded, TSData)
        self.assertEqual(loaded.channel_names, ("A", "E"))
        self.assertEqual(np.asarray(loaded.entries).shape[-1], n)

    def test_load_ldc_data_direct_pick_branch(self):
        n = 16
        t = np.linspace(0.0, 1.0, n)
        dataset = np.zeros(n, dtype=[("t", "f8"), ("A", "f8"), ("E", "f8")])
        dataset["t"] = t
        dataset["A"] = np.sin(2 * np.pi * t)
        dataset["E"] = np.cos(2 * np.pi * t)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            loaded = load_ldc_data(handle.name, name="obs/tdi", channels="AE")

        self.assertEqual(loaded.channel_names, ("A", "E"))

    def test_load_ldc_data_invalid_requested_channels_raises(self):
        n = 8
        t = np.linspace(0.0, 1.0, n)
        dataset = np.zeros(n, dtype=[("t", "f8"), ("A", "f8"), ("E", "f8")])
        dataset["t"] = t

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            with self.assertRaises(ValueError):
                load_ldc_data(handle.name, name="obs/tdi", channels="Q")  # type: ignore[arg-type]

    def test_load_ldc_data_invalid_structure_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=np.arange(8.0))
            with self.assertRaises(ValueError):
                load_ldc_data(handle.name, name="obs/tdi", channels="AE")

    def test_tsdata_draw_uses_ts_plotter(self):
        _, tsdata = _build_tsdata_jax()

        with patch("typed_lisa_toolkit.viz.plotters.TSDataPlotter") as plotter_cls:
            plotter = MagicMock()
            plotter.draw.return_value = "drawn"
            plotter_cls.return_value = plotter

            result = tsdata.draw(interval=(0.0, 1.0))

        self.assertEqual(result, "drawn")
        plotter.draw.assert_called_once()

    def test_fsdata_draw_compare_uses_fs_plotter_compare(self):
        case = build_fd_pair(jnp)
        left = case["left"]
        right = case["right"]

        with patch("typed_lisa_toolkit.viz.plotters.FSDataPlotter") as plotter_cls:
            left_plotter = MagicMock()
            right_plotter = MagicMock()
            left_plotter.compare.return_value = "compared"
            plotter_cls.side_effect = [left_plotter, right_plotter]

            result = left.draw(compare_to=right, interval=(1.0, 3.0))

        self.assertEqual(result, "compared")
        left_plotter.compare.assert_called_once_with(right_plotter)

    def test_wdmdata_draw_uses_tf_plotter(self):
        wdmdata = build_wdm_pair(jnp)["left"]

        with patch("typed_lisa_toolkit.viz.plotters.TFDataPlotter") as plotter_cls:
            plotter = MagicMock()
            plotter.draw.return_value = "tf-drawn"
            plotter_cls.return_value = plotter

            result = wdmdata.draw()

        self.assertEqual(result, "tf-drawn")
        plotter.draw.assert_called_once()


class TestDataInternalAbstractBranchesJAX(DataAbstractBranchesMixin, unittest.TestCase):
    """Abstract/NotImplementedError branch tests (shared via mixin)."""
