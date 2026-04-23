"""Tests for data containers with JAX arrays."""
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false

import tempfile
import warnings
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import typed_lisa_toolkit as tlt
from typed_lisa_toolkit import (
    construct_fsdata,
    construct_stftdata,
    construct_timed_fsdata,
    construct_tsdata,
    construct_wdmdata,
    fsdata,
    linspace_from_array,
    load_data,
    load_ldc_data,
    load_mojito,
    shop,
    stft,
    stftdata,
    time_series,
    tsdata,
    wdm,
    wdmdata,
)
from typed_lisa_toolkit.types import (
    FSData,
    STFTData,
    TimedFSData,
    TSData,
    WDMData,
)

if TYPE_CHECKING:
    from conftest import (
        build_fd_pair,
        build_fdata,
        build_harmonic_projected_frequency_waveform,
        build_wdm_pair,
    )


jax.config.update("jax_enable_x64", val=True)


def _build_tsdata_jax():
    times = tlt.linspace(0.0, 3.0, 8)
    x = jnp.asarray([0.0, 1.0, 0.5, -0.5, -1.0, -0.25, 0.75, 0.0], dtype=jnp.float64)
    y = jnp.asarray([1.0, 0.0, -0.5, 0.25, 0.5, -0.75, 0.0, 1.0], dtype=jnp.float64)
    data = tsdata(
        {
            "X": time_series(times, x[None, None, None, None, :]),
            "Y": time_series(times, y[None, None, None, None, :]),
        },
    )
    return times, data


class TestDataContainersJAX:
    def _assert_construct_tsdata_deprecation(self, **kwargs):
        with pytest.warns(DeprecationWarning, match=r"construct_tsdata"):
            return construct_tsdata(**kwargs)

    def _assert_construct_fsdata_deprecation(self, **kwargs):
        with pytest.warns(DeprecationWarning, match=r"construct_fsdata"):
            return construct_fsdata(**kwargs)

    def _assert_construct_stftdata_deprecation(self, **kwargs):
        with pytest.warns(DeprecationWarning, match=r"construct_stftdata"):
            return construct_stftdata(**kwargs)  # pyright: ignore[reportUnknownVariableType]

    def _assert_construct_wdmdata_deprecation(self, **kwargs):
        with pytest.warns(DeprecationWarning, match=r"construct_wdmdata"):
            return construct_wdmdata(**kwargs)  # pyright: ignore[reportUnknownVariableType]

    def _assert_to_fsdata_deprecation(self, tsdata: TSData, *, keep_times: bool):
        with pytest.warns(
            DeprecationWarning,
            match=r"The 'to_fsdata' method is deprecated",
        ):
            return tsdata.to_fsdata(keep_times=keep_times)

    def _assert_to_tsdata_deprecation(
        self,
        fs_like: FSData | TimedFSData,
        times: np.ndarray | None = None,
    ) -> TSData:
        with pytest.warns(  # noqa: PT031
            DeprecationWarning,
            match=r"The 'to_tsdata' method is deprecated",
        ):
            if times is None:
                return fs_like.to_tsdata()  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            return fs_like.to_tsdata(times)

    def test_tsdata_times_dt_and_get_frequencies(self):
        times, tsdata = _build_tsdata_jax()

        npt.assert_allclose(np.asarray(tsdata.times), np.asarray(times))
        npt.assert_allclose(
            np.asarray(tsdata.get_frequencies()),
            np.asarray(jnp.fft.rfftfreq(len(times), tsdata.dt)),
        )

    def test_fsdata_frequencies_and_df(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)

        frequencies = np.asarray(fsdata.frequencies)
        npt.assert_allclose(
            frequencies,
            np.asarray(jnp.fft.rfftfreq(len(times), tsdata.dt)),
        )
        assert float(fsdata.df) == pytest.approx(float(frequencies[1] - frequencies[0]))

    def test_pick_preserves_requested_order(self):
        case = build_fdata(jnp)

        picked = case.pick(("Y", "X"))

        assert picked.channel_names == ("Y", "X")
        assert np.asarray(picked["Y"].entries).shape[1] == 1
        npt.assert_allclose(
            np.asarray(picked["Y"].entries),
            np.asarray(case["Y"].entries),
        )

    def test_init_rejects_non_unit_harmonic_dimension(self):
        times = tlt.linspace(0.0, 1.0, 4)
        bad_entries = jnp.ones((1, 2, 2, 1, 4), dtype=jnp.float64)
        built = self._assert_construct_tsdata_deprecation(
            times=times,
            entries=bad_entries,
            channels=("X", "Y"),
        )

        assert built.channel_names == ("X", "Y")
        npt.assert_allclose(np.asarray(built.get_kernel()), np.asarray(bad_entries))

    def test_tsdata_to_fsdata_keep_times_returns_timedfsdata(self):
        times, tsdata = _build_tsdata_jax()

        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=True)

        assert isinstance(fsdata, TimedFSData)
        npt.assert_allclose(np.asarray(fsdata.times), np.asarray(times))
        npt.assert_allclose(
            np.asarray(fsdata.frequencies),
            np.asarray(jnp.fft.rfftfreq(len(times), fsdata.dt)),
        )

    def test_phase3_positional_optional_args_emit_warnings(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)
        ts_rep = tsdata["X"]
        fs_rep = fsdata["X"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fs_from_ts_pos = ts_rep.rfft(None)
        assert any(
            "Passing `tapering` positionally to `rfft` is deprecated"
            in str(item.message)
            for item in caught
        )
        assert any(
            "The method `UniformTimeSeries.rfft` is deprecated" in str(item.message)
            for item in caught
        )
        npt.assert_allclose(
            np.asarray(fs_from_ts_pos.entries),
            np.asarray(shop.time2freq(ts_rep).entries),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ts_from_fs_pos = fs_rep.irfft(np.asarray(times), None)
        assert any(
            "Passing `tapering` positionally to `irfft` is deprecated"
            in str(item.message)
            for item in caught
        )
        assert any(
            "The method `UniformFrequencySeries.irfft` is deprecated"
            in str(item.message)
            for item in caught
        )
        npt.assert_allclose(
            np.asarray(ts_from_fs_pos.entries),
            np.asarray(shop.freq2time(fs_rep, times=np.asarray(times)).entries),
        )

    def test_from_waveform_preserves_entries_and_channels(self):
        case = build_harmonic_projected_frequency_waveform(jnp)

        data = fsdata(case["resp_22_map"])

        assert data.channel_names == tuple(case["resp_22_map"].keys())
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

        assert np.asarray(embedded.frequencies).shape[0] == 6
        npt.assert_allclose(got[..., 1:4], source)
        npt.assert_allclose(got[..., 0], 0.0)
        npt.assert_allclose(got[..., -1], 0.0)

    def test_fsdata_set_times_and_drop_times(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)

        timed = fsdata.set_times(np.asarray(times))

        assert isinstance(timed, TimedFSData)
        assert isinstance(timed.drop_times(), FSData)
        npt.assert_allclose(np.asarray(timed.times), np.asarray(times))

    def test_fsdata_set_times_drop_times_and_to_tsdata(self):
        case = build_fdata(jnp)
        times = tlt.linspace(0.0, 7.0, 8)

        timed = case.set_times(times)
        recovered = self._assert_to_tsdata_deprecation(timed)

        assert isinstance(timed, TimedFSData)
        assert isinstance(timed.drop_times(), FSData)
        assert recovered.channel_names == case.channel_names
        npt.assert_allclose(np.asarray(recovered.times), times)

    def test_save_and_load_dispatches_by_type(self):
        case = build_fdata(jnp)
        times = tlt.linspace(0.0, 7.0, 8)
        timed = case.set_times(times)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            timed.save(handle.name)
            with pytest.raises(TypeError):
                _ = load_data(handle.name, domain="frequency", kind="timed")

    def test_fsdata_f_min_and_f_max(self):
        _, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)
        freqs = np.asarray(fsdata.frequencies)
        assert float(fsdata.f_min) == pytest.approx(float(freqs[0]))
        assert float(fsdata.f_max) == pytest.approx(float(freqs[-1]))

    def test_fsdata_to_tsdata_explicit_times(self):
        times, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)
        recovered = self._assert_to_tsdata_deprecation(fsdata, np.asarray(times))
        assert isinstance(recovered, TSData)
        assert recovered.channel_names == fsdata.channel_names
        assert np.asarray(recovered.times).shape[0] == len(times)

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
        assert isinstance(sub, WDMData)
        assert sub.channel_names == wdmdata.channel_names

    def test_load_data_dispatches_tsdata_and_fsdata(self):
        _, tsdata = _build_tsdata_jax()
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            tsdata.save(handle.name)
            loaded = load_data(handle.name, domain="time", kind=None)
        assert isinstance(loaded, TSData)
        npt.assert_allclose(
            np.asarray(loaded.get_kernel()),
            np.asarray(tsdata.get_kernel()),
        )

    def test_channel_mapping_set_name(self):
        case = build_fdata(jnp)
        result = case.set_name("my_data")
        assert result is case
        assert case.name == "my_data"

    def test_channel_mapping_create_like(self):
        case = build_fdata(jnp)
        new_entries = jnp.zeros_like(case.get_kernel())
        new_data = case.create_like(new_entries)
        assert isinstance(new_data, FSData)
        npt.assert_allclose(np.asarray(new_data.get_kernel()), 0.0)

    def test_channel_mapping_repr_includes_class_name(self):
        case = build_fdata(jnp)
        r = repr(case)
        assert "FSData" in r

    def test_channel_mapping_repr_with_name_includes_name(self):
        case = build_fdata(jnp)
        case.set_name("named")
        r = repr(case)
        assert "name='named'" in r

    def test_from_dict_empty_raises(self):
        with pytest.raises(ValueError, match=r".+"):
            fsdata({})

    def test_timedfsdata_set_times_updates_in_place(self):
        case = build_fdata(jnp)
        times_orig = tlt.linspace(0.0, 7.0, 8)
        times_new = tlt.linspace(0.0, 14.0, 8)
        timed = case.set_times(times_orig)
        result = timed.set_times(times_new)
        assert result is timed
        npt.assert_allclose(np.asarray(timed.times), times_new)

    def test_timedfsdata_get_subset_creates_new(self):
        fdata = build_fdata(jnp)
        times = tlt.linspace(0.0, 7.0, 8)
        timed = fdata.set_times(times)
        sub = timed.get_subset(
            interval=(float(fdata.frequencies.start), float(fdata.frequencies.stop)),
        )
        assert isinstance(sub, TimedFSData)

    def test_data_arithmetic_inplace_and_reflected(self):
        case = build_fd_pair(jnp)
        left = case["left"]
        right = case["right"]

        left_copy = fsdata({chn: left[chn] for chn in left.channel_names})
        left_copy += right
        npt.assert_allclose(
            np.asarray(left_copy.get_kernel()),
            np.asarray(left.get_kernel()) + np.asarray(right.get_kernel()),
        )

        scaled = 2.0 * left
        npt.assert_allclose(
            np.asarray(scaled.get_kernel()),
            2.0 * np.asarray(left.get_kernel()),
        )

    def test_channel_mapping_properties_namespace_and_pick_string(self):
        case = build_fdata(jnp)
        left = case
        first = left[left.channel_names[0]]

        assert left.xp is left.__xp__()
        assert left.grid == first.grid
        assert left.domain == first.domain
        assert left.kind == first.kind
        npt.assert_allclose(
            np.asarray(left.get_kernel()),
            np.asarray(left.get_kernel()),
        )

        picked = left.pick("X")
        assert picked.channel_names == ("X",)
        assert np.asarray(picked.get_kernel()).shape[1] == 1

    def test_data_unary_op_and_mismatched_binary_op(self):
        case = build_fd_pair(jnp)
        left = case["left"]
        right = self._assert_construct_fsdata_deprecation(
            frequencies=np.asarray(left.frequencies) + 10.0,
            entries=np.asarray(left.get_kernel()),
            channels=("A", "B"),
        )

        absolute = abs(left)
        npt.assert_allclose(
            np.asarray(absolute.get_kernel()),
            np.abs(np.asarray(left.get_kernel())),
        )

        with pytest.raises((ValueError, TypeError)):
            _ = left + right

    def test_timedfsdata_requires_times(self):
        case = build_fdata(jnp)
        timed = TimedFSData(case.grid, case.entries, channels=case.channel_names)
        with pytest.raises(AttributeError):
            _ = timed.times

    def test_tsdata_get_zero_padded(self):
        _, tsdata = _build_tsdata_jax()
        padded = tsdata.get_zero_padded((tsdata.dt, 2 * tsdata.dt))

        assert isinstance(padded, TSData)
        assert (
            np.asarray(padded.get_kernel()).shape[-1]
            == np.asarray(tsdata.get_kernel()).shape[-1] + 3
        )
        npt.assert_allclose(
            np.asarray(padded.get_kernel())[..., 1:-2],
            np.asarray(tsdata.get_kernel()),
        )
        npt.assert_allclose(np.asarray(padded.get_kernel())[..., :1], 0.0)
        npt.assert_allclose(np.asarray(padded.get_kernel())[..., -2:], 0.0)

    def test_load_data_unknown_type_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.attrs["type"] = "UnknownData"
            with (
                pytest.raises(ValueError, match=r".+"),
                pytest.warns(
                    DeprecationWarning,
                    match="load_data",
                ),
            ):
                load_data(handle.name, legacy=True)  # pyright: ignore[reportCallIssue]

    def test_load_data_dispatches_fsdata(self):
        _, tsdata = _build_tsdata_jax()
        fsdata = self._assert_to_fsdata_deprecation(tsdata, keep_times=False)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            fsdata.save(handle.name)
            loaded = load_data(handle.name, domain="frequency", kind=None)

        assert isinstance(loaded, FSData)
        npt.assert_allclose(
            np.asarray(loaded.get_kernel()),
            np.asarray(fsdata.get_kernel()),
        )

    def test_load_ldc_data_aet_to_xyz(self):
        n = 16
        t = np.linspace(0.0, 1.0, n)
        dataset = np.zeros(
            n,
            dtype=[("t", "f8"), ("A", "f8"), ("E", "f8"), ("T", "f8")],
        )
        dataset["t"] = t
        dataset["A"] = np.sin(2 * np.pi * t)
        dataset["E"] = np.cos(2 * np.pi * t)
        dataset["T"] = 0.5 * np.ones_like(t)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            loaded = load_ldc_data(handle.name, name="obs/tdi", channels="XYZ")

        assert isinstance(loaded, TSData)
        assert loaded.channel_names == ("X", "Y", "Z")
        assert np.asarray(loaded.get_kernel()).shape[-1] == n

    def test_load_ldc_data_xyz_to_ae(self):
        n = 16
        t = np.linspace(0.0, 1.0, n)
        dataset = np.zeros(
            n,
            dtype=[("t", "f8"), ("X", "f8"), ("Y", "f8"), ("Z", "f8")],
        )
        dataset["t"] = t
        dataset["X"] = np.sin(2 * np.pi * t)
        dataset["Y"] = np.cos(2 * np.pi * t)
        dataset["Z"] = t

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            loaded = load_ldc_data(handle.name, name="obs/tdi", channels="AE")

        assert isinstance(loaded, TSData)
        assert loaded.channel_names == ("A", "E")
        assert np.asarray(loaded.get_kernel()).shape[-1] == n

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

        assert loaded.channel_names == ("A", "E")

    def test_load_ldc_data_invalid_requested_channels_raises(self):
        n = 8
        t = np.linspace(0.0, 1.0, n)
        dataset = np.zeros(n, dtype=[("t", "f8"), ("A", "f8"), ("E", "f8")])
        dataset["t"] = t

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            with pytest.raises(ValueError, match=r".+"):
                load_ldc_data(handle.name, name="obs/tdi", channels="Q")

    def test_load_ldc_data_invalid_structure_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=np.arange(8.0))
            with pytest.raises(ValueError, match=r".+"):
                load_ldc_data(handle.name, name="obs/tdi", channels="AE")

    def test_tsdata_draw_uses_ts_plotter(self):
        _, tsdata = _build_tsdata_jax()

        with patch("typed_lisa_toolkit.viz.plotters.TSDataPlotter") as plotter_cls:
            plotter = MagicMock()
            plotter.draw.return_value = "drawn"
            plotter_cls.return_value = plotter

            result = tsdata.draw(interval=(0.0, 1.0))

        assert result == "drawn"
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

        assert result == "compared"
        left_plotter.compare.assert_called_once_with(right_plotter)

    def test_wdmdata_draw_uses_tf_plotter(self):
        wdmdata = build_wdm_pair(jnp)["left"]

        with patch("typed_lisa_toolkit.viz.plotters.TFDataPlotter") as plotter_cls:
            plotter = MagicMock()
            plotter.draw.return_value = "tf-drawn"
            plotter_cls.return_value = plotter

            result = wdmdata.draw()

        assert result == "tf-drawn"
        plotter.draw.assert_called_once()

    def test_factory_constructors_build_expected_types(self):
        times = tlt.linspace(0.0, 3.0, 8)
        freqs = linspace_from_array(jnp.fft.rfftfreq(len(times), d=float(times.step)))

        ts_entries = jnp.ones((1, 2, 1, 1, len(times)), dtype=jnp.float64)
        fs_entries = jnp.ones((1, 2, 1, 1, len(freqs)), dtype=jnp.complex128)
        stft_entries = jnp.ones((1, 2, 1, 1, len(freqs), len(times)), dtype=jnp.float64)
        wdm_entries = jnp.ones((1, 2, 1, 1, len(freqs), len(times)), dtype=jnp.float64)

        ts = self._assert_construct_tsdata_deprecation(
            times=times,
            entries=ts_entries,
            channels=("X", "Y"),
            name="ts",
        )
        fs = self._assert_construct_fsdata_deprecation(
            frequencies=freqs,
            entries=fs_entries,
            channels=("X", "Y"),
            name="fs",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tfs = construct_timed_fsdata(
                frequencies=freqs,
                entries=fs_entries,
                channels=("X", "Y"),
                times=times,
                name="tfs",
            )
        stft_data = self._assert_construct_stftdata_deprecation(
            frequencies=freqs,
            times=times,
            entries=stft_entries,
            channels=("X", "Y"),
            name="stft",
        )
        wdm_data = self._assert_construct_wdmdata_deprecation(
            frequencies=freqs,
            times=times,
            entries=wdm_entries,
            channels=("X", "Y"),
            name="wdm",
        )

        assert isinstance(ts, TSData)
        assert isinstance(fs, FSData)
        assert isinstance(tfs, TimedFSData)
        assert isinstance(stft_data, STFTData)
        assert isinstance(wdm_data, WDMData)
        assert ts.channel_names == ("X", "Y")
        assert fs.channel_names == ("X", "Y")
        assert tfs.channel_names == ("X", "Y")
        assert stft_data.channel_names == ("X", "Y")
        assert wdm_data.channel_names == ("X", "Y")

    def test_stftdata_and_wdmdata_mapping_factories(self):
        times = tlt.linspace(0.0, 3.0, 8)
        freqs = linspace_from_array(jnp.fft.rfftfreq(len(times), d=times.step))

        stft_mapping = {
            "X": stft(
                freqs,
                times,
                jnp.ones((1, 1, 1, 1, len(freqs), len(times)), dtype=jnp.float64),
            ),
            "Y": stft(
                freqs,
                times,
                jnp.ones((1, 1, 1, 1, len(freqs), len(times)), dtype=jnp.float64),
            ),
        }
        wdm_mapping = {
            "X": wdm(
                frequencies=freqs,
                times=times,
                entries=jnp.ones(
                    (1, 1, 1, 1, len(freqs), len(times)),
                    dtype=jnp.float64,
                ),
            ),
            "Y": wdm(
                frequencies=freqs,
                times=times,
                entries=jnp.ones(
                    (1, 1, 1, 1, len(freqs), len(times)),
                    dtype=jnp.float64,
                ),
            ),
        }

        stft_data = stftdata(stft_mapping)
        wdm_data = wdmdata(wdm_mapping)

        assert isinstance(stft_data, STFTData)
        assert isinstance(wdm_data, WDMData)
        assert stft_data.channel_names == ("X", "Y")
        assert wdm_data.channel_names == ("X", "Y")

    def test_factory_constructors_reject_non_uniform_axes(self):
        times = jnp.array([0.0, 1.0, 3.0, 6.0], dtype=jnp.float64)
        freqs = jnp.fft.rfftfreq(len(times), d=1.0)

        ts_entries = jnp.ones((1, 2, 1, 1, len(times)), dtype=jnp.float64)
        stft_entries = jnp.ones((1, 2, 1, 1, len(freqs), len(times)), dtype=jnp.float64)

        with (
            pytest.raises(ValueError, match="grid axes must be uniform"),
            pytest.warns(DeprecationWarning, match=r"construct_tsdata"),
        ):
            _ = construct_tsdata(
                times=times,
                entries=ts_entries,
                channels=("X", "Y"),
            )

        with (
            pytest.raises(ValueError, match="grid axes must be uniform"),
            pytest.warns(DeprecationWarning, match=r"construct_stftdata"),
        ):
            _ = construct_stftdata(
                frequencies=freqs,
                times=times,
                entries=stft_entries,
                channels=("X", "Y"),
            )

    def test_fsdata_legacy_load(self):
        freqs = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = np.array([1.0 + 0.5j, -1.0j, 2.0 + 0.0j], dtype=np.complex128)
        y = np.array([0.5 - 0.25j, -1.0 + 0.25j, 2.0 + 0.5j], dtype=np.complex128)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                for name, values in {"X": x, "Y": y}.items():
                    grp = f.create_group(name)
                    grp.create_dataset("grid", data=freqs)
                    grp.create_dataset("entries", data=values)

            with pytest.warns(
                DeprecationWarning,
                match=r"The 'load' method is deprecated",
            ):
                loaded = FSData.load(handle.name, legacy=True)

        assert isinstance(loaded, FSData)
        assert loaded.channel_names == ("X", "Y")
        npt.assert_allclose(np.asarray(loaded["X"].entries).squeeze(), x)
        npt.assert_allclose(np.asarray(loaded["Y"].entries).squeeze(), y)

    def test_load_sangria_non_time_domain_raises(self):
        dataset = np.zeros(8, dtype=[("f", "f8"), ("A", "f8"), ("E", "f8")])

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            with h5py.File(handle.name, "w") as f:
                f.create_dataset("obs/tdi", data=dataset)
            with pytest.raises(ValueError, match=r".+"):
                load_ldc_data(handle.name, name="obs/tdi", channels="AE")

    def test_load_mojito_builds_tsdata(self):
        t = np.linspace(0.0, 1.0, 8)
        processor = MagicMock()
        processor.channels = ["X", "Y"]
        processor.t = t
        processor.data = {
            "X": np.sin(2.0 * np.pi * t),
            "Y": np.cos(2.0 * np.pi * t),
        }

        loaded = load_mojito(processor)

        assert isinstance(loaded, TSData)
        assert loaded.channel_names == ("X", "Y")
        assert np.asarray(loaded.get_kernel()).shape[-1] == len(t)


class TestDataInternalAbstractBranchesJAX:
    def test_data_base_get_plotter_notimplemented(self, data_abstract_branch_helpers):
        data_abstract_branch_helpers.test_data_base_get_plotter_notimplemented()


class TestDataLoadValidationBranchesJAX:
    def test_load_data_warns_when_domain_not_provided(self):
        _, tsdata = _build_tsdata_jax()

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            tsdata.save(handle.name)
            with pytest.warns(FutureWarning):
                loaded = load_data(handle.name, kind=None)

        assert isinstance(loaded, TSData)

    def test_load_data_domain_mismatch_raises(self):
        _, tsdata = _build_tsdata_jax()

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            tsdata.save(handle.name)
            with pytest.raises(ValueError, match=r".+"):
                load_data(handle.name, domain="frequency", kind=None)

    def test_load_data_kind_mismatch_raises(self):
        _, tsdata = _build_tsdata_jax()
        with pytest.warns(
            DeprecationWarning,
            match=r"The 'to_fsdata' method is deprecated",
        ):
            fsd = tsdata.to_fsdata(keep_times=False)

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            fsd.save(handle.name)
            with pytest.raises(ValueError, match=r".+"):
                load_data(handle.name, domain="frequency", kind="timed")

    def test_load_data_sparse_not_supported_for_time_or_frequency(self):
        _, tsdata = _build_tsdata_jax()

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            tsdata.save(handle.name)
            with pytest.raises(ValueError, match=r".+"):
                load_data(handle.name, domain="time", kind=None, sparse=True)

    def test_load_data_dispatches_sparse_stft_and_dense_wdm(self):
        times = tlt.linspace(0.0, 3.0, 8)
        freqs = linspace_from_array(jnp.fft.rfftfreq(len(times), d=float(times.step)))
        sparse_indices = np.array([[0, 0], [1, 2], [2, 4]], dtype=int)

        stft_entries = jnp.ones((1, 2, 1, 1, len(sparse_indices)), dtype=jnp.float64)
        with pytest.warns(DeprecationWarning, match=r"construct_stftdata"):
            stft_data = construct_stftdata(
                frequencies=freqs,
                times=times,
                entries=stft_entries,
                channels=("X", "Y"),
                sparse_indices=sparse_indices,
            )

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            stft_data.save(handle.name)
            loaded_stft = load_data(
                handle.name,
                domain="time-frequency",
                kind="stft",
                sparse=True,
            )

        assert isinstance(loaded_stft, STFTData)
        npt.assert_allclose(
            np.asarray(loaded_stft.get_kernel()),
            np.asarray(stft_data.get_kernel()),
        )

        wdm_entries = jnp.ones((1, 2, 1, 1, len(freqs), len(times)), dtype=jnp.float64)
        with pytest.warns(DeprecationWarning, match=r"construct_wdmdata"):
            wdm_data = construct_wdmdata(
                frequencies=freqs,
                times=times,
                entries=wdm_entries,
                channels=("X", "Y"),
            )

        with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
            wdm_data.save(handle.name)
            loaded_wdm = load_data(
                handle.name,
                domain="time-frequency",
                kind="wdm",
                sparse=False,
            )

        assert isinstance(loaded_wdm, WDMData)
        npt.assert_allclose(
            np.asarray(loaded_wdm.get_kernel()),
            np.asarray(wdm_data.get_kernel()),
        )

    def test_timedfsdata_to_tsdata_ignores_explicit_times_argument(self):
        times, tsdata = _build_tsdata_jax()
        with pytest.warns(
            DeprecationWarning,
            match=r"The 'to_fsdata' method is deprecated",
        ):
            timed = tsdata.to_fsdata(keep_times=True)
        alt_times = np.linspace(100.0, 103.5, len(times))

        with pytest.warns(
            DeprecationWarning,
            match=r"The 'to_tsdata' method is deprecated",
        ):
            recovered = timed.to_tsdata(alt_times)

        assert isinstance(recovered, TSData)
        assert np.asarray(recovered.times).shape[0] == len(times)
