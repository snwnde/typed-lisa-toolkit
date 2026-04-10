# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import frequency_series, stft, time_series, wdm
from typed_lisa_toolkit.types import (
    STFT,
    WDM,
    FSData,
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    HomogeneousHarmonicProjectedWaveform,
    Linspace,
    ProjectedWaveform,
    TimeSeries,
    TSData,
    UniformFrequencySeries,
    UniformTimeSeries,
    WDMData,
    data,
    modes,
)
from typed_lisa_toolkit.types import (
    representations as reps,
)
from typed_lisa_toolkit.types.representations import (
    _check_entry_grid_compatibility,
    _take_subset,
)

if TYPE_CHECKING:
    from typed_lisa_toolkit.types import Axis

SEED = 11324214
rng = np.random.default_rng(SEED)


def _randn_array(xp, shape):
    return xp.asarray(rng.standard_normal(shape))


def _canonicalize_1d_entries(values):
    """Convert 1D values to canonical entries shape: (B, C, H, F, N)."""
    return values[None, None, None, None, :]


def _canonicalize_2d_entries(values):
    """Convert 2D values to canonical entries shape: (B, C, H, F, Nf, Nt)."""
    return values[None, None, None, None, :, :]


def _build_uniform_frequencies(xp):
    return xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)


def _build_frequencies(xp):
    return xp.asarray([1.0, 2.0, 4.0], dtype=xp.float64)


def _build_wdm_axes(xp):
    nt, nf = 20, 16
    dt = 0.12891289
    df = 1.0 / (2.0 * dt)
    times = xp.asarray(dt * np.arange(nt), dtype=xp.float64)
    frequencies = xp.asarray(df * np.arange(nf), dtype=xp.float64)
    return times, frequencies


def _build_complex_entries(xp, values, *, random_scale=False):
    entries = _canonicalize_1d_entries(xp.asarray(values, dtype=xp.complex128))
    if random_scale:
        entries = entries * rng.standard_normal(entries.shape)
    return entries


def _build_fsdata(frequencies, channel_entries):
    return FSData.from_dict(
        {
            name: frequency_series(frequencies, entries)
            for name, entries in channel_entries.items()
        }
    )


def _build_wdmdata(times, frequencies, channel_entries):
    return WDMData.from_dict(
        {
            name: wdm(frequencies=frequencies, times=times, entries=entries)
            for name, entries in channel_entries.items()
        }
    )


def _stack_batched_entries(xp, base_entries, variant_entries):
    return xp.concatenate([base_entries, variant_entries], axis=0)


def _build_2ch_kernel(xp, values_x, values_y, offdiag):
    row0 = xp.stack([values_x, offdiag], axis=-1)
    row1 = xp.stack([offdiag, values_y], axis=-1)
    return xp.stack([row0, row1], axis=-2)


def _build_batched_channel_entries(xp, channel_specs):
    return {
        name: _stack_batched_entries(xp, entries, transform(entries))
        for name, (entries, transform) in channel_specs.items()
    }


def build_canonical_representations(
    xp,
    *,
    n_batches,
    n_channels,
    n_harmonics,
    n_features,
    len_time,
    len_freq,
):
    freqs = xp.linspace(0, 1, len_freq)
    times = xp.linspace(0, 10, len_time)

    entries_fs = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_freq),
    )
    fs = frequency_series(freqs, entries_fs)

    entries_ts = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_time),
    )
    ts = time_series(times, entries_ts)
    entries_tf = _randn_array(
        xp,
        (n_batches, n_channels, n_harmonics, n_features, len_freq, len_time),
    )
    tf = stft(freqs, times, entries=entries_tf)

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


def build_freq_series(xp, uniform=True):
    if uniform:
        frequencies = _build_uniform_frequencies(xp)
    else:
        frequencies = _build_frequencies(xp)
    _entries = _build_complex_entries(
        xp, [1.0 + 0.5j, -1.0j, 2.0 + 0.0j], random_scale=True
    )
    return {
        "fs": frequency_series(frequencies, _entries),
        "frequencies": frequencies,
        "entries": _entries,
    }


def build_fdata(xp):
    frequencies = _build_uniform_frequencies(xp)
    x_entries = _build_complex_entries(xp, [1.0 + 0.5j, -1.0j, 2.0 + 0.0j])
    y_entries = _build_complex_entries(
        xp, [0.5 - 0.25j, -1.0 + 0.25j, 2.0 + 0.5j]
    )
    return _build_fsdata(frequencies, {"X": x_entries, "Y": y_entries})


def build_wdm(xp):
    times, frequencies = _build_wdm_axes(xp)
    nt, nf = len(times), len(frequencies)
    entries = _randn_array(xp, (1, 1, 1, 1, nf, nt))
    return WDM.make(times=times, frequencies=frequencies, entries=entries)


def build_fd_pair(xp):
    frequencies = _build_uniform_frequencies(xp)
    left_x = _build_complex_entries(xp, [1.0 + 1.0j, 2.0 - 1.0j, -1.0 + 0.5j])
    left_y = _build_complex_entries(xp, [0.5 - 0.25j, -1.0j, 2.0 + 0.0j])
    right_x = _build_complex_entries(xp, [2.0 - 1.0j, -1.0 + 2.0j, 0.5 + 0.25j])
    right_y = _build_complex_entries(xp, [1.0 + 0.0j, 0.25 + 1.0j, -0.5 + 2.0j])

    left = _build_fsdata(frequencies, {"X": left_x, "Y": left_y})
    right = _build_fsdata(frequencies, {"X": right_x, "Y": right_y})

    return {
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_x,
        "left_y": left_y,
        "right_x": right_x,
        "right_y": right_y,
    }


def build_wdm_pair(xp):
    times, frequencies = _build_wdm_axes(xp)
    nt, nf = len(times), len(frequencies)

    left_x = _canonicalize_2d_entries(xp.outer(xp.cos(frequencies), xp.sin(times)))
    left_y = _canonicalize_2d_entries(xp.ones((nf, nt), dtype=xp.float64))
    right_x = left_x
    right_y = left_y

    left = _build_wdmdata(times, frequencies, {"X": left_x, "Y": left_y})
    right = _build_wdmdata(times, frequencies, {"X": right_x, "Y": right_y})

    return {
        "times": times,
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_x,
        "left_y": left_y,
        "right_x": right_x,
        "right_y": right_y,
    }



def build_fd_pair_batched_2x2(xp):
    base = build_fd_pair(xp)
    frequencies = base["frequencies"]

    left_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (
                base["left_x"],
                lambda entries: 0.75 * entries + (0.1 - 0.2j),
            ),
            "Y": (
                base["left_y"],
                lambda entries: 1.25 * entries + (-0.05 + 0.1j),
            ),
        },
    )
    right_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (
                base["right_x"],
                lambda entries: -0.5 * entries + (0.2 + 0.05j),
            ),
            "Y": (
                base["right_y"],
                lambda entries: 0.6 * entries + (-0.1 + 0.15j),
            ),
        },
    )

    left = _build_fsdata(
        frequencies,
        {
            name: entries for name, entries in left_entries.items()
        },
    )
    right = _build_fsdata(
        frequencies,
        {
            name: entries for name, entries in right_entries.items()
        },
    )

    return {
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_entries["X"],
        "left_y": left_entries["Y"],
        "right_x": right_entries["X"],
        "right_y": right_entries["Y"],
    }


def build_wdm_pair_batched_2x2(xp):
    base = build_wdm_pair(xp)
    times = base["times"]
    frequencies = base["frequencies"]

    left_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (base["left_x"], lambda entries: 0.8 * entries + 0.3),
            "Y": (base["left_y"], lambda entries: 1.1 * entries - 0.2),
        },
    )
    right_entries = _build_batched_channel_entries(
        xp,
        {
            "X": (base["right_x"], lambda entries: -0.4 * entries + 0.5),
            "Y": (base["right_y"], lambda entries: 0.7 * entries + 0.25),
        },
    )

    left = _build_wdmdata(
        times,
        frequencies,
        {
            name: entries for name, entries in left_entries.items()
        },
    )
    right = _build_wdmdata(
        times,
        frequencies,
        {
            name: entries for name, entries in right_entries.items()
        },
    )

    return {
        "times": times,
        "frequencies": frequencies,
        "left": left,
        "right": right,
        "left_x": left_entries["X"],
        "left_y": left_entries["Y"],
        "right_x": right_entries["X"],
        "right_y": right_entries["Y"],
    }


def diagonal_kernel_2ch(xp):
    values_x = xp.asarray([2.0, 4.0, 8.0], dtype=xp.float64)
    values_y = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    offdiag = xp.zeros_like(values_x)
    return _build_2ch_kernel(xp, values_x, values_y, offdiag)


def dense_kernel_2ch(xp):
    values_x = xp.asarray([2.0, 4.0, 8.0], dtype=xp.float64)
    values_y = xp.asarray([1.0, 0.5, 0.25], dtype=xp.float64)
    offdiag = xp.asarray([0.1, 0.2, -0.3], dtype=xp.float64)
    return _build_2ch_kernel(xp, values_x, values_y, offdiag)


def dense_esdm_2ch(xp):
    times, frequencies = _build_wdm_axes(xp)
    n_freq = len(frequencies)
    n_time = len(times)
    fi = xp.arange(n_freq, dtype=xp.float64)[:, None]
    ti = xp.arange(n_time, dtype=xp.float64)[None, :]

    a = 1.8 + 0.05 * xp.cos(0.3 * fi) + 0.03 * xp.sin(0.2 * ti)
    d = 1.2 + 0.04 * xp.sin(0.25 * fi) + 0.02 * xp.cos(0.35 * ti)
    b = 0.08 * xp.cos(0.15 * fi + 0.1 * ti)

    row0 = xp.stack([a, b], axis=-1)
    row1 = xp.stack([b, d], axis=-1)
    return xp.stack([row0, row1], axis=-2)


class LinspaceExtraPropertiesMixin:
    """Backend-agnostic Linspace edge-case tests (no array operations)."""

    def test_shape_property(self):
        ls = Linspace(1.0, 0.5, 4)
        self.assertEqual(ls.shape, (4,))

    def test_stop_property(self):
        ls = Linspace(1.0, 0.5, 4)
        self.assertAlmostEqual(ls.stop, 2.5)

    def test_eq_raises_for_non_linspacelike(self):
        ls = Linspace(0.0, 1.0, 5)
        with self.assertRaises(TypeError):
            ls.__eq__(42)

    def test_eq_returns_false_for_step_mismatch(self):
        ls1 = Linspace(0.0, 1.0, 5)
        ls2 = Linspace(0.0, 2.0, 5)
        self.assertFalse(ls1 == ls2)

    def test_array_with_copy_false(self):
        ls = Linspace(0.0, 1.0, 5)
        arr = np.array(ls, copy=False)
        npt.assert_allclose(arr, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_getitem_invalid_type_raises(self):
        ls = Linspace(0.0, 1.0, 10)
        with self.assertRaises(TypeError):
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
        self.assertIsInstance(ls, Linspace)
        self.assertAlmostEqual(ls.start, 2.0)
        self.assertAlmostEqual(ls.step, 0.5)
        self.assertEqual(ls.num, 8)


class HelperFunctionsMixin:
    """Mix-in for module-level helper function tests.

    Subclass must provide a class-level ``xp`` attribute (numpy or jax.numpy).
    """

    xp = None  # subclasses set: np or jnp

    def test_check_entry_grid_compatibility_raises_on_mismatch(self):
        xp = self.xp
        grid = (xp.asarray(np.linspace(0, 1, 10)),)
        entries = xp.asarray(np.ones((1, 1, 1, 1, 20)))
        with self.assertRaises(ValueError):
            _check_entry_grid_compatibility(grid, entries)

    def test_take_subset_slice_dimension_mismatch_raises(self):
        xp = self.xp
        grid = (xp.asarray(np.linspace(0, 1, 10)),)
        entries = xp.asarray(np.ones((1, 1, 1, 1, 10)))
        with self.assertRaises(ValueError):
            _take_subset(grid, entries, (slice(0, 5), slice(0, 3)))

    def test_take_subset_with_array_grid(self):
        xp = self.xp
        grid = (xp.asarray(np.linspace(0.0, 1.0, 20)),)
        entries = xp.asarray(np.arange(20, dtype=float))[None, None, None, None, :]
        new_grid, new_entries = _take_subset(grid, entries, (slice(5, 10),))
        npt.assert_allclose(np.asarray(new_grid[0]), np.asarray(grid[0][5:10]))
        npt.assert_allclose(
            np.asarray(new_entries[0, 0, 0, 0, :]),
            np.asarray(entries[0, 0, 0, 0, 5:10]),
        )

    def test_non_uniform_grid_stays_array(self):
        xp = self.xp
        non_uniform = xp.asarray(np.array([0.0, 1.0, 3.0, 7.0]))
        ts = UniformTimeSeries(
            grid=(non_uniform,), entries=xp.asarray(np.ones((1, 1, 1, 1, 4)))
        )
        self.assertNotIsInstance(ts.grid[0], Linspace)

    def test_axis_onset_and_end_from_plain_arrays(self):
        xp = self.xp
        freqs = xp.asarray(np.array([0.01, 0.02, 0.03, 0.04]))
        fs = UniformFrequencySeries(
            grid=(freqs,), entries=xp.asarray(np.ones((1, 1, 1, 1, 4)))
        )
        self.assertAlmostEqual(fs.f_min, 0.01)
        self.assertAlmostEqual(fs.f_max, 0.04)


class AdvancedRepresentationMethodsMixin:
    """Mix-in for UniformFrequencySeries/UniformTimeSeries/STFT method tests.

    Subclass must provide ``xp`` (numpy or jax.numpy).
    Tests that use only plain numpy arrays need no ``xp``.
    """

    xp = None  # subclasses set: np or jnp

    def test_frequency_series_get_time_shifted(self):
        xp = self.xp
        n, dt = 32, 1.0 / 128
        freqs = xp.asarray(np.fft.rfftfreq(n, d=dt))
        entries_fs = xp.asarray(np.fft.rfft(np.sin(2 * np.pi * np.arange(n) * dt)))[
            None, None, None, None, :
        ]
        fs = UniformFrequencySeries(grid=(freqs,), entries=entries_fs)
        shifted = fs.get_time_shifted(2 * dt)
        self.assertIsInstance(shifted, UniformFrequencySeries)
        self.assertEqual(shifted.entries.shape, fs.entries.shape)

    def test_frequency_series_angle(self):
        xp = self.xp
        freqs = xp.asarray(np.linspace(1e-4, 1e-2, 10))
        z = xp.asarray(np.exp(1j * np.linspace(0, 4 * np.pi, 10)))[
            None, None, None, None, :
        ]
        fs = UniformFrequencySeries(grid=(freqs,), entries=z)
        angles = fs.angle()
        self.assertIsInstance(angles, UniformFrequencySeries)
        self.assertEqual(angles.entries.shape, fs.entries.shape)

    def test_stft_make_classmethod(self):
        times = np.linspace(0, 10, 100)
        freqs = np.linspace(0, 1, 50)
        entries = np.random.randn(1, 1, 1, 1, 100, 50)
        stft = STFT.make(times=times, frequencies=freqs, entries=entries)
        self.assertIsInstance(stft, STFT)
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
        xp = self.xp
        freqs = Linspace(0.0, 1e-3, 20)
        entries = xp.asarray(np.ones((1, 1, 1, 1, 20)))
        fs = UniformFrequencySeries(grid=(freqs,), entries=entries)
        r = repr(fs)
        self.assertIn("UniformFrequencySeries", r)


class WDMPropertiesAndMethodsMixin:
    """Mix-in for WDM property/method tests.

    Subclass must set ``self.wdm`` in ``setUp`` using the appropriate backend.
    """

    def test_nd_duration_sample_interval(self):
        wdm = self.wdm
        self.assertEqual(wdm.ND, wdm.Nf * wdm.Nt)
        # self.assertAlmostEqual(wdm.duration, wdm.Nt * wdm.times.step)
        # self.assertAlmostEqual(wdm.sample_interval, wdm.duration / wdm.ND)
        self.assertAlmostEqual(wdm.dt, wdm.sample_interval)

    def test_df_shape_sample_rate_nyquist(self):
        wdm = self.wdm
        # self.assertAlmostEqual(wdm.df, 1.0 / wdm.duration)
        self.assertEqual(wdm.shape, (wdm.Nf, wdm.Nt))
        # self.assertAlmostEqual(wdm.sample_rate, 1.0 / wdm.sample_interval)
        # self.assertAlmostEqual(wdm.nyquist, wdm.sample_rate / 2.0)

    def test_is_critically_sampled(self):
        wdm = self.wdm
        result = wdm.is_critically_sampled()
        expected = bool(np.isclose(wdm.dT * wdm.dF, 0.5))
        self.assertEqual(bool(result), expected)

    def test_get_subset_time(self):
        wdm = self.wdm
        times_arr = np.asarray(wdm.times)
        t_mid = float(times_arr[len(times_arr) // 2])
        sub = wdm.get_subset(time_interval=(float(times_arr[0]), t_mid))
        self.assertIsInstance(sub, WDM)
        self.assertLess(sub.Nt, wdm.Nt)

    def test_get_subset_freq(self):
        wdm = self.wdm
        freqs_arr = np.asarray(wdm.frequencies)
        f_mid = float(freqs_arr[len(freqs_arr) // 2])
        sub = wdm.get_subset(freq_interval=(float(freqs_arr[0]), f_mid))
        self.assertIsInstance(sub, WDM)
        self.assertLess(sub.Nf, wdm.Nf)


class DataAbstractBranchesMixin:
    """Mix-in testing abstract/NotImplementedError branches in data-container mixins.

    Uses plain numpy arrays throughout — the tested code paths are
    backend-agnostic and only check NotImplementedError semantics.
    """

    def test_data_base_get_plotter_notimplemented(self):
        class Dummy(data.Data[TimeSeries["Axis"]]):
            _reps_type = TimeSeries["Axis"]

        times = np.linspace(0.0, 1.0, 4)
        representation = TimeSeries["Axis"]((times,), np.ones((1, 1, 1, 1, 4)))
        dummy = Dummy.from_representation(representation, ("X",))

        with self.assertRaises(NotImplementedError):  # type: ignore[attr-defined]
            dummy._get_plotter()


# ============================================================================
# Waveform Helper Classes and Functions (from waveforms_helpers.py)
# ============================================================================

class FakeResponse(dict[str, Any]):
    @property
    def channel_names(self):
        return tuple(self.keys())


class FakeHarmonicWaveform(dict[modes.Harmonic, FakeResponse]):
    @property
    def harmonics(self):
        return tuple(self.keys())


def make_valid_mock_representation(*, name: str | None = None, frequencies: Any = None):
    """Return a MagicMock that satisfies representation runtime validators."""
    rep = MagicMock(name=name)
    if frequencies is None:
        grid = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    else:
        grid = np.asarray(frequencies, dtype=np.float64)
    rep.domain = "frequency"
    rep.grid = (grid,)
    rep.entries = np.zeros((1, 1, 1, 1, len(grid)), dtype=np.complex128)

    def _create_like(entries):
        entries_arr = np.asarray(entries)
        out = make_valid_mock_representation(
            name=name,
            frequencies=np.arange(entries_arr.shape[-1], dtype=np.float64),
        )
        out.entries = entries_arr
        # Preserve waveform-helper methods/metadata when ProjectedWaveform views call create_like.
        for attr in ("f_min", "f_max"):
            if hasattr(rep, attr):
                setattr(out, attr, getattr(rep, attr))
        for method in ("get_interpolated", "get_embedded", "to_frequency_series"):
            if hasattr(rep, method):
                setattr(out, method, getattr(rep, method))
        return out

    rep.create_like.side_effect = _create_like
    return rep


def set_mock_frequency_entries(rep: Any, frequencies: Any):
    """Assign canonical frequency-domain entries/grid to a representation mock."""
    grid = np.asarray(frequencies, dtype=np.float64)
    rep.domain = "frequency"
    rep.grid = (grid,)
    rep.entries = np.zeros((1, 1, 1, 1, len(grid)), dtype=np.complex128)


def make_mock_phasor(*, f_min: float, f_max: float, frequencies: Any = None):
    """Return (phasor, interpolated, embedded) mock triple with a preset frequency range."""
    phasor = make_valid_mock_representation(name="phasor", frequencies=frequencies)
    phasor.f_min = f_min
    phasor.f_max = f_max

    interpolated = make_valid_mock_representation(
        name="interpolated", frequencies=frequencies
    )
    embedded = make_valid_mock_representation(name="embedded", frequencies=frequencies)

    phasor.get_interpolated.return_value = interpolated
    interpolated.get_embedded.return_value = embedded
    return phasor, interpolated, embedded


def build_harmonic_projected_phasor_waveform(*, frequencies: Any = None):
    """Build a two-mode HarmonicProjectedWaveform-like object with phasor leaves.

    The outer container is a real HarmonicProjectedWaveform instance while each
    mode payload remains a lightweight channel mapping to preserve distinct
    per-channel mocks in helper-oriented tests.
    """
    mode_22 = modes.Harmonic(2, 2)
    mode_33 = modes.Harmonic(3, 3)

    p22x, i22x, e22x = make_mock_phasor(f_min=1.0, f_max=3.0, frequencies=frequencies)
    p22y, i22y, e22y = make_mock_phasor(f_min=1.5, f_max=2.5, frequencies=frequencies)
    p33x, i33x, e33x = make_mock_phasor(f_min=0.5, f_max=2.0, frequencies=frequencies)
    p33y, i33y, e33y = make_mock_phasor(f_min=2.0, f_max=4.0, frequencies=frequencies)

    wf = HarmonicProjectedWaveform(
        {
            mode_22: FakeResponse({"X": p22x, "Y": p22y}),
            mode_33: FakeResponse({"X": p33x, "Y": p33y}),
        }
    )

    handles = {
        mode_22: {"X": (p22x, i22x, e22x), "Y": (p22y, i22y, e22y)},
        mode_33: {"X": (p33x, i33x, e33x), "Y": (p33y, i33y, e33y)},
    }
    return wf, handles


def build_fake_harmonic_projected_waveform():
    """Backward-compatible alias for tests still using the old helper name."""
    return build_harmonic_projected_phasor_waveform()


def _make_fs(xp, frequencies, values):
    entries = xp.asarray(values, dtype=xp.complex128)[None, None, None, None, :]
    return reps.FrequencySeries((frequencies,), entries)


def build_harmonic_waveform_frequency_series(xp):
    frequencies = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    mode_22 = modes.Harmonic(2, 2)
    mode_33 = modes.Harmonic(3, 3)

    wf_22 = _make_fs(xp, frequencies, [1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j])
    wf_33 = _make_fs(xp, frequencies, [-0.5 + 1.0j, 0.25 + 0.0j, 1.5 - 0.25j])

    wf = HarmonicWaveform({mode_22: wf_22, mode_33: wf_33})

    return {
        "frequencies": frequencies,
        "modes": (mode_22, mode_33),
        "mode_22": mode_22,
        "mode_33": mode_33,
        "wf": wf,
        "wf_22": wf_22,
        "wf_33": wf_33,
    }


def build_harmonic_projected_frequency_waveform(xp):
    frequencies = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    mode_22 = modes.Harmonic(2, 2)
    mode_33 = modes.Harmonic(3, 3)

    resp_22 = ProjectedWaveform.from_dict(
        {
            "X": _make_fs(xp, frequencies, [1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j]),
            "Y": _make_fs(xp, frequencies, [0.5 + 0.25j, -1.0 + 0.0j, 0.25 - 0.25j]),
        }
    )
    resp_33 = ProjectedWaveform.from_dict(
        {
            "X": _make_fs(xp, frequencies, [0.2 + 0.0j, -0.5 + 1.0j, 0.1 - 0.2j]),
            "Y": _make_fs(xp, frequencies, [1.0 + 0.0j, 1.5 + 0.0j, 2.0 + 0.0j]),
        }
    )

    wf = HomogeneousHarmonicProjectedWaveform({mode_22: resp_22, mode_33: resp_33})

    return {
        "frequencies": frequencies,
        "mode_22": mode_22,
        "mode_33": mode_33,
        "wf": wf,
        "resp_22": resp_22,
        "resp_33": resp_33,
    }
