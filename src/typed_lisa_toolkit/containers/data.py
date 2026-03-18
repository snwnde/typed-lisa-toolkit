"""Module for data containers.

.. currentmodule:: typed_lisa_toolkit.containers.data

.. autoclass:: TSData
   :members:
   :member-order: groupwise

.. autoclass:: FSData
   :members:
   :member-order: groupwise

.. autoclass:: TimedFSData
   :members:
   :member-order: groupwise
   :show-inheritance:

.. autoclass:: STFTData
   :members:
   :member-order: groupwise

.. autoclass:: WDMData
   :members:
   :member-order: groupwise

Functions
---------

.. autofunction:: load_data
.. autofunction:: load_ldc_data

"""

from __future__ import annotations

import logging
import pathlib
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload

import array_api_compat as xpc
import h5py
import numpy as np
import numpy.typing as npt

from .. import lib
from . import representations as reps
from . import tapering

if TYPE_CHECKING:
    from . import waveforms as wf

log = logging.getLogger(__name__)

Reps = reps.TimeSeries | reps.FrequencySeries | reps.STFT | reps.Phasor | reps.WDM


class _GetSubsetMixin[RepT: reps.TimeSeries | reps.FrequencySeries | reps.Phasor](
    Mapping[str, RepT]
):
    """Mixin class to provide get_subset method for data containers."""

    representation: RepT
    channel_names: tuple[str, ...]

    def create_new(self, representation: RepT, channels: tuple[str, ...]) -> Self:
        del representation, channels
        raise NotImplementedError("This method must be implemented by subclass.")

    def get_subset(
        self, *, interval: tuple[float, float] | None = None, slice: slice | None = None
    ) -> Self:
        """Return the subset as a new instance."""
        subset = self.representation.get_subset(interval=interval, slice=slice)
        return self.create_new(
            subset,  # type: ignore[arg-type] # mypy's inference is confused
            self.channel_names,
        )

    def _get_plotter(self) -> type[Any]:
        raise NotImplementedError("This method must be implemented by subclass.")

    def draw(
        self,
        compare_to: Self | None = None,
        *,
        interval: tuple[float, float] | None = None,
        **kwargs: Any,
    ):
        """Plot the data.

        If `compare_to` is not `None`, the method draws both the data and the data in `compare_to`.
        """
        plotter = self._get_plotter()

        if compare_to is None:
            return plotter(self.get_subset(interval=interval)).draw(**kwargs)
        return plotter(self.get_subset(interval=interval)).compare(
            plotter(compare_to.get_subset(interval=interval)), **kwargs
        )


class _EmbeddableMixin[RepT: reps.TimeSeries | reps.FrequencySeries](
    Mapping[str, RepT]
):
    """Mixin class to provide get_embedded method for data containers."""

    representation: RepT
    channel_names: tuple[str, ...]

    def create_new(self, representation: RepT, channels: tuple[str, ...]) -> Self:
        del representation, channels
        raise NotImplementedError("This mixin should not be instantiated directly.")

    def get_embedded(self, embedding_grid: npt.NDArray[np.number]) -> Self:
        """Return the embedded data."""
        embedded_repr = self.representation.get_embedded(embedding_grid)
        return self.create_new(embedded_repr, self.channel_names)  # type: ignore[arg-type]
        # mypy's inference is confused


class _ChannelMapping[RepT: Reps](Mapping[str, RepT], lib.mixins.NDArrayMixin):
    def __init__(
        self,
        representation: RepT,
        channels: tuple[str, ...],
        name: str | None = None,
    ):
        entries = cast(Any, representation.entries)
        if entries.shape[1] != len(channels):
            raise ValueError(
                "Channel count mismatch between representation entries and channel names."
            )
        if entries.shape[2] != 1:
            raise ValueError(
                "Data containers require n_harmonics=1 in the representation entries."
            )
        self.representation = representation
        self.channel_names = tuple(channels)
        self._channel_to_idx = {chn: i for i, chn in enumerate(channels)}
        self.name = name

    def create_new(self, representation: RepT, channels: tuple[str, ...]) -> Self:
        """Create a new instance with a representation and channels."""
        return type(self)(representation, channels, self.name)

    def create_like(self, entries: "reps.Array", channels: tuple[str, ...]) -> Self:
        """Create a new instance with different entries but the same grid and type."""
        return type(self)(self.representation.create_like(entries), channels, self.name)

    def __xp__(self, api_version: str | None = None) -> Any:
        """Get the array namespace from the representation entries."""
        return xpc.get_namespace(self.representation.entries, api_version=api_version)

    def _binary_op(
        self,
        other: object,
        op: Callable[[object, object], object],
        /,
        reflected: bool = False,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Apply binary operation using native array ops on representations."""
        del kwargs  # Unused

        if isinstance(other, _ChannelMapping):
            other = cast(_ChannelMapping[Any], other)
            if self.channel_names != other.channel_names:
                raise ValueError("Cannot operate on data with different channel sets.")
            if reflected:
                new_repr = op(other.representation, self.representation)
            else:
                new_repr = op(self.representation, other.representation)
        else:
            # Scalar or array-like broadcast
            if reflected:
                new_repr = op(other, self.representation)
            else:
                new_repr = op(self.representation, other)

        if inplace:
            self.representation = cast(RepT, new_repr)
            return self

        return self.create_new(cast(RepT, new_repr), self.channel_names)

    def _unary_op(self, op: Callable[[object], object], /, **kwargs: Any) -> Self:
        """Apply unary operation using native array ops."""
        new_repr = op(self.representation, **kwargs)
        return self.create_new(cast(RepT, new_repr), self.channel_names)

    def pick(self, channels: str | tuple[str, ...]) -> Self:
        """Pick specific channels.

        Parameters
        ----------
        channels : str or tuple[str, ...]
            Channel name(s) to pick.

        Returns
        -------
        Self
            New instance with only the specified channels.
        """
        if isinstance(channels, str):
            channels = (channels,)

        indices = tuple([self._channel_to_idx[chn] for chn in channels])
        # Slice entries to pick only these channels (canonical shape: n_batches, n_channels, ...)
        xp = xpc.get_namespace(self.representation.entries)
        picked_entries = xp.asarray(self.representation.entries)[:, indices, ...]
        picked_repr = self.representation.create_like(picked_entries)
        return self.create_new(picked_repr, channels)

    @classmethod
    def from_dict(cls, data_dict: Mapping[str, RepT]) -> Self:
        """Create a new instance from a dictionary of channel names to representations."""
        if len(data_dict) == 0:
            raise ValueError("Cannot build data container from an empty mapping.")
        channels = tuple(data_dict.keys())
        xp = xpc.get_namespace(*(cast(Any, data_dict[chn]).entries for chn in channels))
        # Concatenate entries along the channel dimension (canonical shape: n_batches, n_channels, ...)
        entries = xp.concatenate([data_dict[chn].entries for chn in channels], axis=1)
        # Assume all representations have the same grid and type
        first = next(iter(data_dict.values()))
        return cls(first.create_like(entries), channels)

    def set_name(self, name: str | None) -> Self:
        """Set the name of the data container.

        .. note:: The name is only used for labeling the data container in plots.

        .. note:: This method returns `self` to allow for fluent method chaining.
        """
        self.name = name
        return self

    # Implement Mapping protocol
    def __getitem__(self, key: str) -> RepT:
        """Get a channel by name as a view with preserved channel dimension (size 1)."""
        idx = self._channel_to_idx[key]
        entries_view = self.representation.entries[:, idx : idx + 1, 0:1, ...]
        return self.representation.create_like(entries_view)

    def __iter__(self) -> Iterator[str]:
        """Iterate over channel names."""
        return iter(self.channel_names)

    def __len__(self) -> int:
        """Return the number of channels."""
        return len(self.channel_names)

    def __repr__(self):
        items = {key: self[key] for key in self}
        if self.name is not None:
            return f"{self.__class__.__name__}(name={self.name!r}, items={items!r})"
        return f"{self.__class__.__name__}({items!r})"

    @property
    def grid(self):
        """Return the grid."""
        return self.representation.grid

    @property
    def domain(self):
        """Physical domain shared by all channels."""
        return self.representation.domain

    @property
    def kind(self):
        """Semantic kind shared by all channels."""
        return self.representation.kind

    @property
    def entries(self) -> Any:
        """Return the raw entries array for direct matrix operations.

        Shape is (n_channels, ...) where ... depends on the representation type.
        Use this for complicated array operations with numpy/jax.
        """
        return self.representation.entries

    def get_kernel(self):
        """Return kernel entries in conventional shape."""
        return self.representation.entries


class Data[RepT: Reps](_ChannelMapping[RepT]):
    """Channel-indexed data containers.

    Stores a single homogeneous representation with channels as the first dimension,
    providing per-channel access via views and the Mapping protocol.
    """

    _reps_type: type[RepT]

    def _get_plotter(self) -> type[Any]:
        """Return the plotter class."""
        raise NotImplementedError("The method must be implemented in the subclass.")

    def _additional_save(self, f: h5py.File):
        del f

    def save(self, file_path: str | pathlib.Path):
        """Save the data to an HDF5 file.

        HDF5 File Structure
        -------------------
        The data are saved in the following structure:

        - The root level contains the attribute 'type' with the class name.
        - Each channel is saved as a group.
        - The group contains two datasets:

          - `grid` for the grid.
          - `entries` for the entries.

        - For :class:`.TimedFSData`, there will be a dataset 'times' at the root level containing the time grid.

        """
        with h5py.File(str(file_path), "a") as f:
            f.attrs["type"] = self.__class__.__name__
            f.attrs["channels"] = self.channel_names
            self._additional_save(f)
            grp = f.create_group("data")
            representation = cast(Any, self.representation)
            grp.create_dataset("grid", data=representation.grid)
            grp.create_dataset("entries", data=representation.entries)

    @classmethod
    def _additional_load(cls, f: h5py.File) -> dict[str, Any]:
        del f
        return {}

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> Self:
        """Load the data from an HDF5 file."""
        with h5py.File(str(file_path), "r") as f:
            channels_attr = cast(Iterable[Any], f.attrs["channels"])
            channels = tuple(str(ch) for ch in channels_attr)
            additions = cls._additional_load(f)
            data_group = cast(h5py.Group, f["data"])
            grid_data = cast(h5py.Dataset, data_group["grid"])[()]
            entries_data = cast(h5py.Dataset, data_group["entries"])[()]
            repr_obj = cls._reps_type(grid_data, entries_data)
        return cls(repr_obj, channels, **additions)  # type: ignore[arg-type] # mypy's inference is confused

    @classmethod
    def from_waveform(cls, waveform: "wf.ProjectedWaveform[RepT]"):
        """Create a new instance from a projected waveform."""
        return cls(waveform.representation, waveform.channel_names)


class _SeriesData[RepT: reps.TimeSeries | reps.FrequencySeries](
    Data[RepT], _GetSubsetMixin[RepT], _EmbeddableMixin[RepT]
): ...


class TSData(_SeriesData[reps.TimeSeries]):
    """Multi-channel time series data container."""

    _reps_type = reps.TimeSeries

    @property
    def times(self) -> npt.NDArray[np.floating[Any]]:
        """Return the times."""
        representation = cast(Any, self.representation)
        return np.asarray(representation.times)

    @property
    def dt(self) -> float:
        """Return the time step."""
        return self.representation.dt

    @property
    def t_start(self) -> float:
        """Return the start time."""
        return self.representation.t_start

    @property
    def t_end(self) -> float:
        """Return the end time."""
        return self.representation.t_end

    def get_frequencies(self):
        """Return the frequencies grid matching the time grid."""
        return np.fft.rfftfreq(len(self.times), d=self.dt)

    @overload
    def to_fsdata(
        self,
        *,
        keep_times: Literal[False],
        tapering: tapering.Tapering | None = None,
    ) -> FSData: ...

    @overload
    def to_fsdata(
        self,
        *,
        keep_times: Literal[True] = True,
        tapering: tapering.Tapering | None = None,
    ) -> TimedFSData: ...

    def to_fsdata(
        self,
        *,
        keep_times: bool = True,
        tapering: tapering.Tapering | None = None,
    ):
        """Return the frequency series data.

        Returns
        -------
        :class:`.FSData` | :class:`.TimedFSData`
            The frequency series data. If `keep_times` is `True`, the time grid is kept.
        """
        fsrepr = self.representation.rfft(tapering=tapering)
        if keep_times:
            return TimedFSData(fsrepr, self.channel_names, times=self.times)
        return FSData(fsrepr, self.channel_names)

    def to_stftdata(
        self,
        *,
        win: npt.NDArray[np.floating[Any]],
        hop: int,
    ) -> STFTData:
        """Return the time-frequency data."""
        tfrepr = self.representation.stfft(win, hop)
        tf_entries = cast(Any, tfrepr.entries)
        tfdict = {
            chn: reps.STFT(
                tfrepr.grid,
                tf_entries[:, idx : idx + 1, 0:1, ...],
            )
            for idx, chn in enumerate(self.channel_names)
        }
        return STFTData.from_dict(tfdict).set_name(self.name)

    def get_zero_padded(
        self,
        pad_time: tuple[float, float],
        tapering: tapering.Tapering | None = None,
    ) -> Self:
        """Return the zero-padded data."""
        xp = xpc.get_namespace(self.entries)
        pad_width = tuple(int(xp.rint(time / self.dt)) for time in pad_time)
        time_end_values = (
            -self.dt * pad_width[0] + self.times[0],
            self.dt * pad_width[1] + self.times[-1],
        )
        padded_time = xp.pad(
            self.times,
            pad_width,
            mode="linear_ramp",
            end_values=time_end_values,
        )

        tapering_window = cast(Any, tapering(self.times)) if tapering is not None else 1
        signal = self.entries * tapering_window
        padded_signal = xp.pad(
            signal,
            ((0, 0), (0, 0), (0, 0), (0, 0), pad_width),
            mode="constant",
        )

        padded_repr = reps.TimeSeries(padded_time, padded_signal)
        return self.create_new(padded_repr, self.channel_names)

    def _get_plotter(self):
        from ..viz import plotters

        return plotters.TSDataPlotter


class FSData(_SeriesData[reps.FrequencySeries]):
    """Multi-channel frequency series data container."""

    _reps_type = reps.FrequencySeries

    @property
    def frequencies(self):
        """Return the frequencies."""
        return self.representation.frequencies

    @property
    def df(self):
        """Return the frequency step."""
        return self.representation.df

    @property
    def f_min(self):
        """Return the minimum frequency."""
        return self.representation.f_min

    @property
    def f_max(self):
        """Return the maximum frequency."""
        return self.representation.f_max

    def set_times(self, times: npt.NDArray[np.floating[Any]]):
        """Set the time grid."""
        return TimedFSData(self.representation, self.channel_names, times)

    def to_tsdata(
        self,
        times: npt.NDArray[np.floating[Any]] | reps.Linspace,
        *,
        tapering: tapering.Tapering | None = None,
    ):
        """Return the time series data."""
        tsrepr = self.representation.irfft(np.asarray(times), tapering=tapering)
        return TSData(tsrepr, self.channel_names)

    def _get_plotter(self):
        from ..viz import plotters

        return plotters.FSDataPlotter

    def to_wdm_data(
        self,
        *args: int | float | None,
        Nf: int | None = None,
        Nt: int | None = None,
        nx: float = 4.0,
    ) -> WDMData:
        """Return the WDM data.

        See :py:meth:`.representations.FrequencySeries.to_wdm`
        """
        if len(args) > 3:
            raise TypeError(
                "to_wdm_data() accepts at most 3 positional optional arguments: Nf, Nt, nx."
            )
        if len(args) > 0:
            if Nf is not None or Nt is not None or nx != 4.0:
                raise TypeError(
                    "to_wdm_data() received optional arguments as both positional and keyword arguments."
                )
            warnings.warn(
                "Passing `Nf`, `Nt`, or `nx` positionally to `to_wdm_data` is deprecated and "
                "will be removed in 0.7.0; pass them as keyword arguments instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if len(args) >= 1:
                Nf = args[0] if args[0] is None else int(args[0])
            if len(args) >= 2:
                Nt = args[1] if args[1] is None else int(args[1])
            if len(args) == 3:
                nx = float(args[2])  # type: ignore[assignment]

        wdmdict = {
            chn: self[chn].to_wdm(Nf=Nf, Nt=Nt, nx=nx) for chn in self.channel_names
        }
        return WDMData.from_dict(wdmdict).set_name(self.name)

    def to_WDMdata(
        self,
        Nf: int | None = None,
        Nt: int | None = None,
        nx: float = 4.0,
    ) -> WDMData:
        """Deprecated alias for :meth:`to_wdm_data`.

        This alias will be removed in 0.7.0.
        """  # noqa: D401
        warnings.warn(
            "`to_WDMdata` is deprecated and will be removed in 0.7.0; "
            "use `to_wdm_data` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_wdm_data(Nf=Nf, Nt=Nt, nx=nx)


class TimedFSData(FSData):
    """Multi-channel frequency series data with time information."""

    def __init__(
        self,
        representation: reps.FrequencySeries,
        channels: tuple[str, ...],
        times: reps.Linspace | npt.NDArray[np.floating[Any]] | None = None,
        name: str | None = None,
    ):
        super().__init__(representation, channels, name)
        if times is None:
            raise ValueError("TimedFSData requires times parameter")
        self.times = reps.Linspace.make(times)
        self.dt = self.times.step

    def _additional_save(self, f: h5py.File):
        f.create_dataset("times", data=np.asarray(self.times))

    @classmethod
    def _additional_load(cls, f: h5py.File):
        times_data = cast(h5py.Dataset, f["times"])[()]
        return {"times": times_data}

    def set_times(self, times: npt.NDArray[np.floating[Any]] | reps.Linspace):
        """Set the time grid.

        This method returns `self` to allow for fluent method chaining.
        """
        self.times = reps.Linspace.make(times)
        self.dt = self.times.step
        return self

    def create_new(
        self, representation: reps.FrequencySeries, channels: tuple[str, ...]
    ):
        """Create a new instance preserving times."""
        new = type(self)(representation, channels, times=self.times, name=self.name)
        return new

    def drop_times(self):
        """Drop the time grid."""
        return FSData(self.representation, self.channel_names, self.name)

    def to_tsdata(
        self,
        times: npt.NDArray[np.floating[Any]] | reps.Linspace | None = None,
        *,
        tapering: tapering.Tapering | None = None,
    ):
        """Return the time series data with times grid."""
        del times
        tsrepr = self.representation.irfft(np.asarray(self.times), tapering=tapering)
        return TSData(tsrepr, self.channel_names, self.name)


class STFTData(Data[reps.STFT]):
    """Dictionary data container of short-time Fourier transform data."""

    def get_subset(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
    ):
        """Return the subset as a new instance."""
        tfdict = {
            chnname: chn.get_subset(
                time_interval=time_interval, freq_interval=freq_interval
            )
            for chnname, chn in self.items()
        }
        return type(self).from_dict(tfdict).set_name(self.name)

    # def draw(
    #     self,
    #     *,
    #     time_interval: tuple[float, float] | None = None,
    #     freq_interval: tuple[float, float] | None = None,
    #     **kwargs: Any,
    # ):
    #     """Plot the data."""
    #     from ..viz import plotters

    # plotter = plotters.TFDataPlotter(
    #     self.get_subset(
    #         time_interval=time_interval,
    #         freq_interval=freq_interval,
    #     )
    # )
    #     return plotter.draw(**kwargs)


class WDMData(Data[reps.WDM]):
    """Dictionary data container of WDM time-frequency data."""

    _reps_type = reps.WDM

    def get_subset(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
    ):
        """Return the subset as a new instance."""
        tfdict = {
            chnname: chn.get_subset(
                time_interval=time_interval, freq_interval=freq_interval
            )
            for chnname, chn in self.items()
        }
        return type(self).from_dict(tfdict).set_name(self.name)

    def to_fsdata(self, *, nx: float = 4.0, mask: Any | None = None):
        """Convert to FSData.

        See :py:meth:`.representations.WDM.to_frequency_series`.
        """
        fsdict = {
            chnname: chn.to_frequency_series(nx=nx, mask=mask)
            for chnname, chn in self.items()
        }
        return FSData.from_dict(fsdict)

    def draw(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
        **kwargs: Any,
    ):
        """Plot the data."""
        from ..viz import plotters

        plotter = plotters.TFDataPlotter(
            self.get_subset(
                time_interval=time_interval,
                freq_interval=freq_interval,
            )
        )
        return plotter.draw(**kwargs)


def load_data(file_path: str | pathlib.Path):
    """Load the data from a saved HDF5 file."""
    with h5py.File(str(file_path), "r") as f:
        data_type = str(f.attrs["type"])
    if data_type == "TSData":
        return TSData.load(file_path)
    if data_type == "FSData":
        return FSData.load(file_path)
    if data_type == "TimedFSData":
        return TimedFSData.load(file_path)
    raise ValueError(f"Unknown data type: {data_type}")


def load_ldc_data(
    file_path: str | pathlib.Path,
    name: Literal[
        "obs/tdi", "sky/mbhb/tdi", "sky/igb/tdi", "sky/vgb/tdi", "sky/dgb/tdi"
    ] = "obs/tdi",
    channels: Literal["AE", "AET", "XYZ"] = "AE",
) -> TSData:
    """Load the data from LDC dataset."""

    def XYZ2AET(
        X: reps.TimeSeries,
        Y: reps.TimeSeries,
        Z: reps.TimeSeries,
    ) -> tuple[reps.TimeSeries, reps.TimeSeries, reps.TimeSeries]:
        A, E, T = (
            (Z - X) / np.sqrt(2.0),
            (X - 2.0 * Y + Z) / np.sqrt(6.0),
            (X + Y + Z) / np.sqrt(3.0),
        )
        return A, E, T

    def AET2XYZ(
        A: reps.TimeSeries,
        E: reps.TimeSeries,
        T: reps.TimeSeries,
    ) -> tuple[reps.TimeSeries, reps.TimeSeries, reps.TimeSeries]:
        X, Y, Z = [
            -1 / np.sqrt(2.0) * A + 1 / np.sqrt(6.0) * E + 1 / np.sqrt(3.0) * T,
            -np.sqrt(2.0 / 3.0) * E + 1 / np.sqrt(3.0) * T,
            1 / np.sqrt(2.0) * A + 1 / np.sqrt(6.0) * E + 1 / np.sqrt(3.0) * T,
        ]
        return X, Y, Z

    def transform_channels(data: TSData) -> TSData:
        channel_names = tuple(name for name in channels)
        if set(channel_names).issubset(data.channel_names):
            return data.pick(channel_names)
        if set(channel_names).issubset(("X", "Y", "Z")):
            X, Y, Z = AET2XYZ(data["A"], data["E"], data["T"])
            return type(data).from_dict({"X": X, "Y": Y, "Z": Z}).pick(channel_names)
        if set(channel_names).issubset(("A", "E", "T")):
            A, E, T = XYZ2AET(data["X"], data["Y"], data["Z"])
            return type(data).from_dict({"A": A, "E": E, "T": T}).pick(channel_names)
        raise ValueError(
            f"The data does not have the expected channels. The data has {data.channel_names}."
        )

    with h5py.File(str(file_path), "r") as fid:
        dataset = np.array(fid[name])
    if dtype_names := dataset.dtype.names:
        domain = dtype_names[0]
        channel_names = dtype_names[1:]
        # I assume we only need to read LDC data in the time domain.
        # When this is not true anymore, we can enhance this function.
        if domain == "t":
            tsdata = TSData.from_dict(
                {
                    chnname: reps.TimeSeries(
                        grid=(dataset[domain].squeeze(),),
                        entries=dataset[chnname].squeeze()[None, None, None, None, :],
                    )
                    for chnname in channel_names
                }
            )
            return transform_channels(tsdata)
    raise ValueError("The dataset does not have the expected structure.")
