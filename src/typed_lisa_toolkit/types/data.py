"""Data container types."""

from __future__ import annotations

import abc
import logging
import pathlib

# import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast, overload

import array_api_compat as xpc
import h5py
import numpy as np
import numpy.typing as npt

from . import _mixins, tapering
from . import representations as reps
from .misc import AnyGrid, Axis, Linspace

if TYPE_CHECKING:
    from . import waveforms as wf
    from .representations import Representation

    AnyReps = Representation[AnyGrid]

    class _SubsettableRep(AnyReps, Protocol):
        def get_subset(
            self,
            *,
            interval: tuple[float, float] | None = None,
            slice: slice | None = None,
        ) -> Self: ...


log = logging.getLogger(__name__)


class _GetSubsetMixin[RepT: "_SubsettableRep"](Mapping[str, RepT], abc.ABC):
    """Mixin class to provide get_subset method for data containers."""

    @property
    @abc.abstractmethod
    def channel_names(self) -> tuple[str, ...]:
        """Return the channel names."""

    @property
    @abc.abstractmethod
    def representation(self) -> RepT:
        """Return the underlying representation."""

    @abc.abstractmethod
    def create_new(self, representation: RepT, channels: tuple[str, ...]) -> Self: ...

    def get_subset(
        self, *, interval: tuple[float, float] | None = None, slice: slice | None = None
    ) -> Self:
        """Return the subset as a new instance."""
        subset = self.representation.get_subset(interval=interval, slice=slice)
        return self.create_new(
            subset,
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


class Data[RepT: "AnyReps"](_mixins.ChannelMapping[RepT]):
    """Channel-indexed data containers.

    Stores a single homogeneous representation with channels as the first dimension,
    providing per-channel access via views and the Mapping protocol.
    """

    _reps_type: type[RepT]

    def _init_repr(self, representation: "AnyReps"):
        self._representation: "AnyReps" = self._reps_type(
            representation.grid, representation.entries
        )

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
            grp.create_dataset("grid", data=self.representation.grid)
            grp.create_dataset("entries", data=cast(Any, self.representation.entries))

    @classmethod
    def _additional_load(cls, f: h5py.File) -> dict[str, Any]:
        del f
        return {}

    @classmethod
    def _load_legacy(cls, file_path: str | pathlib.Path) -> Self:
        """Load the data from an HDF5 file."""
        with h5py.File(str(file_path), "r") as f:
            additions = cls._additional_load(f)
            dict_ = {
                chnname: cls._reps_type(
                    grid=(f[chnname]["grid"][...],),  # type: ignore
                    entries=f[chnname]["entries"][...][None, None, None, None, ...],  # type: ignore
                )
                for chnname in f
                if isinstance(f[chnname], h5py.Group)
            }
        return cls.from_dict(dict_, **additions)

    @classmethod
    def load(cls, file_path: str | pathlib.Path, legacy: bool = False) -> Self:
        """Load the data from an HDF5 file."""
        if legacy:
            return cls._load_legacy(file_path)
        with h5py.File(str(file_path), "r") as f:
            channels_attr = cast(Iterable[object], f.attrs["channels"])
            channels = tuple(str(ch) for ch in channels_attr)
            additions = cls._additional_load(f)
            data_group = cast(h5py.Group, f["data"])
            grid_data = cast(h5py.Dataset, data_group["grid"])[()]
            entries_data = cast(h5py.Dataset, data_group["entries"])[()]
            repr_obj = cls._reps_type(grid_data, entries_data)
        return cls(repr_obj, channels, **additions)

    @classmethod
    def from_waveform(cls, waveform: "wf.ProjectedWaveform[RepT]"):
        """Create a new instance from a projected waveform."""
        return cls(waveform.representation, waveform.channel_names)


class _SeriesData[RepT: reps.UniformTimeSeries | reps.UniformFrequencySeries](
    Data[RepT], _GetSubsetMixin[RepT]
): ...


class TSData(_SeriesData[reps.UniformTimeSeries]):
    """Multi-channel time series data container."""

    _reps_type: type[reps.UniformTimeSeries] = reps.UniformTimeSeries

    @property
    def times(self):
        """Return the times."""
        return self.representation.times

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

    def get_embedded(self, embedding_grid: "reps.Grid1D[Axis]"):
        """Return data embedded on a new 1D grid."""
        embedded = self.representation.get_embedded(embedding_grid)
        return type(self)(
            embedded,
            self.channel_names,
        ).set_name(self.name)

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
        """Return the frequency series data (*Deprecated*).

        Returns
        -------
        :class:`.FSData` | :class:`.TimedFSData`
            The frequency series data. If `keep_times` is `True`, the time grid is kept.

        .. warning::
            This method is deprecated and will be removed in 0.8.0; use `shop.time2freq` instead.
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
        tf_entries = tfrepr.entries
        tfdict = {
            chn: reps.STFT["Linspace", "Linspace"](
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
        _times = xp.asarray(self.times)
        pad_width = tuple(int(xp.rint(time / self.dt)) for time in pad_time)
        time_end_values = (
            -self.dt * pad_width[0] + self.times.start,
            self.dt * pad_width[1] + self.times.stop,
        )
        padded_time = xp.pad(
            _times,
            pad_width,
            mode="linear_ramp",
            end_values=time_end_values,
        )

        tapering_window = tapering(_times) if tapering is not None else 1
        signal = self.entries * tapering_window
        padded_signal = xp.pad(
            signal,
            ((0, 0), (0, 0), (0, 0), (0, 0), pad_width),
            mode="constant",
        )

        padded_repr = reps.UniformTimeSeries((padded_time,), padded_signal)
        return self.create_new(padded_repr, self.channel_names)

    def _get_plotter(self):
        from ..viz import plotters

        return plotters.TSDataPlotter


class FSData(_SeriesData[reps.UniformFrequencySeries]):
    """Multi-channel frequency series data container."""

    _reps_type: type[reps.UniformFrequencySeries] = reps.UniformFrequencySeries

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

    def set_times(self, times: "Axis"):
        """Set the time grid."""
        return TimedFSData(self.representation, self.channel_names, times)

    def get_embedded(self, embedding_grid: "reps.Grid1D[Axis]"):
        """Return data embedded on a new 1D grid."""
        embedded = self.representation.get_embedded(embedding_grid)
        return type(self)(
            embedded,
            self.channel_names,
        ).set_name(self.name)

    def to_tsdata(
        self,
        times: npt.NDArray[np.floating[Any]] | Linspace,
        *,
        tapering: tapering.Tapering | None = None,
    ):
        """Return the time series data (*Deprecated*).

        .. warning::
            This method is deprecated and will be removed in 0.8.0; use
            `shop.freq2time` instead.
        """
        tsrepr = self.representation.irfft(np.asarray(times), tapering=tapering)
        return TSData(tsrepr, self.channel_names)

    def _get_plotter(self):
        from ..viz import plotters

        return plotters.FSDataPlotter

    # def to_wdm_data(
    #     self,
    #     *args: int | float | None,
    #     Nf: int | None = None,
    #     Nt: int | None = None,
    #     nx: float = 4.0,
    # ) -> WDMData:
    #     """Return the WDM data.

    #     See :py:meth:`.representations.FrequencySeries.to_wdm`
    #     """
    #     if len(args) > 3:
    #         raise TypeError(
    #             "to_wdm_data() accepts at most 3 positional optional arguments: Nf, Nt, nx."
    #         )
    #     if len(args) > 0:
    #         if Nf is not None or Nt is not None or nx != 4.0:
    #             raise TypeError(
    #                 "to_wdm_data() received optional arguments as both positional and keyword arguments."
    #             )
    #         warnings.warn(
    #             "Passing `Nf`, `Nt`, or `nx` positionally to `to_wdm_data` is deprecated and "
    #             + "will be removed in 0.7.0; pass them as keyword arguments instead.",
    #             DeprecationWarning,
    #             stacklevel=2,
    #         )
    #         if len(args) >= 1:
    #             Nf = args[0] if args[0] is None else int(args[0])
    #         if len(args) >= 2:
    #             Nt = args[1] if args[1] is None else int(args[1])
    #         if len(args) == 3:
    #             nx = float(args[2])

    #     wdmdict = {
    #         chn: self[chn].to_wdm(Nf=Nf, Nt=Nt, nx=nx) for chn in self.channel_names
    #     }
    #     return WDMData.from_dict(wdmdict).set_name(self.name)

    # def to_WDMdata(
    #     self,
    #     Nf: int | None = None,
    #     Nt: int | None = None,
    #     nx: float = 4.0,
    # ) -> WDMData:
    #     """Deprecated alias for :meth:`to_wdm_data`.

    #     This alias will be removed in 0.7.0.
    #     """  # noqa: D401
    #     warnings.warn(
    #         "`to_WDMdata` is deprecated and will be removed in 0.7.0; "
    #         + "use `to_wdm_data` instead.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     return self.to_wdm_data(Nf=Nf, Nt=Nt, nx=nx)


class TimedFSData(FSData):
    """Multi-channel frequency series data with time information."""

    def __init__(
        self,
        representation: "AnyReps",
        channels: tuple[str, ...],
        times: Axis | None = None,
        name: str | None = None,
    ):
        super().__init__(representation, channels, name)
        if times is None:
            raise ValueError("TimedFSData requires times parameter")
        self.set_times(times)

    def _additional_save(self, f: h5py.File):
        f.create_dataset("times", data=np.asarray(self.times))

    @classmethod
    def _additional_load(cls, f: h5py.File):
        times_data = cast(h5py.Dataset, f["times"])[()]
        return {"times": times_data}

    def set_times(self, times: Axis):
        """Set the time grid.

        This method returns `self` to allow for fluent method chaining.
        """
        self.times: Linspace = Linspace.make(times)
        self.dt: float = self.times.step
        return self

    def create_new(self, representation: "AnyReps", channels: tuple[str, ...]):
        """Create a new instance preserving times."""
        new = type(self)(representation, channels, times=self.times, name=self.name)
        return new

    def drop_times(self):
        """Drop the time grid."""
        return FSData(self.representation, self.channel_names, self.name)

    def to_tsdata(
        self,
        times: npt.NDArray[np.floating[Any]] | Linspace | None = None,
        *,
        tapering: tapering.Tapering | None = None,
    ):
        """Return the time series data with times grid (*Deprecated*).

        .. warning::
            This method is deprecated and will be removed in 0.8.0; use
            `to_tsdata(times=self.times, tapering=...)` instead.
        """
        del times
        tsrepr = self.representation.irfft(np.asarray(self.times), tapering=tapering)
        return TSData(tsrepr, self.channel_names, self.name)


class STFTData(Data[reps.STFT["Linspace", "Linspace"]]):
    """Dictionary data container of short-time Fourier transform data."""

    _reps_type: type[reps.STFT["Linspace", "Linspace"]] = reps.STFT[
        "Linspace", "Linspace"
    ]

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


class WDMData(Data[reps.WDM]):
    """Dictionary data container of WDM time-frequency data."""

    _reps_type: type[reps.WDM] = reps.WDM

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

    # def to_fsdata(self, *, nx: float = 4.0, mask: Any | None = None):
    #     """Convert to FSData.

    #     See :py:meth:`.representations.WDM.to_frequency_series`.
    #     """
    #     fsdict = {
    #         chnname: chn.to_frequency_series(nx=nx, mask=mask)
    #         for chnname, chn in self.items()
    #     }
    #     return FSData.from_dict(fsdict)

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


def tsdata(
    mapping: Mapping[str, reps.TimeSeries[Linspace]], name: str | None = None
) -> TSData:
    """Construct a :class:`~types.data.TSData` instance."""
    return TSData.from_dict(mapping, name=name)


def fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]], name: str | None = None
) -> FSData:
    """Construct a :class:`~types.data.FSData` instance."""
    return FSData.from_dict(mapping, name=name)


def timed_fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]],
    times: Linspace | npt.NDArray[np.floating[Any]],
    name: str | None = None,
) -> TimedFSData:
    """Construct a :class:`~types.data.TimedFSData` instance."""
    return TimedFSData.from_dict(mapping, times=times, name=name)


def stftdata(
    mapping: Mapping[str, reps.STFT[Linspace, Linspace]],
    name: str | None = None,
) -> STFTData:
    """Construct a :class:`~types.data.STFTData` instance."""
    return STFTData.from_dict(mapping, name=name)


def wdmdata(
    mapping: Mapping[str, reps.WDM],
    name: str | None = None,
) -> WDMData:
    """Construct a :class:`~types.data.WDMData` instance."""
    return WDMData.from_dict(mapping, name=name)


def load_data(file_path: str | pathlib.Path, legacy: bool = False):
    """Load the data from a saved HDF5 file."""
    with h5py.File(str(file_path), "r") as f:
        data_type = str(f.attrs["type"])
    if data_type == "TSData":
        return TSData.load(file_path, legacy=legacy)
    if data_type == "FSData":
        return FSData.load(file_path, legacy=legacy)
    if data_type == "TimedFSData":
        return TimedFSData.load(file_path, legacy=legacy)
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
        X: reps.UniformTimeSeries,
        Y: reps.UniformTimeSeries,
        Z: reps.UniformTimeSeries,
    ) -> tuple[reps.UniformTimeSeries, reps.UniformTimeSeries, reps.UniformTimeSeries]:
        A, E, T = (
            (Z - X) / np.sqrt(2.0),
            (X - 2.0 * Y + Z) / np.sqrt(6.0),
            (X + Y + Z) / np.sqrt(3.0),
        )
        return A, E, T

    def AET2XYZ(
        A: reps.UniformTimeSeries,
        E: reps.UniformTimeSeries,
        T: reps.UniformTimeSeries,
    ) -> tuple[reps.UniformTimeSeries, reps.UniformTimeSeries, reps.UniformTimeSeries]:
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
                    chnname: reps.UniformTimeSeries(
                        grid=(dataset[domain].squeeze(),),
                        entries=dataset[chnname].squeeze()[None, None, None, None, :],
                    )
                    for chnname in channel_names
                }
            )
            return transform_channels(tsdata)
    raise ValueError("The dataset does not have the expected structure.")
