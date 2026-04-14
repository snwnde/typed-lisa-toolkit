"""Data container types."""

from __future__ import annotations

import abc
import logging
import pathlib
import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast, overload

import array_api_compat as xpc
import h5py
import numpy as np
import numpy.typing as npt

from . import _mixins, tapering
from . import representations as reps
from .misc import (
    AnyGrid,
    Array,
    Axis,
    Domain,
    Grid1D,
    Grid2D,
    Grid2DCartesian,
    Grid2DSparse,
    Linspace,
    build_grid2d,
)

if TYPE_CHECKING:
    from MojitoProcessor import SignalProcessor

    # from .waveforms import ProjectedWaveform
    from .representations import Representation

    AnyReps = Representation[AnyGrid]

    class _1DSubsettableRep(AnyReps, Protocol):
        def get_subset(
            self,
            *,
            interval: tuple[float, float] | None = None,
            slice: slice | None = None,
        ) -> Self: ...

    class _2DSubsettableRep(AnyReps, Protocol):
        def get_subset(
            self,
            *,
            time_interval: tuple[float, float] | None = None,
            freq_interval: tuple[float, float] | None = None,
            slices: tuple[slice, slice] | None = None,
            copy: bool = True,
        ) -> Self: ...


log = logging.getLogger(__name__)


def _save_axis(grp: h5py.Group, name: str, axis: "Axis") -> None:
    """Serialize one grid axis, preserving Linspace compactness."""
    axis_grp = grp.create_group(name)
    if isinstance(axis, Linspace):
        axis_grp.attrs["linspace"] = True
        axis_grp.attrs["start"] = axis.start
        axis_grp.attrs["step"] = axis.step
        axis_grp.attrs["num"] = axis.num
        return

    axis_grp.create_dataset("values", data=np.asarray(axis))


def _attr_float(attrs: Any, key: str) -> float:
    return float(attrs[key])


def _attr_int(attrs: Any, key: str) -> int:
    return int(attrs[key])


def _attr_bool(attrs: Any, key: str, default: bool = False) -> bool:
    return bool(attrs.get(key, default))


def _load_axis(node: h5py.Group | h5py.Dataset) -> "Axis":
    """Deserialize one grid axis.

    Supports both new grouped axis format and old raw-dataset axis format.
    """
    if isinstance(node, h5py.Dataset):
        return cast("Axis", node[()])

    linspace = _attr_bool(node.attrs, "linspace", default=False)
    if linspace:
        return Linspace(
            start=_attr_float(node.attrs, "start"),
            step=_attr_float(node.attrs, "step"),
            num=_attr_int(node.attrs, "num"),
        )
    return cast("Axis", cast(h5py.Dataset, node["values"])[()])


def _save_grid(grp: h5py.Group, grid: "AnyGrid") -> None:
    """Serialize a representation grid into an HDF5 group."""
    if isinstance(grid, Grid2DSparse):
        grp.attrs["sparse"] = True
        grp.attrs["dim"] = 2
        _save_axis(grp, "axis0", grid[0])
        _save_axis(grp, "axis1", grid[1])
        grp.create_dataset("sparse_indices", data=np.asarray(grid.indices))
        return

    if len(grid) == 1:
        grp.attrs["dim"] = 1
        _save_axis(grp, "axis0", grid[0])
        return
    if len(grid) == 2:
        grp.attrs["sparse"] = False
        grp.attrs["dim"] = 2
        _save_axis(grp, "axis0", grid[0])
        _save_axis(grp, "axis1", grid[1])
        return


def _load_grid(node: h5py.Group | h5py.Dataset) -> "AnyGrid":
    """Deserialize a representation grid from HDF5."""
    # Backward compatibility: previous format stored `grid` as a single dataset.
    if isinstance(node, h5py.Dataset):
        return cast("AnyGrid", node[()])

    sparse = _attr_bool(node.attrs, "sparse", default=False)
    dim = _attr_int(node.attrs, "dim") if "dim" in node.attrs else -1
    if not sparse and dim == 1:
        axis0 = _load_axis(cast(h5py.Group | h5py.Dataset, node["axis0"]))
        return (axis0,)
    if not sparse and dim == 2:
        axis0 = _load_axis(cast(h5py.Group | h5py.Dataset, node["axis0"]))
        axis1 = _load_axis(cast(h5py.Group | h5py.Dataset, node["axis1"]))
        return (axis0, axis1)
    if sparse and dim == 2:
        axis0 = _load_axis(cast(h5py.Group | h5py.Dataset, node["axis0"]))
        axis1 = _load_axis(cast(h5py.Group | h5py.Dataset, node["axis1"]))
        sparse_indices = cast(h5py.Dataset, node["sparse_indices"])[()]
        return Grid2DSparse(axis0, axis1, sparse_indices=sparse_indices)

    raise ValueError(f"Unknown grid serialization format with attributes: {node.attrs}")


def _load_data[DataT: "Data[AnyReps]"](
    cls: type[DataT], file_path: str | pathlib.Path, legacy: bool = False
) -> DataT:
    """Load data from an HDF5 file."""
    if legacy:
        return cls._load_legacy(file_path)  # pyright: ignore[reportPrivateUsage]
    with h5py.File(str(file_path), "r") as f:
        channels_attr = cast(Iterable[object], f.attrs["channels"])
        channels = tuple(str(ch) for ch in channels_attr)
        additions = cls._additional_load(f)  # pyright: ignore[reportPrivateUsage]
        data_group = cast(h5py.Group, f["data"])
        grid_node = cast(h5py.Group | h5py.Dataset, data_group["grid"])
        grid_data = _load_grid(grid_node)
        entries_data = cast(h5py.Dataset, data_group["entries"])[()]
        return cls(grid_data, entries_data, channels=channels, **additions)


class _1DSubsetMixin[RepT: _1DSubsettableRep](_mixins.ChannelMapping[RepT], abc.ABC):
    def get_subset(
        self, *, interval: tuple[float, float] | None = None, slice: slice | None = None
    ) -> Self:
        """Return the subset as a new instance."""
        subset_dict = {
            chnname: chn.get_subset(interval=interval, slice=slice)
            for chnname, chn in self.items()
        }
        return type(self).from_dict(subset_dict).set_name(self.name)

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

    def _get_plotter(self) -> type[Any]:
        """Return the plotter class."""
        raise NotImplementedError("The method must be implemented in the subclass.")


class _2DSubsetMixin[RepT: _2DSubsettableRep](_mixins.ChannelMapping[RepT], abc.ABC):
    def get_subset(
        self,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
        slices: tuple[slice, slice] | None = None,
        copy: bool = True,
    ) -> Self:
        """Return the subset as a new instance."""
        subset_dict = {
            chnname: chn.get_subset(
                time_interval=time_interval,
                freq_interval=freq_interval,
                slices=slices,
                copy=copy,
            )
            for chnname, chn in self.items()
        }
        return type(self).from_dict(subset_dict).set_name(self.name)

    def draw(
        self,
        compare_to: Self | None = None,
        *,
        time_interval: tuple[float, float] | None = None,
        freq_interval: tuple[float, float] | None = None,
        **kwargs: Any,
    ):
        """Plot the data.

        If `compare_to` is not `None`, the method draws both the data and the data in `compare_to`.
        """
        plotter = self._get_plotter()

        if compare_to is None:
            return plotter(
                self.get_subset(
                    time_interval=time_interval, freq_interval=freq_interval
                )
            ).draw(**kwargs)
        return plotter(
            self.get_subset(time_interval=time_interval, freq_interval=freq_interval)
        ).compare(
            plotter(
                compare_to.get_subset(
                    time_interval=time_interval, freq_interval=freq_interval
                )
            ),
            **kwargs,
        )

    def _get_plotter(self) -> type[Any]:
        """Return the plotter class."""
        raise NotImplementedError("The method must be implemented in the subclass.")


class Data[RepT: "AnyReps"](_mixins.ChannelMapping[RepT], abc.ABC):
    """Channel-indexed data containers.

    Stores a single homogeneous _representation with channels as the first dimension,
    providing per-channel access via views and the Mapping protocol.
    """

    _REP_TYPE: type[RepT]

    def _additional_save(self, f: h5py.File):
        del f

    def save(self, file_path: str | pathlib.Path):
        """Save the data to an HDF5 file.

        The data are saved in the following structure:

        - The root level contains the attribute ``type`` with the class name.
        - Each channel is saved as a group.
        - The group contains two datasets:

          - ``grid`` for the grid.
          - ``entries`` for the entries.

        - For :class:`.TimedFSData`, there will be a dataset ``times`` at the root level containing the time grid.

        """
        with h5py.File(str(file_path), "a") as f:
            f.attrs["domain"] = self.domain
            f.attrs["kind"] = str(self.kind)
            f.attrs["channels"] = self.channel_names
            self._additional_save(f)
            grp = f.create_group("data")
            _save_grid(grp.create_group("grid"), self.grid)
            grp.create_dataset("entries", data=cast(Any, self.get_kernel()))

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
                chnname: cls._REP_TYPE(
                    grid=(f[chnname]["grid"][...],),  # type: ignore
                    entries=f[chnname]["entries"][...][None, None, None, None, ...],  # type: ignore
                )
                for chnname in f
                if isinstance(f[chnname], h5py.Group)
            }
        return cls.from_dict(dict_, **additions)

    @classmethod
    def load(cls, file_path: str | pathlib.Path, legacy: bool = False):
        """Load the data from an HDF5 file (*Deprecated*).

        Warning
        -------
        This method is deprecated and will be removed in 0.8.0; use :func:`~typed_lisa_toolkit.load_data` instead.
        """
        warnings.warn(
            "The 'load' method is deprecated and will be removed in 0.8.0; use the function `load_data` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _load_data(cls, file_path, legacy=legacy)


class _SeriesData[RepT: reps.UniformTimeSeries | reps.UniformFrequencySeries](  # pyright: ignore[reportUnsafeMultipleInheritance]
    Data[RepT], _1DSubsetMixin[RepT], abc.ABC
):
    def get_embedded(
        self,
        embedding_grid: "Grid1D[Axis]",
        *,
        known_slices: tuple[slice, ...] | None = None,
    ):
        """Return data embedded on a new 1D grid."""
        grid, entries = _mixins.embed_entries_to_grid(
            self.grid,
            self.entries,
            embedding_grid,
            known_slices=known_slices,
        )
        return type(self)(
            grid=grid, entries=entries, channels=self.channel_names, name=self.name
        )


class TSData(_SeriesData[reps.UniformTimeSeries]):
    """Multi-channel time series data container.

    .. note::
        To construct a :class:`.TSData`, use factory
        functions: :func:`~typed_lisa_toolkit.tsdata`
        or :func:`~typed_lisa_toolkit.construct_tsdata`.

    """

    _REP_TYPE: type[reps.UniformTimeSeries] = reps.UniformTimeSeries

    @classmethod
    def from_entries(
        cls,
        *,
        times: Axis,
        entries: Array,
        channels: tuple[str, ...],
        name: str | None = None,
    ) -> Self:
        """Construct from raw time-domain entries and explicit channel names."""
        return cls((times,), entries, channels=channels, name=name)

    @property
    def kind(self) -> None:
        """Semantic kind of the data."""
        return None

    @property
    def times(self):
        """Return the times."""
        return self.grid[0]

    @property
    def dt(self) -> float:
        """Return the time step."""
        return self.times.step

    @property
    def t_start(self) -> float:
        """Return the start time."""
        return self.times.start

    @property
    def t_end(self) -> float:
        """Return the end time."""
        return self.times.stop

    def get_frequencies(self):
        """Return the frequencies grid matching the time grid."""
        return self.xp.fft.rfftfreq(len(self.times), d=self.dt)

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

        .. warning::
            This method is deprecated and will be removed in 0.8.0; use
            :func:`~typed_lisa_toolkit.shop.time2freq` instead.

        Returns
        -------
        :class:`.FSData` | :class:`.TimedFSData`
            The frequency series data. If `keep_times` is `True`, the time grid is kept.

        """
        from ..shop import time2freq

        warnings.warn(
            "The 'to_fsdata' method is deprecated and will be removed in 0.8.0; use the function `shop.time2freq` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _window = tapering(self.xp.asarray(self.times)) if tapering is not None else 1
        return time2freq(self * _window, keep_time=keep_times)

    def get_zero_padded(
        self,
        pad_time: tuple[float, float],
        tapering: tapering.Tapering | None = None,
    ) -> Self:
        """Return the zero-padded data."""
        xp = xpc.get_namespace(self.get_kernel())
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
        signal = self.get_kernel() * tapering_window
        padded_signal = xp.pad(
            signal,
            ((0, 0), (0, 0), (0, 0), (0, 0), pad_width),
            mode="constant",
        )
        return type(self).from_entries(
            times=padded_time,
            entries=padded_signal,
            channels=self.channel_names,
            name=self.name,
        )

    def _get_plotter(self):
        from ..viz import plotters

        return plotters.TSDataPlotter


class FSData(_SeriesData[reps.UniformFrequencySeries]):
    """Multi-channel frequency series data container.

    .. note::
        To construct a :class:`.FSData`, use factory
        functions: :func:`~typed_lisa_toolkit.fsdata`
        or :func:`~typed_lisa_toolkit.construct_fsdata`.

    """

    _REP_TYPE: type[reps.UniformFrequencySeries] = reps.UniformFrequencySeries

    @classmethod
    def from_entries(
        cls,
        *,
        frequencies: Axis,
        entries: Array,
        channels: tuple[str, ...],
        name: str | None = None,
    ) -> Self:
        """Construct from raw frequency-domain entries and explicit channel names."""
        return cls((frequencies,), entries, channels=channels, name=name)

    @property
    def kind(self) -> None | str:
        """Semantic kind of the data."""
        return None

    @property
    def frequencies(self):
        """Return the frequencies."""
        return self.grid[0]

    @property
    def df(self):
        """Return the frequency step."""
        return self.frequencies.step

    @property
    def f_min(self):
        """Return the minimum frequency."""
        return self.frequencies.start

    @property
    def f_max(self):
        """Return the maximum frequency."""
        return self.frequencies.stop

    def set_times(self, times: "Axis") -> TimedFSData:
        """Return a :class:`.TimedFSData` with the time grid set."""
        return TimedFSData(
            self.grid, self.entries, channels=self.channel_names, name=self.name
        ).set_times(times)

    def to_tsdata(
        self,
        times: npt.NDArray[np.floating[Any]] | Linspace,
        *,
        tapering: tapering.Tapering | None = None,
    ):
        """Return the time series data (*Deprecated*).

        .. warning::
            This method is deprecated and will be removed in 0.8.0; use
            :func:`~typed_lisa_toolkit.shop.freq2time` instead.
        """
        from ..shop import freq2time

        warnings.warn(
            "The 'to_tsdata' method is deprecated and will be removed in 0.8.0; use the function `shop.freq2time` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        _window = tapering(self.xp.asarray(times)) if tapering is not None else 1
        return freq2time(self * _window, times=times)

    def _get_plotter(self):
        from ..viz import plotters

        return plotters.FSDataPlotter


class TimedFSData(FSData):
    """Multi-channel frequency series data with time information.

    .. note::
        To construct a :class:`.TimedFSData`, use factory
        functions: :func:`~typed_lisa_toolkit.timed_fsdata`
        or :func:`~typed_lisa_toolkit.construct_timed_fsdata`.
    """

    @property
    def kind(self):
        """Semantic kind of the data."""
        return "timed"

    def _additional_save(self, f: h5py.File):
        f.create_dataset("times", data=np.asarray(self.times))

    @classmethod
    def _additional_load(cls, f: h5py.File):
        times_data = cast(h5py.Dataset, f["times"])[()]
        return {"times": times_data}

    def set_times(self, times: Axis) -> Self:
        """Set the time grid.

        .. note::
            This method returns ``self`` to allow for fluent method chaining.
        """
        self._times: Linspace = Linspace.make(times)
        self._dt: float = self._times.step
        return self

    @property
    def times(self):
        """Associated time grid."""
        return self._times

    @property
    def dt(self) -> float:
        """Step size of the associated time grid."""
        return self._dt

    def drop_times(self):
        """Drop the time grid."""
        return FSData(
            self.grid,
            self.get_kernel(),
            channels=self.channel_names,
            name=self.name,
        )

    def to_tsdata(
        self,
        times: npt.NDArray[np.floating[Any]] | Linspace | None = None,
        *,
        tapering: tapering.Tapering | None = None,
    ):
        """Return the time series data with times grid (*Deprecated*).

        .. warning::
            This method is deprecated and will be removed in 0.8.0; use
            :func:`~typed_lisa_toolkit.shop.freq2time` instead.
        """
        del times
        return super().to_tsdata(times=self.times, tapering=tapering)


class _2DGridData[  # pyright: ignore[reportUnsafeMultipleInheritance]
    RepT: reps.STFT[Grid2D[Linspace, Linspace]] | reps.WDM[Grid2D[Linspace, Linspace]]
](
    Data[RepT],
    _2DSubsetMixin[RepT],
    abc.ABC,
):
    @classmethod
    def from_entries(
        cls,
        *,
        frequencies: Axis,
        times: Axis,
        entries: Array,
        channels: tuple[str, ...],
        sparse_indices: Array | None = None,
        name: str | None = None,
    ) -> Self:
        """Construct from raw time-frequency entries and explicit channel names."""
        grid2d = build_grid2d(frequencies, times, sparse_indices=sparse_indices)
        return cls(grid2d, entries, channels=channels, name=name)

    def _get_plotter(self):
        from ..viz import plotters

        return plotters.TFDataPlotter


class STFTData[GridT: Grid2D[Linspace, Linspace]](_2DGridData[reps.STFT[GridT]]):
    """Multi-channel short-time Fourier transform data container.

    .. note::
        To construct a :class:`.STFTData`, use factory
        functions: :func:`~typed_lisa_toolkit.stftdata`
        or :func:`~typed_lisa_toolkit.construct_stftdata`.
    """

    _REP_TYPE: type[reps.STFT[GridT]] = reps.STFT[GridT]

    @property
    def kind(self) -> str:
        """Semantic kind of the data."""
        return "stft"


class WDMData[GridT: Grid2D[Linspace, Linspace]](_2DGridData[reps.WDM[GridT]]):
    """Multi-channel wavelet domain model data container.

    .. note::
        To construct a :class:`.WDMData`, use factory
        functions: :func:`~typed_lisa_toolkit.wdmdata`
        or :func:`~typed_lisa_toolkit.construct_wdmdata`.
    """

    _REP_TYPE: type[reps.WDM[GridT]] = reps.WDM[GridT]

    @property
    def kind(self) -> str:
        """Semantic kind of the data."""
        return "wdm"


def _enforce_uniform(ary: Axis) -> Linspace:
    """Enforce that the given array is uniform and return it as a Linspace."""
    try:
        return Linspace.make(ary)
    except ValueError as e:
        raise ValueError(
            "To construct data objects, the grid axes must be uniform"
        ) from e


def tsdata(
    mapping: Mapping[str, reps.TimeSeries[Linspace]], name: str | None = None
) -> TSData:
    """Construct :class:`~types.data.TSData` from a mapping of channel names to :class:`~typed_lisa_toolkit.types.TimeSeries`."""
    _ = _mixins.validate_maps_to_reps(mapping)
    return TSData.from_dict(mapping, name=name)


def fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]], name: str | None = None
) -> FSData:
    """Construct :class:`~types.data.FSData` from a mapping of channel names to :class:`~typed_lisa_toolkit.types.FrequencySeries`."""
    _ = _mixins.validate_maps_to_reps(mapping)
    return FSData.from_dict(mapping, name=name)


def timed_fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]],
    times: Linspace | npt.NDArray[np.floating[Any]],
    name: str | None = None,
) -> TimedFSData:
    """Construct :class:`~types.data.TimedFSData` from a mapping of channel names to :class:`~typed_lisa_toolkit.types.FrequencySeries` and a time grid."""
    _ = _mixins.validate_maps_to_reps(mapping)
    return TimedFSData.from_dict(mapping, times=times, name=name)


def stftdata[GridT: Grid2D[Linspace, Linspace]](
    mapping: Mapping[str, reps.STFT[GridT]],
    name: str | None = None,
) -> STFTData[GridT]:
    """Construct :class:`~types.data.STFTData` from a mapping of channel names to :class:`~typed_lisa_toolkit.types.ShortTimeFourierTransform`."""
    _ = _mixins.validate_maps_to_reps(mapping)
    return STFTData[GridT].from_dict(mapping, name=name)


def wdmdata[GridT: Grid2D[Linspace, Linspace]](
    mapping: Mapping[str, reps.WDM[GridT]],
    name: str | None = None,
) -> WDMData[GridT]:
    """Construct :class:`~types.WDMData` from a mapping of channel names to :class:`~typed_lisa_toolkit.types.WilsonDaubechiesMeyer`."""
    _ = _mixins.validate_maps_to_reps(mapping)
    return WDMData[GridT].from_dict(mapping, name=name)


def construct_tsdata(
    *,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    name: str | None = None,
) -> TSData:
    """Construct a :class:`~types.TSData`.

    Parameters
    ----------
    times: :class:`~typed_lisa_toolkit.types.misc.Axis`
        A uniform array of shape ``(n_times,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the time grid of the data.

    entries: :class:`~typed_lisa_toolkit.types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nt)`` where ``Nt`` is the size of ``times``.
        See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels: tuple[str, ...]
        Names of the channels in the data.

    name: str | None
        Name of the data. Default is ``None``.

    Note
    ----
    See the general description of the shape convention for
    :external+l2d-interface:attr:`entries <l2d_interface.contract.Representation.entries>`.
    """
    times = _enforce_uniform(times)
    return TSData.from_entries(
        times=times, entries=entries, channels=channels, name=name
    )


def construct_fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    name: str | None = None,
) -> FSData:
    """Construct an :class:`~types.data.FSData`.

    Parameters
    ----------
    frequencies: :class:`~typed_lisa_toolkit.types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the frequency grid of the data.

    entries: :class:`~typed_lisa_toolkit.types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nf)`` where ``Nf`` is the size of ``frequencies``.
        See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels: tuple[str, ...]
        Names of the channels in the data.

    name: str | None
        Name of the data. Default is ``None``.

    Note
    ----
    See the general description of the shape convention for
    :external+l2d-interface:attr:`entries <l2d_interface.contract.Representation.entries>`.
    """
    frequencies = _enforce_uniform(frequencies)
    return FSData.from_entries(
        frequencies=frequencies,
        entries=entries,
        channels=channels,
        name=name,
    )


def construct_timed_fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    times: Linspace | npt.NDArray[np.floating[Any]],
    name: str | None = None,
) -> TimedFSData:
    """Construct a :class:`~types.data.TimedFSData`.

    Parameters
    ----------
    frequencies: :class:`~typed_lisa_toolkit.types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the frequency grid of the data.

    entries: :class:`~typed_lisa_toolkit.types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nf)`` where ``Nf`` is the size of ``frequencies``.
        See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels: tuple[str, ...]
        Names of the channels in the data.

    times: :class:`~typed_lisa_toolkit.types.misc.Axis`
        An array of shape ``(n_times,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the associated time grid of the data.

    name: str | None
        Name of the data. Default is ``None``.

    Note
    ----
    The associated time grid does not count as a grid dimension of the data,
    hence the shape of `entries` does include the time dimension.

    """
    frequencies = _enforce_uniform(frequencies)
    return TimedFSData.from_entries(
        frequencies=frequencies,
        entries=entries,
        channels=channels,
        name=name,
    ).set_times(times)


@overload
def construct_stftdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: None = None,
    name: str | None = None,
) -> STFTData[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def construct_stftdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: Array,
    name: str | None = None,
) -> STFTData[Grid2DSparse[Linspace, Linspace]]: ...


def construct_stftdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: Array | None = None,
    name: str | None = None,
):
    """Construct an :class:`~types.data.STFTData`.

    Parameters
    ----------
    frequencies: :class:`~typed_lisa_toolkit.types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the frequency grid of the data.

    times: :class:`~typed_lisa_toolkit.types.misc.Axis`
        A uniform array of shape ``(n_times,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the time grid of the data.

    entries: :class:`~typed_lisa_toolkit.types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, n_freqs, n_times)`` or
        ``(n_batch, n_channels, n_harmonics, n_features, n_sparse)``. See the
        :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels: tuple[str, ...]
        Names of the channels in the data.

    sparse_indices: :class:`~typed_lisa_toolkit.types.misc.Array` | None
        An array of shape ``(n_sparse, 2)`` containing the indices of the non-zero entries in the sparse grid.
        See :attr:`~typed_lisa_toolkit.types.misc.Grid2DSparse.indices` for more details.
        If ``None``, the data is treated as dense.

    name: str | None
        Name of the data. Default is ``None``.
    """
    frequencies = _enforce_uniform(frequencies)
    times = _enforce_uniform(times)
    if sparse_indices is None:
        return STFTData[Grid2DCartesian[Linspace, Linspace]].from_entries(
            frequencies=frequencies,
            times=times,
            entries=entries,
            channels=channels,
            sparse_indices=sparse_indices,
            name=name,
        )
    return STFTData[Grid2DSparse[Linspace, Linspace]].from_entries(
        frequencies=frequencies,
        times=times,
        entries=entries,
        channels=channels,
        sparse_indices=sparse_indices,
        name=name,
    )


@overload
def construct_wdmdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: None = None,
    name: str | None = None,
) -> WDMData[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def construct_wdmdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: Array,
    name: str | None = None,
) -> WDMData[Grid2DSparse[Linspace, Linspace]]: ...


def construct_wdmdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: Array | None = None,
    name: str | None = None,
):
    """Construct a :class:`~types.data.WDMData`.

    Parameters
    ----------
    frequencies: :class:`~typed_lisa_toolkit.types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the frequency grid of the data.

    times: :class:`~typed_lisa_toolkit.types.misc.Axis`
        A uniform array of shape ``(n_times,)`` or a :class:`~typed_lisa_toolkit.types.Linspace`
        object representing the time grid of the data.

    entries: :class:`~typed_lisa_toolkit.types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, n_freqs, n_times)`` or
        ``(n_batch, n_channels, n_harmonics, n_features, n_sparse)``. See the
        :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels: tuple[str, ...]
        Names of the channels in the data.

    sparse_indices: :class:`~typed_lisa_toolkit.types.misc.Array` | None
        An array of shape ``(n_sparse, 2)`` containing the indices of the non-zero entries in the sparse grid.
        See :attr:`~typed_lisa_toolkit.types.misc.Grid2DSparse.indices` for more details.
        If ``None``, the data is treated as dense.

    name: str | None
        Name of the data. Default is ``None``.

    Note
    ----
    See the general description of the shape convention for
    :external+l2d-interface:attr:`entries <l2d_interface.contract.Representation.entries>`.
    """
    frequencies = _enforce_uniform(frequencies)
    times = _enforce_uniform(times)
    if sparse_indices is None:
        return WDMData[Grid2DCartesian[Linspace, Linspace]].from_entries(
            frequencies=frequencies,
            times=times,
            entries=entries,
            channels=channels,
            sparse_indices=sparse_indices,
            name=name,
        )
    return WDMData[Grid2DSparse[Linspace, Linspace]].from_entries(
        frequencies=frequencies,
        times=times,
        entries=entries,
        channels=channels,
        name=name,
    )


@overload
def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Literal["time"],
    kind: str | None = None,
    sparse: bool = False,
    legacy: bool = False,
) -> TSData: ...


@overload
def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Literal["frequency"],
    kind: None = None,
    sparse: bool = False,
    legacy: bool = False,
) -> FSData: ...


@overload
def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Literal["frequency"],
    kind: Literal["timed"],
    sparse: bool = False,
    legacy: bool = False,
) -> TimedFSData: ...


@overload
def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Literal["time-frequency"],
    kind: Literal["stft"],
    sparse: Literal[False] = False,
    legacy: bool = False,
) -> STFTData[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Literal["time-frequency"],
    kind: Literal["stft"],
    sparse: Literal[True],
    legacy: bool = False,
) -> STFTData[Grid2DSparse[Linspace, Linspace]]: ...


@overload
def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Literal["time-frequency"],
    kind: Literal["wdm"],
    sparse: Literal[False] = False,
    legacy: bool = False,
) -> WDMData[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Literal["time-frequency"],
    kind: Literal["wdm"],
    sparse: Literal[True],
    legacy: bool = False,
) -> WDMData[Grid2DSparse[Linspace, Linspace]]: ...


def load_data(
    file_path: str | pathlib.Path,
    *,
    domain: Domain | None = None,
    kind: str | None = None,
    sparse: bool = False,
    legacy: bool = False,
):
    """Load the data from a saved HDF5 file."""
    if legacy:
        warnings.warn(
            "The `legacy` mode of `load_data` is deprecated and will be removed in 0.9.0; ",
            DeprecationWarning,
            stacklevel=2,
        )
        with h5py.File(str(file_path), "r") as f:
            data_type = str(f.attrs["type"])
        if data_type == "TSData":
            return _load_data(TSData, file_path, legacy=legacy)
        if data_type == "FSData":
            return _load_data(FSData, file_path, legacy=legacy)
        if data_type == "TimedFSData":
            return _load_data(TimedFSData, file_path, legacy=legacy)
        raise ValueError(f"Unsupported data type: {data_type}")
    with h5py.File(str(file_path), "r") as f:
        domain_attr = str(f.attrs["domain"])
        kind_attr = str(f.attrs["kind"])
        expected_kind = str(kind)
        if domain is None:
            warnings.warn(
                "The `domain` argument of `load_data` is not provided. From 0.8.0,"
                + " the `domain` argument will be required.",
                FutureWarning,
                stacklevel=2,
            )
        elif domain_attr != domain:
            raise ValueError(
                f"The domain of the data does not match the provided `domain` argument. "
                + f"The data has domain {domain_attr}, but `domain` is {domain}."
            )
        if kind_attr != expected_kind:
            raise ValueError(
                f"The kind of the data does not match the provided `kind` argument. "
                + f"The data has kind {kind_attr}, but `kind` is {kind}."
            )
        if domain_attr == "time" or domain_attr == "frequency":
            if sparse:
                raise ValueError(
                    f"Sparse grid is not supported for domain {domain_attr}."
                )
    if domain_attr == "time":
        return _load_data(TSData, file_path)
    if domain_attr == "frequency":
        if kind_attr == "timed":
            return _load_data(TimedFSData, file_path)
        return _load_data(FSData, file_path)
    if domain_attr == "time-frequency":
        if kind_attr == "stft":
            if sparse:
                return _load_data(STFTData[Grid2DSparse[Linspace, Linspace]], file_path)
            return _load_data(STFTData[Grid2DCartesian[Linspace, Linspace]], file_path)
        if kind_attr == "wdm":
            if sparse:
                return _load_data(WDMData[Grid2DSparse[Linspace, Linspace]], file_path)
            return _load_data(WDMData[Grid2DCartesian[Linspace, Linspace]], file_path)
    raise ValueError(
        f"Unsupported combination of domain and kind: domain={domain_attr}, kind={kind_attr}."
    )


def load_sangria(
    file_path: str | pathlib.Path,
    name: Literal[
        "obs/tdi", "sky/mbhb/tdi", "sky/igb/tdi", "sky/vgb/tdi", "sky/dgb/tdi"
    ] = "obs/tdi",
    channels: Literal["AE", "AET", "XYZ"] = "AE",
) -> TSData:
    """Load Sangria dataset or Sangria HM dataset."""
    from ..shop import aet2xyz, xyz2aet

    def transform_channels(data: TSData) -> TSData:
        channel_names = tuple(name for name in channels)
        if set(channel_names).issubset(data.channel_names):
            return data.pick(channel_names)
        if set(channel_names).issubset(("X", "Y", "Z")):
            return aet2xyz(data).pick(channel_names)
        if set(channel_names).issubset(("A", "E", "T")):
            return xyz2aet(data).pick(channel_names)
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


def load_ldc_data(file_path: str | pathlib.Path, **kwargs: Any) -> TSData:
    """Load the LDC dataset."""
    # Currently among LDC datasets only Sangria and Sangria HM are supported
    return load_sangria(file_path, **kwargs)


def load_mojito(processed_data: SignalProcessor):
    """Load the data from a preprocessed Mojito data object."""
    channel_names = tuple(processed_data.channels)
    _data = cast(dict[str, "Array"], processed_data.data)
    _mapping = {
        chnname: reps.time_series(
            processed_data.t, _data[chnname][None, None, None, None, :]
        )
        for chnname in channel_names
    }
    return tsdata(cast(Mapping[str, reps.UniformTimeSeries], _mapping))
