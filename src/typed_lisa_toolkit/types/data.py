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

    class _SubsettableRep1D(AnyReps, Protocol):
        def get_subset(
            self,
            *,
            interval: tuple[float, float] | None = None,
            slice: slice | None = None,
        ) -> Self: ...

    class _SubsettableRep2D(AnyReps, Protocol):
        def get_subset(
            self,
            *,
            time_interval: tuple[float, float] | None = None,
            freq_interval: tuple[float, float] | None = None,
            slices: tuple[slice, slice] | None = None,
            copy: bool = True,
        ) -> Self: ...


log = logging.getLogger(__name__)


def _save_axis(grp: h5py.Group, name: str, axis: Axis) -> None:
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


def _attr_bool(attrs: Any, key: str, *, default: bool = False) -> bool:
    return bool(attrs.get(key, default))


def _load_axis(node: h5py.Group | h5py.Dataset) -> Axis:
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
    return cast("Axis", cast("h5py.Dataset", node["values"])[()])


def _save_grid(grp: h5py.Group, grid: AnyGrid) -> None:
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
    if len(grid) == 2:  # noqa: PLR2004
        grp.attrs["sparse"] = False
        grp.attrs["dim"] = 2
        _save_axis(grp, "axis0", grid[0])
        _save_axis(grp, "axis1", grid[1])
        return


def _load_grid(node: h5py.Group | h5py.Dataset) -> AnyGrid:
    """Deserialize a representation grid from HDF5."""
    # Backward compatibility: previous format stored `grid` as a single dataset.
    if isinstance(node, h5py.Dataset):
        return cast("AnyGrid", node[()])

    sparse = _attr_bool(node.attrs, "sparse", default=False)
    dim = _attr_int(node.attrs, "dim") if "dim" in node.attrs else -1
    if not sparse and dim == 1:
        axis0 = _load_axis(cast("h5py.Group | h5py.Dataset", node["axis0"]))
        return (axis0,)
    if not sparse and dim == 2:  # noqa: PLR2004
        axis0 = _load_axis(cast("h5py.Group | h5py.Dataset", node["axis0"]))
        axis1 = _load_axis(cast("h5py.Group | h5py.Dataset", node["axis1"]))
        return (axis0, axis1)
    if sparse and dim == 2:  # noqa: PLR2004
        axis0 = _load_axis(cast("h5py.Group | h5py.Dataset", node["axis0"]))
        axis1 = _load_axis(cast("h5py.Group | h5py.Dataset", node["axis1"]))
        sparse_indices = cast("h5py.Dataset", node["sparse_indices"])[()]
        return Grid2DSparse(axis0, axis1, sparse_indices=sparse_indices)

    msg = f"Unknown grid serialization format with attributes: {node.attrs}"
    raise ValueError(msg)


def _load_data[DataT: "Data[AnyReps]"](
    cls: type[DataT],
    file_path: str | pathlib.Path,
    *,
    legacy: bool = False,
) -> DataT:
    """Load data from an HDF5 file."""
    if legacy:
        return cls._load_legacy(file_path)  # pyright: ignore[reportPrivateUsage]
    with h5py.File(str(file_path), "r") as f:
        channels_attr = cast("Iterable[object]", f.attrs["channels"])
        channels = tuple(str(ch) for ch in channels_attr)
        additions = cls._additional_load(f)  # pyright: ignore[reportPrivateUsage]
        data_group = cast("h5py.Group", f["data"])
        grid_node = cast("h5py.Group | h5py.Dataset", data_group["grid"])
        grid_data = _load_grid(grid_node)
        entries_data = cast("h5py.Dataset", data_group["entries"])[()]
        return cls(grid_data, entries_data, channels=channels, **additions)


class _SubsetMixin1D[RepT: _SubsettableRep1D](_mixins.ChannelMapping[RepT], abc.ABC):
    def get_subset(
        self,
        *,
        interval: tuple[float, float] | None = None,
        slice: slice | None = None,
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

        If `compare_to` is not `None`, the method draws both
        the data and the data in `compare_to`.
        """
        plotter = self._get_plotter()

        if compare_to is None:
            return plotter(self.get_subset(interval=interval)).draw(**kwargs)
        return plotter(self.get_subset(interval=interval)).compare(
            plotter(compare_to.get_subset(interval=interval)),
            **kwargs,
        )

    def _get_plotter(self) -> type[Any]:
        """Return the plotter class."""
        msg = "The method _get_plotter() must be implemented in the subclass."
        raise NotImplementedError(msg)


class _SubsetMixin2D[RepT: _SubsettableRep2D](_mixins.ChannelMapping[RepT], abc.ABC):
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

        If `compare_to` is not `None`, the method draws both
        the data and the data in `compare_to`.
        """
        plotter = self._get_plotter()

        if compare_to is None:
            return plotter(
                self.get_subset(
                    time_interval=time_interval,
                    freq_interval=freq_interval,
                ),
            ).draw(**kwargs)
        return plotter(
            self.get_subset(time_interval=time_interval, freq_interval=freq_interval),
        ).compare(
            plotter(
                compare_to.get_subset(
                    time_interval=time_interval,
                    freq_interval=freq_interval,
                ),
            ),
            **kwargs,
        )

    def _get_plotter(self) -> type[Any]:
        """Return the plotter class."""
        msg = "The method _get_plotter() must be implemented in the subclass."
        raise NotImplementedError(msg)


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

        - For :class:`.TimedFSData`, there will be a dataset ``times`` at
          the root level containing the time grid.

        """
        with h5py.File(str(file_path), "a") as f:
            f.attrs["domain"] = self.domain
            f.attrs["kind"] = str(self.kind)
            f.attrs["channels"] = self.channel_names
            self._additional_save(f)
            grp = f.create_group("data")
            _save_grid(grp.create_group("grid"), self.grid)
            grp.create_dataset("entries", data=cast("Any", self.get_kernel()))

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
                    grid=(f[chnname]["grid"][...],),  # pyright: ignore[reportArgumentType, reportIndexIssue]
                    entries=f[chnname]["entries"][...][None, None, None, None, ...],  # pyright: ignore[reportArgumentType, reportIndexIssue]
                )
                for chnname in f
                if isinstance(f[chnname], h5py.Group)
            }
        return cls.from_dict(dict_, **additions)

    @classmethod
    def load(cls, file_path: str | pathlib.Path, *, legacy: bool = False):
        """Load the data from an HDF5 file (*Deprecated*).

        Warning
        -------
        This method is deprecated and will be removed in 0.8.0;
        use :func:`~typed_lisa_toolkit.load_data` instead.
        """
        msg = (
            "The 'load' method is deprecated and will be removed in 0.8.0; "
            "use the function `load_data` instead."
        )
        warnings.warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        return _load_data(cls, file_path, legacy=legacy)


class _SeriesData[RepT: reps.UniformTimeSeries | reps.UniformFrequencySeries](  # pyright: ignore[reportUnsafeMultipleInheritance]
    Data[RepT],
    _SubsetMixin1D[RepT],
    abc.ABC,
):
    def get_embedded(
        self,
        embedding_grid: Grid1D[Axis],
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
            grid=grid,
            entries=entries,
            channels=self.channel_names,
            name=self.name,
        )


class TSData(_SeriesData[reps.UniformTimeSeries]):
    """Multi-channel time series data container.

    .. note::
        To construct a :class:`.TSData`,
        use the factory function :func:`~typed_lisa_toolkit.tsdata`.

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
        """Construct from raw time-domain entries and explicit channel names.

        Warning
        -------
        This method is considered an expert-level API; for most users,
        prefer to construct a :class:`.TSData` with the factory function
        :func:`~typed_lisa_toolkit.tsdata`.
        """
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

        msg = (
            "The 'to_fsdata' method is deprecated and will be removed in 0.8.0; "
            "use the function `shop.time2freq` instead."
        )
        warnings.warn(
            msg,
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
        To construct a :class:`.FSData`, use the factory
        function :func:`~typed_lisa_toolkit.fsdata`.

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
        """Construct from raw frequency-domain entries and explicit channel names.

        Warning
        -------
        This method is considered an expert-level API; for most users,
        prefer to construct a :class:`.FSData` with the factory function
        :func:`~typed_lisa_toolkit.fsdata`.
        """
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

    def set_times(self, times: Axis) -> TimedFSData:
        """Return a :class:`.TimedFSData` with the time grid set."""
        return TimedFSData(
            self.grid,
            self.entries,
            channels=self.channel_names,
            name=self.name,
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

        msg = (
            "The 'to_tsdata' method is deprecated and will be removed in 0.8.0; "
            "use the function `shop.freq2time` instead."
        )

        warnings.warn(
            msg,
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
        To construct a :class:`.TimedFSData`, use the factory
        function :func:`~typed_lisa_toolkit.fsdata`.
    """

    @property
    def kind(self):
        """Semantic kind of the data."""
        return "timed"

    def _additional_save(self, f: h5py.File):
        f.create_dataset("times", data=np.asarray(self.times))

    @classmethod
    def _additional_load(cls, f: h5py.File):
        times_data = cast("h5py.Dataset", f["times"])[()]
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


class _Grid2DData[  # pyright: ignore[reportUnsafeMultipleInheritance]
    RepT: reps.STFT[Grid2D[Linspace, Linspace]] | reps.WDM[Grid2D[Linspace, Linspace]],
](
    Data[RepT],
    _SubsetMixin2D[RepT],
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


class STFTData[GridT: Grid2D[Linspace, Linspace]](_Grid2DData[reps.STFT[GridT]]):
    """Multi-channel short-time Fourier transform data container.

    .. note::
        To construct a :class:`.STFTData`, use the factory
        function :func:`~typed_lisa_toolkit.stftdata`.
    """

    _REP_TYPE: type[reps.STFT[GridT]] = reps.STFT[GridT]

    @property
    def kind(self) -> str:
        """Semantic kind of the data."""
        return "stft"


class WDMData[GridT: Grid2D[Linspace, Linspace]](_Grid2DData[reps.WDM[GridT]]):
    """Multi-channel wavelet domain model data container.

    .. note::
        To construct a :class:`.WDMData`, use the factory
        function :func:`~typed_lisa_toolkit.wdmdata`.
    """

    _REP_TYPE: type[reps.WDM[GridT]] = reps.WDM[GridT]

    @property
    def kind(self) -> str:
        """Semantic kind of the data."""
        return "wdm"


def _enforce_uniform(ary: Axis, /) -> Linspace:
    """Enforce that the given array is uniform and return it as a Linspace."""
    try:
        return Linspace.make(ary)
    except ValueError as e:
        msg = "To construct data objects, the grid axes must be uniform"
        raise ValueError(msg) from e


def _validate_rep_shape(mapping: Mapping[Any, AnyReps], /):
    _msg = (
        "For data objects, the representation entries must have shape "
        "(n_batches, 1, 1, n_features, ...)."
    )
    for rep in mapping.values():
        if rep.entries.shape[1:3] != (1, 1):
            raise ValueError(_msg)


_func_deprecation_msg = (
    "The factory function `{deprecated_func_name}` is deprecated "
    "and will be removed in 0.8.0;"
    " use the factory function {func_name} instead."
)


@overload
def tsdata(
    mapping: Mapping[str, reps.TimeSeries[Linspace]],
    /,
    *,
    name: str | None = None,
) -> TSData: ...


@overload
def tsdata(
    *,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    name: str | None = None,
) -> TSData: ...


def tsdata(
    mapping: Mapping[str, reps.TimeSeries[Linspace]] | None = None,
    /,
    *,
    times: Axis | None = None,
    entries: Array | None = None,
    channels: tuple[str, ...] | None = None,
    name: str | None = None,
) -> TSData:
    """Construct a :class:`~types.TSData`.

    This function provides **two mutually exclusive** construction ways:

    **First**, from a positional-only `mapping` argument:

    Parameters
    ----------
    mapping:
        A mapping from channel names to :class:`~types.TimeSeries`
        with :class:`~types.Linspace` axes.

    name: str, optional
        Name of the data.


    **Second**, from several keyword arguments:

    Parameters
    ----------
    times: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
        object representing the time grid of the data.

    entries: :class:`~types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nt)``
        where ``Nt`` is the size of ``times``.
        See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels:
        A tuple of channel names in the data.

    name: str, optional
        Name of the data.
    """  # noqa: E501
    if mapping is not None:
        _msg_from_mapping = (
            "Cannot specify `times`, `entries`, or `channels` "
            "when using `mapping` to construct TSData."
        )
        if not all(arg is None for arg in (times, entries, channels)):
            raise ValueError(_msg_from_mapping)
        _ = _mixins.validate_maps_to_reps(mapping)
        _ = _validate_rep_shape(mapping)
        return TSData.from_dict(mapping, name=name)
    _msg = (
        "Must specify `times`, `entries`, and `channels`"
        "when not using `mapping` to construct TSData."
    )
    if not (times is not None and entries is not None and channels is not None):
        raise ValueError(_msg)
    times = _enforce_uniform(times)
    return TSData.from_entries(
        times=times,
        entries=entries,
        channels=channels,
        name=name,
    )


@overload
def fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]],
    /,
    *,
    times: Axis,
    name: str | None = None,
) -> TimedFSData: ...


@overload
def fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]],
    /,
    *,
    name: str | None = None,
) -> FSData: ...


@overload
def fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    times: None = None,
    name: str | None = None,
) -> FSData: ...


@overload
def fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    times: Axis,
    name: str | None = None,
) -> TimedFSData: ...


def fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]] | None = None,
    /,
    *,
    frequencies: Axis | None = None,
    entries: Array | None = None,
    channels: tuple[str, ...] | None = None,
    times: Axis | None = None,
    name: str | None = None,
):
    """Construct :class:`~types.FSData` or :class:`.TimedFSData`.

    This function provides **two mutually exclusive** construction ways:

    **First**, from a positional-only `mapping` argument:

    Parameters
    ----------
        mapping:
            A mapping from channel names to :class:`~types.FrequencySeries`
            with :class:`~types.Linspace` axes.

        times: :class:`~types.misc.Axis`, optional
            A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
            object representing the associated time grid of the data.
            Returns a :class:`~types.TimedFSData` if provided, otherwise returns
            a :class:`~types.FSData`.

        name: str, optional
            Name of the data.


    **Second**, from several keyword arguments:

    Parameters
    ----------
        frequencies: :class:`~types.misc.Axis`
            A uniform array of shape ``(n_freqs,)`` or a :class:`~types.Linspace`
            object representing the frequency grid of the data.

        entries: :class:`~types.misc.Array`
            A array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nf)``
            where ``Nf`` is the size of ``frequencies``.
            See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
            for more details.

        channels:
            A tuple of channel names in the data.

        times: :class:`~types.misc.Axis`, optional
            A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
            object representing the associated time grid of the data.
            Returns a :class:`~types.TimedFSData` if provided, otherwise returns
            a :class:`~types.FSData`.

        name: str, optional
            Name of the data.

    Note
    ----
    The associated time grid does not count as a grid dimension of the data,
    hence the shape of `entries` does include the time dimension.
    """  # noqa: E501
    if mapping is not None:
        _msg_from_mapping = (
            "Cannot specify `frequencies`, `entries`, or `channels` "
            "when using `mapping` to construct FSData."
        )
        if not all(arg is None for arg in (frequencies, entries, channels)):
            raise ValueError(_msg_from_mapping)
        _ = _mixins.validate_maps_to_reps(mapping)
        _ = _validate_rep_shape(mapping)
        _fsdata = FSData.from_dict(mapping, name=name)
    else:
        _msg = (
            "Must specify `frequencies`, `entries`, and `channels` when "
            "not using `mapping` to construct FSData."
        )
        if not (
            frequencies is not None and entries is not None and channels is not None
        ):
            raise ValueError(_msg)
        frequencies = _enforce_uniform(frequencies)
        _fsdata = FSData.from_entries(
            frequencies=frequencies,
            entries=entries,
            channels=channels,
            name=name,
        )
    if times is not None:
        times = _enforce_uniform(times)
        return _fsdata.set_times(times)
    return _fsdata


def timed_fsdata(
    mapping: Mapping[str, reps.FrequencySeries[Linspace]],
    times: Linspace | npt.NDArray[np.floating[Any]],
    name: str | None = None,
) -> TimedFSData:
    """Construct :class:`~types.TimedFSData` (*Deprecated*).

    Warning
    -------
    This function is deprecated and will be removed in 0.8.0;
    use the :func:`.fsdata` function with the `times` argument instead.
    """
    _warn_msg = _func_deprecation_msg.format(
        deprecated_func_name="timed_fsdata",
        func_name="`fsdata` with `times` argument",
    )
    warnings.warn(
        _warn_msg,
        DeprecationWarning,
        stacklevel=2,
    )
    return fsdata(mapping, times=times, name=name)


@overload
def stftdata[GridT: Grid2D[Linspace, Linspace]](
    mapping: Mapping[str, reps.STFT[GridT]],
    /,
    *,
    name: str | None = None,
) -> STFTData[GridT]: ...


@overload
def stftdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: None = None,
    name: str | None = None,
) -> STFTData[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def stftdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: Array,
    name: str | None = None,
) -> STFTData[Grid2DSparse[Linspace, Linspace]]: ...


def stftdata[GridT: Grid2D[Linspace, Linspace]](
    mapping: Mapping[str, reps.STFT[GridT]] | None = None,
    /,
    *,
    frequencies: Axis | None = None,
    times: Axis | None = None,
    entries: Array | None = None,
    channels: tuple[str, ...] | None = None,
    sparse_indices: Array | None = None,
    name: str | None = None,
):
    """Construct :class:`~types.STFTData`.

    This function provides **two mutually exclusive** construction ways:

    **First**, from a positional-only `mapping` argument:

    Parameters
    ----------
        mapping:
            A mapping from channel names to :class:`~types.STFT`
            with :class:`~types.misc.Grid2D`
            or :class:`~types.Linspace` axes.

        name: str, optional
            Name of the data.


    **Second**, from several keyword arguments:

    Parameters
    ----------
        frequencies: :class:`~types.misc.Axis`
            A uniform array of shape ``(n_freqs,)`` or a :class:`~types.Linspace`
            object representing the frequency grid of the data.

        times: :class:`~types.misc.Axis`
            A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
            object representing the time grid of the data.

        entries: :class:`~types.misc.Array`
            An array of shape ``(n_batch, n_channels, n_harmonics, n_features, n_freqs, n_times)`` or
            ``(n_batch, n_channels, n_harmonics, n_features, n_sparse)``. See the
            :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
            for more details.

        channels:
            A tuple of channel names in the data.

        sparse_indices: :class:`~types.misc.Array`, optional
            An array of shape ``(n_sparse, 2)`` containing the indices of the
            non-zero entries in the sparse grid.
            See :attr:`~types.misc.Grid2DSparse.indices` for more details.
            If not provided, the data is treated as dense.

        name: str, optional
            Name of the data.
    """  # noqa: E501
    if mapping is not None:
        _msg_from_mapping = (
            "Cannot specify `frequencies`, `times`, `entries`, `channels`, "
            "or `sparse_indices` when using `mapping` to construct STFTData."
        )
        if not all(
            arg is None
            for arg in (frequencies, times, entries, channels, sparse_indices)
        ):
            raise ValueError(_msg_from_mapping)
        _ = _mixins.validate_maps_to_reps(mapping)
        _ = _validate_rep_shape(mapping)
        return STFTData[GridT].from_dict(mapping, name=name)
    _msg = (
        "Must specify `frequencies`, `times`, `entries`, and `channels` "
        "when not using `mapping` to construct STFTData."
    )
    if not (
        frequencies is not None
        and times is not None
        and entries is not None
        and channels is not None
    ):
        raise ValueError(_msg)
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
def wdmdata[GridT: Grid2D[Linspace, Linspace]](
    mapping: Mapping[str, reps.WDM[GridT]],
    /,
    *,
    name: str | None = None,
) -> WDMData[GridT]: ...


@overload
def wdmdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: None = None,
    name: str | None = None,
) -> WDMData[Grid2DCartesian[Linspace, Linspace]]: ...


@overload
def wdmdata(
    *,
    frequencies: Axis,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    sparse_indices: Array,
    name: str | None = None,
) -> WDMData[Grid2DSparse[Linspace, Linspace]]: ...


def wdmdata[GridT: Grid2D[Linspace, Linspace]](
    mapping: Mapping[str, reps.WDM[GridT]] | None = None,
    /,
    *,
    frequencies: Axis | None = None,
    times: Axis | None = None,
    entries: Array | None = None,
    channels: tuple[str, ...] | None = None,
    sparse_indices: Array | None = None,
    name: str | None = None,
):
    """Construct :class:`~types.WDMData`.

    This function provides **two mutually exclusive** construction ways:

    **First**, from a positional-only `mapping` argument:

    Parameters
    ----------
        mapping:
            A mapping from channel names to :class:`~types.WDM`
            with :class:`~types.misc.Grid2D`
            or :class:`~types.Linspace` axes.

        name: str, optional
            Name of the data.


    **Second**, from several keyword arguments:

    Parameters
    ----------
        frequencies: :class:`~types.misc.Axis`
            A uniform array of shape ``(n_freqs,)`` or a :class:`~types.Linspace`
            object representing the frequency grid of the data.

        times: :class:`~types.misc.Axis`
            A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
            object representing the time grid of the data.

        entries: :class:`~types.misc.Array`
            An array of shape ``(n_batch, n_channels, n_harmonics, n_features, n_freqs, n_times)`` or
            ``(n_batch, n_channels, n_harmonics, n_features, n_sparse)``. See the
            :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
            for more details.

        channels:
            A tuple of channel names in the data.

        sparse_indices: :class:`~types.misc.Array`, optional
            An array of shape ``(n_sparse, 2)`` containing the indices of
            the non-zero entries in the sparse grid.
            See :attr:`~types.misc.Grid2DSparse.indices` for more details.
            If not provided, the data is treated as dense.

        name: str, optional
            Name of the data.
    """  # noqa: E501
    if mapping is not None:
        _msg_from_mapping = (
            "Cannot specify `frequencies`, `times`, `entries`, `channels`, "
            "or `sparse_indices` when using `mapping` to construct WDMData."
        )
        if not all(
            arg is None
            for arg in (frequencies, times, entries, channels, sparse_indices)
        ):
            raise ValueError(_msg_from_mapping)
        _ = _mixins.validate_maps_to_reps(mapping)
        _ = _validate_rep_shape(mapping)
        return WDMData[GridT].from_dict(mapping, name=name)
    _msg = (
        "Must specify `frequencies`, `times`, `entries`, and `channels` "
        "when not using `mapping` to construct WDMData."
    )
    if not (
        frequencies is not None
        and times is not None
        and entries is not None
        and channels is not None
    ):
        raise ValueError(_msg)
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
        sparse_indices=sparse_indices,
        name=name,
    )


def construct_tsdata(
    *,
    times: Axis,
    entries: Array,
    channels: tuple[str, ...],
    name: str | None = None,
) -> TSData:
    """Construct a :class:`~types.TSData` (*Deprecated*).

    Parameters
    ----------
    times: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
        object representing the time grid of the data.

    entries: :class:`~types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nt)``
        where ``Nt`` is the size of ``times``.
        See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels:
        Names of the channels in the data.

    name: str, optional
        Name of the data.


    Warning
    -------
    This function is deprecated and will be removed in 0.8.0; use the :func:`.tsdata`
    function with the keyword arguments instead.
    """  # noqa: E501
    _msg = _func_deprecation_msg.format(
        deprecated_func_name="construct_tsdata",
        func_name="`tsdata` with keyword arguments",
    )
    warnings.warn(
        _msg,
        DeprecationWarning,
        stacklevel=2,
    )
    return tsdata(times=times, entries=entries, channels=channels, name=name)


@overload
def construct_fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    times: Axis,
    name: str | None = None,
) -> TimedFSData: ...


@overload
def construct_fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    name: str | None = None,
) -> FSData: ...


def construct_fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    name: str | None = None,
    times: Axis | None = None,
) -> FSData:
    """Construct an :class:`~types.data.FSData` (*Deprecated*).

    Parameters
    ----------
    frequencies: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~types.Linspace`
        object representing the frequency grid of the data.

    entries: :class:`~types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nf)``
        where ``Nf`` is the size of ``frequencies``.
        See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels:
        Names of the channels in the data.

    name: str, optional
        Name of the data.

    Warning
    -------
    This function is deprecated and will be removed in 0.8.0; use the
    :func:`.fsdata` function with the keyword arguments instead.
    """  # noqa: E501
    _msg = _func_deprecation_msg.format(
        deprecated_func_name="construct_fsdata",
        func_name="`fsdata` with keyword arguments",
    )
    warnings.warn(
        _msg,
        DeprecationWarning,
        stacklevel=2,
    )
    return fsdata(
        frequencies=frequencies,
        entries=entries,
        channels=channels,
        name=name,
        times=times,
    )


def construct_timed_fsdata(
    *,
    frequencies: Axis,
    entries: Array,
    channels: tuple[str, ...],
    times: Linspace | npt.NDArray[np.floating[Any]],
    name: str | None = None,
) -> TimedFSData:
    """Construct a :class:`~types.data.TimedFSData` (*Deprecated*).

    Parameters
    ----------
    frequencies: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~types.Linspace`
        object representing the frequency grid of the data.

    entries: :class:`~types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, Nf)``
        where ``Nf`` is the size of ``frequencies``.
        See the :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels: tuple[str, ...]
        Names of the channels in the data.

    times: :class:`~types.misc.Axis`
        An array of shape ``(n_times,)`` or a :class:`~types.Linspace`
        object representing the associated time grid of the data.

    name: str | None
        Name of the data. Default is ``None``.

    Note
    ----
    The associated time grid does not count as a grid dimension of the data,
    hence the shape of `entries` does include the time dimension.

    Warning
    -------
    This function is deprecated and will be removed in 0.8.0; use the
    :func:`.fsdata` function with the `times` argument instead.

    """  # noqa: E501
    _msg = _func_deprecation_msg.format(
        deprecated_func_name="construct_timed_fsdata",
        func_name="`fsdata` with `times` argument",
    )
    warnings.warn(
        _msg,
        DeprecationWarning,
        stacklevel=2,
    )
    return fsdata(
        frequencies=frequencies,
        entries=entries,
        channels=channels,
        times=times,
        name=name,
    )


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
    """Construct an :class:`~types.data.STFTData` (*Deprecated*).

    Parameters
    ----------
    frequencies: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~types.Linspace`
        object representing the frequency grid of the data.

    times: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
        object representing the time grid of the data.

    entries: :class:`~types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, n_freqs, n_times)`` or
        ``(n_batch, n_channels, n_harmonics, n_features, n_sparse)``. See the
        :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels: tuple[str, ...]
        Names of the channels in the data.

    sparse_indices: :class:`~types.misc.Array`, optional
        An array of shape ``(n_sparse, 2)`` containing the indices of
        the non-zero entries in the sparse grid.
        See :attr:`~types.misc.Grid2DSparse.indices` for more details.
        If not provided, the data is treated as dense.

    name: str, optional
        Name of the data.

    Warning
    -------
    This function is deprecated and will be removed in 0.8.0; use the
    :func:`.stftdata` function with the keyword arguments instead.
    """  # noqa: E501
    _msg = _func_deprecation_msg.format(
        deprecated_func_name="construct_stftdata",
        func_name="`stftdata` with keyword arguments",
    )
    warnings.warn(
        _msg,
        DeprecationWarning,
        stacklevel=2,
    )
    return stftdata(
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
    """Construct a :class:`~types.data.WDMData` (*Deprecated*).

    Parameters
    ----------
    frequencies: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_freqs,)`` or a :class:`~types.Linspace`
        object representing the frequency grid of the data.

    times: :class:`~types.misc.Axis`
        A uniform array of shape ``(n_times,)`` or a :class:`~types.Linspace`
        object representing the time grid of the data.

    entries: :class:`~types.misc.Array`
        An array of shape ``(n_batch, n_channels, n_harmonics, n_features, n_freqs, n_times)`` or
        ``(n_batch, n_channels, n_harmonics, n_features, n_sparse)``. See the
        :external+l2d-interface:attr:`convention <l2d_interface.contract.Representation.entries>`
        for more details.

    channels:
        Names of the channels in the data.

    sparse_indices: :class:`~types.misc.Array`, optional
        An array of shape ``(n_sparse, 2)`` containing the indices of
        the non-zero entries in the sparse grid.
        See :attr:`~types.misc.Grid2DSparse.indices` for more details.
        If not provided, the data is treated as dense.

    name: str, optional
        Name of the data. Default is ``None``.

    Warning
    -------
    This function is deprecated and will be removed in 0.8.0; use the
    :func:`.wdmdata` function with the keyword arguments instead.
    """  # noqa: E501
    _msg = _func_deprecation_msg.format(
        deprecated_func_name="construct_wdmdata",
        func_name="`wdmdata` with keyword arguments",
    )
    warnings.warn(
        _msg,
        DeprecationWarning,
        stacklevel=2,
    )
    return wdmdata(
        frequencies=frequencies,
        times=times,
        entries=entries,
        channels=channels,
        sparse_indices=sparse_indices,
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
        msg = (
            "The `legacy` mode of `load_data` is deprecated "
            "and will be removed in 0.9.0;"
        )
        warnings.warn(
            msg,
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
        _msg = (
            f"Unsupported data type in legacy mode: {data_type}. "
            "Only TSData, FSData, and TimedFSData are supported in legacy mode."
        )
        raise ValueError(_msg)
    with h5py.File(str(file_path), "r") as f:
        domain_attr = str(f.attrs["domain"])
        kind_attr = str(f.attrs["kind"])
        expected_kind = str(kind)
        if domain is None:
            _msg = (
                "The `domain` argument of `load_data` is not provided. "
                "From 0.8.0, this argument will be required. "
                "Currently, it is inferred from the data, "
                "but this behavior is deprecated."
            )
            warnings.warn(
                _msg,
                FutureWarning,
                stacklevel=2,
            )
        elif domain_attr != domain:
            _msg = (
                "The domain of the data does not match the provided `domain` argument. "
            )
            f"The data has domain {domain_attr}, but `domain` is {domain}."
            raise ValueError(_msg)
        if kind_attr != expected_kind:
            _msg = (
                "The kind of the data does not match the provided `kind` argument. "
                f"The data has kind {kind_attr}, but `kind` is {kind}."
            )
            raise ValueError(_msg)
        if domain_attr in {"time", "frequency"} and sparse:
            _msg = f"Sparse grid is not supported for domain {domain_attr}."
            raise ValueError(_msg)

    classes = {
        ("time", "None", False): TSData,
        ("frequency", "None", False): FSData,
        ("frequency", "timed", False): TimedFSData,
        ("time-frequency", "stft", False): STFTData[
            Grid2DCartesian[Linspace, Linspace]
        ],
        ("time-frequency", "stft", True): STFTData[Grid2DSparse[Linspace, Linspace]],
        ("time-frequency", "wdm", False): WDMData[Grid2DCartesian[Linspace, Linspace]],
        ("time-frequency", "wdm", True): WDMData[Grid2DSparse[Linspace, Linspace]],
    }
    cls = classes.get((domain_attr, kind_attr, sparse))
    if cls is None:
        _msg = (
            "Unsupported combination of domain, kind, "
            f"and sparse: domain={domain_attr}, "
            f"kind={kind_attr}, sparse={sparse}. "
            "Supported combinations are: "
            f"{', '.join(str(key) for key in classes)}."
        )
        raise ValueError(_msg)
    return _load_data(cls, file_path)


def load_sangria(
    file_path: str | pathlib.Path,
    name: Literal[
        "obs/tdi",
        "sky/mbhb/tdi",
        "sky/igb/tdi",
        "sky/vgb/tdi",
        "sky/dgb/tdi",
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
        msg = (
            "Cannot transform the channels to the requested ones. "
            f"The data has channels {data.channel_names}, "
            f"but the requested channels are {channel_names}."
        )
        raise ValueError(msg)

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
                },
            )
            return transform_channels(tsdata)
    msg = "The dataset does not have the expected structure."
    raise ValueError(msg)


def load_ldc_data(file_path: str | pathlib.Path, **kwargs: Any) -> TSData:
    """Load the LDC dataset."""
    # Currently among LDC datasets only Sangria and Sangria HM are supported
    return load_sangria(file_path, **kwargs)


def load_mojito(processed_data: SignalProcessor):
    """Load the data from a preprocessed Mojito data object."""
    channel_names = tuple(processed_data.channels)
    _data = cast("dict[str, Array]", processed_data.data)
    _mapping = {
        chnname: reps.time_series(
            processed_data.t,
            _data[chnname][None, None, None, None, :],
        )
        for chnname in channel_names
    }
    return tsdata(cast("Mapping[str, reps.UniformTimeSeries]", _mapping))
