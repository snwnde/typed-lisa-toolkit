"""Module for plotting tools."""

# pyright: reportUnknownMemberType=false
from __future__ import annotations

import abc
import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, Unpack, cast

from jax.random import f
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.typing
import numpy as np
import numpy.typing as npt
from typing_extensions import TypeIs

from ..types import (
    AnyAxis,
    AnyGrid,
    Array,
    Axis,
    Grid2D,
    Linspace,
    _mixins,
    misc,
    waveforms,
    representations,
)

if TYPE_CHECKING:
    AnyReps = representations.Representation[AnyGrid]

    class _IsPlottable[RepT: AnyReps](AnyReps, Protocol):
        @property
        def plot(self) -> _RepPlotter[RepT]: ...


class _FigureKwargs(TypedDict, total=False):
    dpi: float | None
    edgecolor: matplotlib.typing.ColorType | None
    facecolor: matplotlib.typing.ColorType | None
    figsize: npt.ArrayLike | None
    frameon: bool
    num: int | str | matplotlib.figure.Figure | matplotlib.figure.SubFigure | None
    tight_layout: object


class _AddSubplotKwargs(TypedDict, total=False):
    polar: object
    projection: object
    sharex: object
    sharey: object


class _LegendKwargs(TypedDict, total=False):
    bbox_to_anchor: object
    borderaxespad: object
    borderpad: object
    columnspacing: object
    edgecolor: matplotlib.typing.ColorType | None
    facecolor: matplotlib.typing.ColorType | None
    fancybox: object
    fontsize: object
    framealpha: object
    frameon: bool
    handlelength: object
    handler_map: object
    handletextpad: object
    labelspacing: object
    loc: object
    markerfirst: object
    markerscale: object
    mode: object
    ncol: object
    numpoints: object
    prop: object
    reverse: object
    scatterpoints: object
    scatteryoffsets: object
    shadow: object
    title: object
    title_fontsize: object


class _LineKwargs(TypedDict, total=False):
    alpha: object
    c: object
    color: object
    label: str | None
    linestyle: object
    linewidth: object
    ls: object
    lw: object
    marker: object
    markersize: object
    markevery: object
    ms: object
    zorder: object
    rasterized: object


class _ImshowKwargs(TypedDict, total=False):
    cmap: object
    norm: object
    aspect: object
    interpolation: object
    alpha: object
    vmin: object
    vmax: object
    origin: object
    extent: object
    filternorm: object
    filterrad: object
    resample: object
    url: object


class _PlotKwargs(
    _FigureKwargs,
    _AddSubplotKwargs,
    _LegendKwargs,
    _LineKwargs,
    _ImshowKwargs,
    total=False,
):
    transparent: object
    format: object
    metadata: object
    bbox_inches: object
    pad_inches: object
    backend: object
    orientation: object
    papertype: object
    bbox_extra_artists: object
    pil_kwargs: object


class _Keys[
    T: (
        _FigureKwargs,
        _AddSubplotKwargs,
        _LegendKwargs,
        _LineKwargs,
        _ImshowKwargs,
    )
]:
    def __init__(self, kwargs_cls: type[T], /) -> None:
        self.keys: frozenset[str] = kwargs_cls.__optional_keys__


FIGURE_KWARGS = _Keys(_FigureKwargs)
ADD_SUBPLOT_KWARGS = _Keys(_AddSubplotKwargs)
LEGEND_KWARGS = _Keys(_LegendKwargs)
LINE_KWARGS = _Keys(_LineKwargs)
IMSHOW_KWARGS = _Keys(_ImshowKwargs)

LINE_PLOT_MODES = Literal["loglog", "semilogx", "plot", "semilogy"]


def sieve_kwargs[
    T: (
        _FigureKwargs,
        _AddSubplotKwargs,
        _LegendKwargs,
        _LineKwargs,
        _ImshowKwargs,
    )
](
    to_accept: _Keys[T],
    /,
    kwargs: Mapping[str, Any],
) -> T:
    """Filter keyword arguments to the accepted subset."""
    _dict = {key: value for key, value in kwargs.items() if key in to_accept.keys}
    return cast("T", cast("object", _dict))


def _get_fig_ax(
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs: Unpack[_PlotKwargs],
):
    if fig is None:
        if ax is not None:
            msg = "If `ax` is provided, `fig` must also be provided."
            raise ValueError(msg)
        fig_kwargs = sieve_kwargs(FIGURE_KWARGS, kwargs)
        fig = plt.figure(**fig_kwargs)
    if ax is None:
        subplot_kwargs = sieve_kwargs(ADD_SUBPLOT_KWARGS, kwargs)
        ax = fig.add_subplot(**subplot_kwargs)
    return fig, ax


class _RepPlotter[RepT: AnyReps](abc.ABC):
    def __init__(self, series: RepT) -> None:
        self.rep: RepT = copy.deepcopy(series)

    @abc.abstractmethod
    def __call__(
        self,
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
        *,
        xlabel: Literal[False] | str = "auto",
        ylabel: Literal[False] | str = False,
        set_legend: bool = False,
        plot_mode: LINE_PLOT_MODES | Literal["default"] = "default",
        **kwargs: Unpack[_PlotKwargs],
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: ...


class _LineRepPlotter[RepT: AnyReps](_RepPlotter[RepT], abc.ABC):
    DEFAULT_PLOT_METHOD: LINE_PLOT_MODES = "plot"

    @abc.abstractmethod
    def _get_xaxis_xlabel(self) -> tuple[Array, str]: ...

    def _get_line(self, batch_idx: int) -> Array:
        return self.rep.entries[batch_idx].squeeze()

    @classmethod
    def _get_plot_callable(
        cls,
        ax: matplotlib.axes.Axes,
        plot_mode: LINE_PLOT_MODES | Literal["default"] = "default",
    ):
        if plot_mode == "default":
            plot_mode = cls.DEFAULT_PLOT_METHOD
        return getattr(ax, plot_mode)

    def _guard_shape(self) -> None:
        if self.rep.entries.shape[1:4] != (1, 1, 1):
            msg = (
                f"{type(self).__name__} only supports n_channels=1, "
                "n_harmonics=1, n_features=1. "
                f"Got {self.rep.entries.shape[1:4]}."
            )
            raise ValueError(msg)

    def __call__(
        self,
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
        *,
        xlabel: Literal[False] | str = "auto",
        ylabel: Literal[False] | str = False,
        set_legend: bool = False,
        plot_mode: LINE_PLOT_MODES | Literal["default"] = "default",
        **kwargs: Unpack[_PlotKwargs],
    ):
        """Plot the series on the provided Axes."""
        self._guard_shape()
        fig, ax = _get_fig_ax(fig, ax, **kwargs)
        line_kwargs = sieve_kwargs(LINE_KWARGS, kwargs)
        xaxis, auto_xlabel = self._get_xaxis_xlabel()

        _ax_plot = self._get_plot_callable(ax, plot_mode)

        for batch_idx in range(self.rep.entries.shape[0]):
            _ax_plot(xaxis, self._get_line(batch_idx), **line_kwargs)
        if xlabel:
            _xlabel = auto_xlabel if xlabel == "auto" else xlabel
            ax.set_xlabel(_xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if set_legend:
            legend_kwargs = sieve_kwargs(LEGEND_KWARGS, kwargs)
            ax.legend(**legend_kwargs)
        return fig, ax


def _get_times_and_label(times: AnyAxis) -> tuple[Array, str]:
    # Default unit is seconds
    duration = times.stop - times.start
    # If the duration is more than a week, use days as the time unit
    hour = 3600
    day = hour * 24
    week = day * 7
    if duration > week:
        time_unit = "days"
        xaxis = times.asarray(np) / day
    # If the duration is more than 2 hours, use hours as the time unit
    elif duration > 2 * hour:
        time_unit = "hrs"
        xaxis = times.asarray(np) / hour
    # Otherwise, use seconds as the time unit
    else:
        time_unit = "s"
        xaxis = times.asarray(np)
    return xaxis, f"Time [{time_unit}]"


def _get_freqs_and_label(freqs: AnyAxis) -> tuple[Array, str]:
    # Default unit is Hz
    max_freq = freqs.stop
    # If the maximum frequency is more than 1 kHz, use kHz as the frequency unit
    kHz = 1e3  # noqa: N806
    mHz = 1e-3  # noqa: N806
    if max_freq > kHz:
        freq_unit = "kHz"
        xaxis = freqs.asarray(np) / kHz
    # If the maximum frequency is more than 1 Hz, use Hz as the frequency unit
    elif max_freq > 1:
        freq_unit = "Hz"
        xaxis = freqs.asarray(np)
    # Otherwise, use mHz as the frequency unit
    else:
        freq_unit = "mHz"
        xaxis = freqs.asarray(np) / mHz
    return xaxis, f"Frequency [{freq_unit}]"


class TSPlotter(_LineRepPlotter[representations.TimeSeries["AnyAxis"]]):
    """Plotter for :class:`~types.TimeSeries`."""

    def _get_xaxis_xlabel(self) -> tuple[Array, str]:
        return _get_times_and_label(self.rep.times)


class FSPlotter(_LineRepPlotter[representations.FrequencySeries["AnyAxis"]]):
    """Plotter for :class:`~types.FrequencySeries`."""

    DEFAULT_PLOT_METHOD: LINE_PLOT_MODES = "loglog"

    def _get_xaxis_xlabel(self) -> tuple[Array, str]:
        return _get_freqs_and_label(self.rep.frequencies)

    def _get_line(self, batch_idx: int) -> Array:
        return self.rep.abs().entries[batch_idx].squeeze()


class PhasorPlotter[AxisT: "AnyAxis"](_LineRepPlotter[representations.Phasor[AxisT]]):
    """Plotter for :class:`~types.Phasor`."""

    DEFAULT_PLOT_METHOD: LINE_PLOT_MODES = "semilogx"

    def _guard_shape(self) -> None:
        if self.rep.entries.shape[1:4] != (1, 1, 2):
            msg = (
                f"{type(self).__name__} only supports n_channels=1, "
                "n_harmonics=1, n_features=2. "
                f"Got {self.rep.entries.shape[1:4]}."
            )
            raise ValueError(msg)

    def _get_line(self, batch_idx: int) -> Array:
        return self.rep.phases[batch_idx].squeeze()

    _get_xaxis_xlabel = FSPlotter._get_xaxis_xlabel  # pyright: ignore[reportPrivateUsage, reportUnannotatedClassAttribute]


class _ImRepPlotter[RepT: AnyReps](_RepPlotter[RepT], abc.ABC):
    @abc.abstractmethod
    def _get_x_extent_and_label(self) -> tuple[tuple[float, float], str]: ...

    @abc.abstractmethod
    def _get_y_extent_and_label(self) -> tuple[tuple[float, float], str]: ...

    @abc.abstractmethod
    def _get_toshow(self) -> Array: ...

    def _guard_shape(self) -> None:
        if self.rep.entries.shape[0:4] != (1, 1, 1, 1):
            msg = (
                f"{type(self).__name__} only supports n_batches=1, n_channels=1, "
                "n_harmonics=1, n_features=1. "
                f"Got {self.rep.entries.shape[0:4]}."
            )
            raise ValueError(msg)

    def __call__(
        self,
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
        *,
        xlabel: Literal[False] | str = "auto",
        ylabel: Literal[False] | str = False,
        set_legend: bool = False,
        plot_mode: LINE_PLOT_MODES | Literal["default"] = "default",
        **kwargs: Unpack[_PlotKwargs],
    ):
        """Plot the series on the provided Axes."""
        self._guard_shape()
        if plot_mode != "default":
            msg = (
                f"{type(self).__name__} only supports plot_mode='default'. "
                f"Got {plot_mode}."
            )
            raise ValueError(msg)
        fig, ax = _get_fig_ax(fig, ax, **kwargs)
        imshow_kwargs = sieve_kwargs(IMSHOW_KWARGS, kwargs)
        x_extent, auto_xlabel = self._get_x_extent_and_label()
        y_extent, auto_ylabel = self._get_y_extent_and_label()
        extent = x_extent + y_extent
        ax.imshow(
            self._get_toshow(),
            origin=imshow_kwargs.pop("origin", "lower"),
            aspect=imshow_kwargs.pop("aspect", "auto"),
            extent=imshow_kwargs.pop("extent", extent),
            **imshow_kwargs,  # pyright: ignore[reportCallIssue]
        )
        if xlabel:
            x_label = auto_xlabel if xlabel == "auto" else xlabel
            ax.set_xlabel(x_label)
        if ylabel:
            y_label = auto_ylabel if ylabel == "auto" else ylabel
            ax.set_ylabel(y_label)
        if set_legend:
            legend_kwargs = sieve_kwargs(LEGEND_KWARGS, kwargs)
            ax.legend(**legend_kwargs)
        return fig, ax


class STFTPlotter[GridT: Grid2D[AnyAxis, AnyAxis]](
    _ImRepPlotter[representations.STFT[GridT]]
):
    """Plotter for :class:`~types.STFT`."""

    def _get_x_extent_and_label(self):
        times, time_label = _get_times_and_label(self.rep.times)
        return (cast("float", times[0]), cast("float", times[-1])), time_label

    def _get_y_extent_and_label(self):
        freqs, freq_label = _get_freqs_and_label(self.rep.frequencies)
        return (cast("float", freqs[0]), cast("float", freqs[-1])), freq_label

    def _get_toshow(self) -> Array:
        return self.rep.abs().entries.squeeze()  # shape (n_freqs, n_times)


class WDMPlotter[GridT: Grid2D[Axis[Linspace], Axis[Linspace]]](
    _ImRepPlotter[representations.WDM[GridT]]
):
    """Plotter for :class:`~types.WDM`."""

    def _get_x_extent_and_label(self):
        _orig_times = self.rep.times.ax
        _times = misc.axis(
            misc.linspace(_orig_times.start, _orig_times.stop, _orig_times.num + 1)
        )
        times, time_label = _get_times_and_label(_times)
        return (cast("float", times[0]), cast("float", times[-1])), time_label

    def _get_y_extent_and_label(self):
        _orig_freqs = self.rep.frequencies.ax
        _freqs = misc.axis(
            misc.linspace(_orig_freqs.start, _orig_freqs.stop, _orig_freqs.num + 1)
        )
        freqs, freq_label = _get_freqs_and_label(_freqs)
        return (cast("float", freqs[0]), cast("float", freqs[-1])), freq_label

    def _get_toshow(self) -> Array:
        return self.rep.entries.squeeze()  # shape (n_freqs, n_times)


def _is_plottable(thing: Any, /) -> TypeIs[_IsPlottable[AnyReps]]:
    return hasattr(thing, "plot") and isinstance(thing.plot, property)


def _is_chanmap(
    thing: Any,
    /,
) -> TypeIs[_mixins.ChannelMapping[_IsPlottable[AnyReps]]]:
    return isinstance(thing, _mixins.ChannelMapping) and all(
        _is_plottable(chn_rep) for chn_rep in thing.values()
    )


def _is_modemap_rep(
    thing: Any,
    /,
) -> TypeIs[_mixins.ModeMapping[_mixins.Mode, _IsPlottable[AnyReps]]]:
    return isinstance(thing, _mixins.ModeMapping) and all(
        _is_plottable(mode_rep) for mode_rep in thing.values()
    )


def _is_modemap_chanmap(
    thing: Any,
    /,
) -> TypeIs[waveforms.HarmonicProjectedWaveform[_mixins.Mode, AnyReps]]:
    return isinstance(thing, waveforms.HarmonicProjectedWaveform)


def _get_name_and_label(
    thing: Any, /, label_base: str | None
) -> tuple[str | None, str | None]:
    _name: str | None = getattr(thing, "name", None)
    label = f"{label_base} {_name}" if label_base else _name
    return _name, label


def _get_diff_label(thing1: Any, thing2: Any, /, label_base: str | None) -> str:
    _, label1 = _get_name_and_label(thing1, label_base=label_base)
    _, label2 = _get_name_and_label(thing2, label_base=label_base)
    if label1 and label2:
        return f"{label1} - {label2}"
    if label_base:
        return f"{label_base} 1 - {label_base} 2"
    return "1 - 2"


def plot(
    thing: Any,
    /,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
    plot_mode: LINE_PLOT_MODES | Literal["default"] = "default",
    **kwargs: Unpack[_PlotKwargs],
):
    """Plot the given object."""
    msg = (
        "This object "
        "cannot be plotted on a single Axes, "
        "so the `ax` argument should not be provided."
    )
    if _is_plottable(thing):
        fig, ax = thing.plot(fig=fig, ax=ax, plot_mode=plot_mode, **kwargs)
        return fig, [ax]
    if fig is None:
        fig_kwargs = sieve_kwargs(FIGURE_KWARGS, kwargs)
        fig = plt.figure(**fig_kwargs)
    label_base = kwargs.pop("label", None)
    if _is_chanmap(thing):
        chn_num = len(thing.channel_names)
        if ax is not None and chn_num > 1:
            raise ValueError(msg)
        _axs = fig.subplots(chn_num, sharex=True)
        axs: list[matplotlib.axes.Axes] = _axs if chn_num > 1 else [_axs]
        for idx, chnname in enumerate(thing.channel_names):
            xlabel_bool = idx == chn_num - 1
            xlabel = "auto" if xlabel_bool else False
            _name, _label = _get_name_and_label(thing, label_base)
            thing[chnname].plot(
                ax=axs[idx],
                xlabel=xlabel,
                ylabel=chnname,
                plot_mode=plot_mode,
                label=_label,
                **kwargs,  # pyright: ignore[reportCallIssue]
            )
        return fig, axs
    if _is_modemap_rep(thing):
        mode_num = len(thing.harmonics)
        if ax is None:
            ax = fig.subplots(1)
        for idx, mode in enumerate(thing.harmonics):
            xlabel_bool = idx == mode_num - 1
            xlabel = "auto" if xlabel_bool else False
            _label = f"{label_base} {mode!r}" if label_base else repr(mode)
            thing[mode].plot(
                ax=ax,
                xlabel=xlabel,
                ylabel=False,
                plot_mode=plot_mode,
                label=_label,
                **kwargs,  # pyright: ignore[reportCallIssue]
            )
    if _is_modemap_chanmap(thing):
        mode_num = len(thing.channel_names)

    _msg = f"The plotting of {type(thing)!r} is not implemented."
    raise NotImplementedError(_msg)


def plot_compare(
    thing1: Any,
    thing2: Any,
    /,
    fig: matplotlib.figure.Figure | None = None,
    plot_mode: LINE_PLOT_MODES | Literal["default"] = "default",
    **kwargs: Unpack[_PlotKwargs],
):
    """Plot the comparison of two objects."""
    if fig is None:
        fig_kwargs = sieve_kwargs(FIGURE_KWARGS, kwargs)
        fig = plt.figure(**fig_kwargs)
    label_base = kwargs.pop("label", None)
    try:
        diff = thing1 - thing2
    except (ValueError, TypeError):
        diff = None
    ncols = 2 if diff is not None else 1

    if _is_plottable(thing1) and _is_plottable(thing2):
        _axs = fig.subplots(ncols=ncols, sharey=False)
        # It is actually possibly a numpy array of object but
        # let us just say it is a list
        axs: list[matplotlib.axes.Axes] = _axs if ncols > 1 else [_axs]  # pyright: ignore[reportRedeclaration]
        for idx, thing in enumerate((thing1, thing2)):
            _label = f"{label_base} {idx + 1}" if label_base else f"{idx + 1}"
            thing.plot(ax=axs[0], plot_mode=plot_mode, label=_label, **kwargs)  # pyright: ignore[reportCallIssue]
        if diff is not None:
            _label = _get_diff_label(thing1, thing2, label_base)
            diff.plot(ax=axs[1], plot_mode=plot_mode, label=_label, **kwargs)  # pyright: ignore[reportCallIssue]
        return fig, axs

    if _is_chanmap(thing1) and _is_chanmap(thing2):
        if thing1.channel_names != thing2.channel_names:
            msg = "The two objects have different channels, cannot be compared."
            raise ValueError(msg)
        chn_num = len(thing1.channel_names)
        _axs = fig.subplots(nrows=chn_num, ncols=ncols, sharex=True)
        axs: list[list[matplotlib.axes.Axes]] = (
            [[_axs]]
            if chn_num == 1 and ncols == 1
            else [_axs]
            if chn_num == 1 or ncols == 1
            else _axs
        )

        for idx, chnname in enumerate(thing1.channel_names):
            xlabel_bool = idx == chn_num - 1
            xlabel = "auto" if xlabel_bool else False
            for thing in (thing1, thing2):
                _, _label = _get_name_and_label(thing, label_base)
                thing[chnname].plot(
                    ax=axs[idx][0],
                    xlabel=xlabel,
                    ylabel=chnname,
                    plot_mode=plot_mode,
                    label=_label,
                    **kwargs,  # pyright: ignore[reportCallIssue]
                )
            _label = _get_diff_label(thing1, thing2, label_base)
            if diff is not None:
                diff[chnname].plot(
                    ax=axs[idx][1],
                    xlabel=xlabel,
                    ylabel=chnname,
                    plot_mode=plot_mode,
                    label=_label,
                    **kwargs,  # pyright: ignore[reportCallIssue]
                )
        return fig, axs

    _msg = (
        f"The comparison of {type(thing1)!r} and {type(thing2)!r} is not implemented."
    )
    raise NotImplementedError(_msg)


# class _DataPlotter1D[
#     RepT: representations.TimeSeries["AnyAxis"]
#     | representations.FrequencySeries["AnyAxis"],
# ](abc.ABC):
#     def __init__(self, data: data.Data[RepT]) -> None:
#         self.data: data.Data[RepT] = copy.deepcopy(data)

#     def _draw(
#         self,
#         plotter: type[Any],
#         *,
#         set_legend: bool = False,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         chn_num = len(self.data.channel_names)

#         fig, axs = plt.subplots(chn_num, sharex=True)  # pyright: ignore[reportUnknownMemberType]

#         # If only one channel, axs is not a list
#         try:
#             axs[0]
#         except TypeError:
#             axs = [axs]

#         kwargs_dict = dict(kwargs)
#         label = kwargs_dict.pop("label", self.data.name)
#         forward_kwargs = sieve_kwargs(LINE_KWARGS.difference({"label"}), kwargs_dict)

#         for idx, chnname in enumerate(self.data.channel_names):
#             xlabel_bool = idx == chn_num - 1
#             plotter(self.data[chnname]).plot(
#                 axs[idx],
#                 set_xlabel=xlabel_bool,
#                 set_ylabel=True,
#                 set_legend=set_legend,
#                 label=label,
#                 ylabel=f"{chnname}",
#                 **forward_kwargs,
#             )
#         return fig

#     def _compare(
#         self,
#         plotter: type[Any],
#         other: Self,
#         *,
#         set_ylabel: bool = False,
#         diff_ylabel: str = "difference",
#         plot_residual: bool = True,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         chn_num = len(self.data.channel_names)
#         if plot_residual:
#             fig, axs = plt.subplots(2 * chn_num, sharex=True)  # pyright: ignore[reportUnknownMemberType]
#         else:
#             fig, axs = plt.subplots(chn_num, sharex=True)  # pyright: ignore[reportUnknownMemberType]
#         forward_kwargs = sieve_kwargs(LINE_KWARGS.difference({"label"}), dict(kwargs))
#         for idx, chnname in enumerate(self.data.channel_names):
#             xlabel_bool = idx == chn_num - 1
#             if plot_residual:
#                 orig_idx = 2 * idx
#                 diff_idx = 2 * idx + 1
#             else:
#                 orig_idx = idx
#             plotter(self.data[chnname]).plot(
#                 axs[orig_idx],
#                 set_xlabel=False,
#                 set_legend=True,
#                 set_ylabel=set_ylabel,
#                 label=self.data.name,
#                 ylabel=f"{chnname}",
#                 **forward_kwargs,
#             )
#             plotter(other.data[chnname]).plot(
#                 axs[orig_idx],
#                 set_legend=True,
#                 set_xlabel=xlabel_bool and not plot_residual,
#                 label=other.data.name,
#                 **forward_kwargs,
#             )
#             if plot_residual:
#                 plotter(self.data[chnname] - other.data[chnname]).plot(
#                     axs[diff_idx],  # pyright: ignore[reportPossiblyUnboundVariable]
#                     set_xlabel=xlabel_bool,
#                     set_ylabel=set_ylabel,
#                     set_legend=False,
#                     plot_mode="semilogx",
#                     ylabel=f"{chnname}: {diff_ylabel}",
#                     **forward_kwargs,
#                 )
#         return fig

#     @abc.abstractmethod
#     def draw(self, **kwargs: Any) -> matplotlib.figure.Figure: ...

#     @abc.abstractmethod
#     def compare(self, other: Self, **kwargs: Any) -> matplotlib.figure.Figure: ...


# class TSDataPlotter(_DataPlotter1D["representations.TimeSeries[AnyAxis]"]):
#     """Plotter for :class:`.containers.data.TSData`."""

#     def draw(
#         self,
#         *,
#         set_legend: bool = False,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         """Draw the time series data."""
#         return self._draw(plotter=TSPlotter, set_legend=set_legend, **kwargs)

#     def compare(  # pyright: ignore[reportIncompatibleMethodOverride]
#         self,
#         other: Self,
#         *,
#         set_ylabel: bool = False,
#         diff_ylabel: str = "difference",
#         plot_residual: bool = True,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         """Compare two time series data."""
#         return self._compare(
#             plotter=TSPlotter,
#             other=other,
#             set_ylabel=set_ylabel,
#             diff_ylabel=diff_ylabel,
#             plot_residual=plot_residual,
#             **kwargs,
#         )


# class FSDataPlotter(_DataPlotter1D["representations.FrequencySeries[AnyAxis]"]):
#     """Plotter for :class:`.containers.data.FSData`."""

#     def _draw_angle(
#         self,
#         *,
#         set_legend: bool = False,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         chn_num = len(self.data.channel_names)
#         fig, axs = plt.subplots(2 * chn_num, sharex=True)  # pyright: ignore[reportUnknownMemberType]
#         forward_kwargs = sieve_kwargs(LINE_KWARGS.difference({"label"}), dict(kwargs))
#         for idx, chnname in enumerate(self.data.channel_names):
#             xlabel_bool = idx == chn_num - 1
#             amp_idx = 2 * idx
#             phase_idx = 2 * idx + 1
#             plotter = FSPlotter(self.data[chnname])
#             plotter.plot(
#                 axs[amp_idx],
#                 set_xlabel=False,
#                 set_ylabel=True,
#                 set_legend=set_legend,
#                 label=self.data.name,
#                 ylabel=f"{chnname} Amplitude",
#                 **forward_kwargs,
#             )
#             plotter.plot_angle(
#                 axs[phase_idx],
#                 set_xlabel=xlabel_bool,
#                 set_ylabel=True,
#                 set_legend=False,
#                 label=self.data.name,
#                 **forward_kwargs,
#             )
#         return fig

#     def _compare_angle(
#         self,
#         other: Self,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         forward_kwargs = sieve_kwargs(LINE_KWARGS.difference({"label"}), dict(kwargs))
#         chn_num = len(self.data.channel_names)
#         fig, axs = plt.subplots(4 * chn_num, sharex=True)  # pyright: ignore[reportUnknownMemberType]
#         for idx, chnname in enumerate(self.data.channel_names):
#             xlabel_bool = idx == chn_num - 1
#             orig_amp_idx = 4 * idx
#             orig_phase_idx = 4 * idx + 1
#             diff_amp_idx = 4 * idx + 2
#             diff_phase_idx = 4 * idx + 3
#             self_plotter = FSPlotter(self.data[chnname])
#             other_plotter = FSPlotter(other.data[chnname])
#             diff_plotter = FSPlotter(self.data[chnname] - other.data[chnname])
#             self_plotter.plot(
#                 axs[orig_amp_idx],
#                 set_xlabel=False,
#                 set_ylabel=True,
#                 set_legend=True,
#                 label=self.data.name,
#                 ylabel=f"{chnname} Amplitude",
#                 **forward_kwargs,
#             )
#             other_plotter.plot(
#                 axs[orig_amp_idx],
#                 label=other.data.name,
#                 **forward_kwargs,
#             )
#             self_plotter.plot(
#                 axs[orig_phase_idx],
#                 set_xlabel=False,
#                 set_ylabel=True,
#                 set_legend=True,
#                 label=self.data.name,
#                 ylabel=f"{chnname} Phase [rad]",
#                 **forward_kwargs,
#             )
#             other_plotter.plot(
#                 axs[orig_phase_idx],
#                 label=other.data.name,
#                 **forward_kwargs,
#             )
#             diff_plotter.plot(
#                 axs[diff_amp_idx],
#                 set_xlabel=False,
#                 set_ylabel=True,
#                 set_legend=True,
#                 label=f"{chnname}: {self.data.name} - {other.data.name}",
#                 **forward_kwargs,
#             )
#             diff_plotter.plot(
#                 axs[diff_phase_idx],
#                 set_xlabel=xlabel_bool,
#                 set_ylabel=True,
#                 set_legend=True,
#                 label=f"{chnname}: {self.data.name} - {other.data.name}",
#                 **forward_kwargs,
#             )
#         return fig

#     def draw(
#         self,
#         *,
#         set_legend: bool = False,
#         angle: bool = False,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         """Draw the frequency series data."""
#         if not angle:
#             return self._draw(plotter=FSPlotter, set_legend=set_legend, **kwargs)
#         return self._draw_angle(set_legend=set_legend, **kwargs)

#     def compare(  # pyright: ignore[reportIncompatibleMethodOverride]
#         self,
#         other: Self,
#         *,
#         set_ylabel: bool = False,
#         diff_ylabel: str = "difference",
#         angle: bool = False,
#         plot_residual: bool = True,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         """Compare two frequency series data."""
#         if not angle:
#             return self._compare(
#                 plotter=FSPlotter,
#                 other=other,
#                 set_ylabel=set_ylabel,
#                 diff_ylabel=diff_ylabel,
#                 plot_residual=plot_residual,
#                 **kwargs,
#             )
#         return self._compare_angle(other=other, **kwargs)


# class TFDataPlotter[
#     DataT: data.WDMData[Grid2D[Axis[Linspace], Axis[Linspace]]]
#     | data.STFTData[Grid2D[Axis[Linspace], Axis[Linspace]]],
# ]:
#     """Plotter for :class:`.containers.data.TFData`."""

#     def __init__(self, data: DataT) -> None:
#         self.data: DataT = copy.deepcopy(data)

#     def draw(
#         self,
#         *,
#         set_legend: bool = False,
#         **kwargs: Unpack[_PlotKwargs],
#     ) -> matplotlib.figure.Figure:
#         """Draw the time-frequency data."""
#         fig, axs = plt.subplots(len(self.data.channel_names), sharex=True)  # pyright: ignore[reportUnknownMemberType]
#         # If only one channel, axs is not a list
#         try:
#             axs[0]
#         except TypeError:
#             axs = [axs]
#         for idx, chnname in enumerate(self.data.channel_names):
#             xlabel_bool = idx == len(self.data.channel_names) - 1
#             self.data[chnname].get_plotter().plot(
#                 axs[idx],
#                 set_xlabel=xlabel_bool,
#                 set_ylabel=True,
#                 set_legend=set_legend,
#                 **kwargs,
#             )
#         return fig
