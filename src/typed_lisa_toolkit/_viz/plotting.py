"""Module for plotting tools."""

# pyright: reportUnknownMemberType=false

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    Unpack,
    cast,
    final,
)

import astropy.units as u
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..types import (
    STFT,
    WDM,
    AnyAxis,
    Array,
    Axis,
    FrequencyPhasor,
    FrequencySeries,
    Grid2D,
    Grid2DSparse,
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    Linspace,
    TimePhasor,
    TimeSeries,
    _mixins,
)
from ..types.waveforms import harmonic_waveform
from .options import (
    FIGURE_KWARGS,
    IMSHOW_KWARGS,
    LEGEND_KWARGS,
    LINE_KWARGS,
    LINE_PLOT_MODES,
    SUBPLOT_KWARGS,
    LineKwargs,
    PlotKwargs,
    sieve_kwargs,
)

LineRep = (
    FrequencySeries[AnyAxis]
    | TimeSeries[AnyAxis]
    | TimePhasor[AnyAxis]
    | FrequencyPhasor[AnyAxis]
)
ImRep = STFT[Grid2D[AnyAxis, AnyAxis]] | WDM[Grid2D[Axis[Linspace], Axis[Linspace]]]
HMRep = HarmonicWaveform[_mixins.Mode, LineRep] | HarmonicWaveform[_mixins.Mode, ImRep]
ChanRep = _mixins.ChannelMapping[LineRep] | _mixins.ChannelMapping[ImRep]
HPW = HarmonicProjectedWaveform[_mixins.Mode, LineRep]
AxPlottable = LineRep | ImRep | HMRep
MultChan = ChanRep | HPW

# ═════════════════════════════════════════════════════════════════════════════
# Plot specification types and utilities
# ═════════════════════════════════════════════════════════════════════════════

LinePlotMode = Literal["loglog", "semilogx", "plot", "semilogy", "imshow"]
Unit = u.Unit | u.IrreducibleUnit | u.CompositeUnit


def _format_axis_label(name: Literal[False] | str, unit_str: str) -> str:
    if not name:
        return ""
    if unit_str:
        return f"{name} ({unit_str})"
    return name


def _get_unit_scale(
    from_unit: Unit,
    to_unit: Unit,
) -> float:
    try:
        scale = cast("float", from_unit.to(to_unit))
    except u.UnitConversionError as e:
        msg = f"Incompatible units: {from_unit} and {to_unit}"
        raise ValueError(msg) from e
    return scale


def _swap_hpw(hpw: HPW):
    return {
        chnname: harmonic_waveform({mode: hpw[mode][chnname] for mode in hpw.harmonics})
        for chnname in hpw.channel_names
    }


def listify_axs(
    axs: Sequence[Sequence[matplotlib.axes.Axes]] | np.ndarray,
) -> list[list[matplotlib.axes.Axes]]:

    def _tolist(
        row: Sequence[matplotlib.axes.Axes] | np.ndarray,
    ) -> list[matplotlib.axes.Axes]:
        if isinstance(row, np.ndarray):
            return row.tolist()
        return list(row)

    return [_tolist(row) for row in axs]


class XYLabel(abc.ABC):
    xunit: Unit
    yunit: Unit
    xname: Literal[False] | str
    yname: Literal[False] | str

    @property
    def xlabel(self):
        # Assume sequence of length 1 or always normalized beforehand
        unit_str = self.xunit.to_string() if self.xunit else ""
        return _format_axis_label(self.xname, unit_str)

    @property
    def ylabel(self):
        # Assume sequence of length 1 or always normalized beforehand
        unit_str = self.yunit.to_string() if self.yunit else ""
        return _format_axis_label(self.yname, unit_str)


def _guard_shape(obj: LineRep | ImRep) -> None:
    if isinstance(obj, (TimePhasor, FrequencyPhasor)):
        if obj.entries.shape[1:4] != (1, 1, 2):
            msg = (
                f"{type(obj).__name__} only supports n_channels=1, "
                "n_harmonics=1, n_features=2. "
                f"Got {obj.entries.shape[1:4]}."
            )
            raise ValueError(msg)
        return
    if obj.entries.shape[1:4] != (1, 1, 1):
        msg = (
            f"{type(obj).__name__} only supports n_channels=1, "
            "n_harmonics=1, n_features=1. "
            f"Got {obj.entries.shape[1:4]}."
        )
        raise ValueError(msg)


def _get_times_axis(times: AnyAxis) -> tuple[Array, Unit]:
    # Default unit is seconds
    duration = times.stop - times.start
    # If the duration is more than a week, use days as the time unit
    hour = cast("float", u.hour.to(u.second))
    day = cast("float", u.day.to(u.second))
    week = cast("float", u.week.to(u.second))
    if duration > week:
        time_unit = u.day
        xaxis = times.asarray(np) / day
    # If the duration is more than 2 hours, use hours as the time unit
    elif duration > 2 * hour:
        time_unit = u.hour
        xaxis = times.asarray(np) / hour
    # Otherwise, use seconds as the time unit
    else:
        time_unit = u.second
        xaxis = times.asarray(np)
    return xaxis, time_unit


def _get_freqs_axis(freqs: AnyAxis) -> tuple[Array, Unit]:
    # Default unit is Hz
    max_freq = freqs.stop
    # If the maximum frequency is more than 1 kHz, use kHz as the frequency unit
    kHz = cast("float", u.kHz.to(u.Hz))  # noqa: N806
    mHz = cast("float", u.mHz.to(u.Hz))  # noqa: N806
    if max_freq > kHz:
        freq_unit = u.kHz
        xaxis = freqs.asarray(np) / kHz
    # If the maximum frequency is more than 1 Hz, use Hz as the frequency unit
    elif max_freq > 1:
        freq_unit = u.Hz
        xaxis = freqs.asarray(np)
    # Otherwise, use mHz as the frequency unit
    else:
        freq_unit = u.mHz
        xaxis = freqs.asarray(np) / mHz
    return xaxis, freq_unit


# ═════════════════════════════════════════════════════════════════════════════
# Plotting classes
# ═════════════════════════════════════════════════════════════════════════════


@final
class SingleLine(XYLabel):
    def __init__(
        self,
        xaxis: Array,
        line: Array,
        *,
        xunit: Unit | None = None,
        yunit: Unit | None = None,
        xname: Literal[False] | str = False,
        yname: Literal[False] | str = False,
    ) -> None:
        self.xaxis = xaxis
        self.line = line
        self.xunit = u.dimensionless_unscaled if xunit is None else xunit
        self.yunit = u.dimensionless_unscaled if yunit is None else yunit
        self.xname: str | Literal[False] = xname
        self.yname: str | Literal[False] = yname

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        plot_mode: LinePlotMode,
        **kwargs: Unpack[PlotKwargs],
    ):
        _ax_plot = getattr(ax, plot_mode)
        _kwargs = sieve_kwargs(LINE_KWARGS, kwargs)
        _ax_plot(self.xaxis, self.line, **_kwargs)
        return ax

    def normalize(self, xunit: Unit, yunit: Unit):
        xaxis = self.xaxis * _get_unit_scale(self.xunit, xunit)
        line = self.line * _get_unit_scale(self.yunit, yunit)
        self.xunit = xunit
        self.yunit = yunit
        self.xaxis = xaxis
        self.line = line
        return self


@final
class SingleImage(XYLabel):
    def __init__(
        self,
        toshow: Array,
        *,
        extent: tuple[float, float, float, float],
        xunit: Unit | None = None,
        yunit: Unit | None = None,
        xname: Literal[False] | str = False,
        yname: Literal[False] | str = False,
    ):
        self.toshow = toshow
        self.extent = extent
        self.xunit = u.dimensionless_unscaled if xunit is None else xunit
        self.yunit = u.dimensionless_unscaled if yunit is None else yunit
        self.xname: str | Literal[False] = xname
        self.yname: str | Literal[False] = yname

    def normalize(self, xunit: Unit, yunit: Unit):
        extent = (
            self.extent[0] * _get_unit_scale(self.xunit, xunit),
            self.extent[1] * _get_unit_scale(self.xunit, xunit),
            self.extent[2] * _get_unit_scale(self.yunit, yunit),
            self.extent[3] * _get_unit_scale(self.yunit, yunit),
        )
        self.xunit = xunit
        self.yunit = yunit
        self.extent = extent
        return self

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        plot_mode: LinePlotMode,
        **kwargs: Unpack[PlotKwargs],
    ):
        if plot_mode != "imshow":
            msg = (
                "SingleImage can only be plotted with plot_mode='imshow'. "
                f"Got {plot_mode!r}."
            )
            raise ValueError(msg)
        image_kwargs = sieve_kwargs(IMSHOW_KWARGS, kwargs)
        image_kwargs["origin"] = image_kwargs.get("origin", "lower")
        image_kwargs["aspect"] = image_kwargs.get("aspect", "auto")
        ax.imshow(
            self.toshow,
            **image_kwargs,
        )
        return ax


def get_suitable_units(peers: Sequence[SingleLine | SingleImage], /):
    final_xunit = peers[0].xunit
    final_yunit = peers[0].yunit

    for peer in peers:
        xscale = _get_unit_scale(peer.xunit, final_xunit)
        yscale = _get_unit_scale(peer.yunit, final_yunit)
        if xscale > 1:
            final_xunit = peer.xunit
        if yscale > 1:
            final_yunit = peer.yunit
    return final_xunit, final_yunit


@final
class LabelContext:
    def __init__(self, prefix: str = "", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix

    def __call__(self, **kwargs: Unpack[LineKwargs]) -> LineKwargs:
        base = kwargs.get("label", "")
        if self.prefix:
            base = f"{self.prefix} {base}" if base else self.prefix
        if self.suffix:
            base = f"{base} {self.suffix}" if base else self.suffix
        return kwargs | {"label": base}


trivial_ctx = LabelContext()


@dataclass
class PlotNode:
    label_ctx: LabelContext
    payload: SingleLine | SingleImage | None = None
    plot_mode: LinePlotMode | None = None
    children: list["PlotNode"] | None = None
    multirow: Literal[False] | list[str] = False
    _normalized: bool = False
    _all_leaves: list["PlotNode"] | None = None

    def __post_init__(self):
        if self.payload is not None and self.children is not None:
            msg = "A PlotNode cannot have both a payload and children."
            raise ValueError(msg)
        if self.payload is None and self.children is None:
            msg = "A PlotNode must have either a payload or children."
            raise ValueError(msg)
        if self.payload is not None and self.plot_mode is None:
            msg = "A leaf PlotNode with a payload must have a plot_mode."
            raise ValueError(msg)

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def get_all_leaves(self) -> list["PlotNode"]:
        if self._all_leaves is not None:
            return self._all_leaves
        if self.is_leaf:
            return [self]
        if self.children is None:
            msg = "Non-leaf node must have children."
            raise ValueError(msg)
        leaves: list[PlotNode] = []
        for child in self.children:
            leaves.extend(child.get_all_leaves())
        self._all_leaves = leaves
        return leaves

    def normalize(self):
        if self._normalized:
            return self
        payloads = [leave.payload for leave in self.get_all_leaves() if leave.payload]
        xunit, yunit = get_suitable_units(payloads)
        for payload in payloads:
            payload.normalize(xunit, yunit)
        self._normalized = True
        return self

    @property
    def xlabel(self) -> str:
        _ = self.normalize()
        leaves = self.get_all_leaves()
        return cast("SingleLine|SingleImage", leaves[0].payload).xlabel

    @property
    def ylabel(self) -> str:
        _ = self.normalize()
        leaves = self.get_all_leaves()
        return cast("SingleLine|SingleImage", leaves[0].payload).ylabel

    def __call__(
        self,
        axs: Sequence[matplotlib.axes.Axes],
        plot_mode: LinePlotMode | None = None,
        **kwargs: Unpack[LineKwargs],
    ):
        if self.payload:
            if len(axs) != 1:
                msg = "Expected a single Axes for a leaf node."
                raise ValueError(msg)
            plt_mode = plot_mode or self.plot_mode
            if plt_mode is None:
                msg = "Leaf node must have a plot mode."
                raise ValueError(msg)
            self.payload.plot(axs[0], plot_mode=plt_mode, **self.label_ctx(**kwargs))
            return axs
        if not self.children:
            msg = "Non-leaf node must have children."
            raise ValueError(msg)
        if not self.multirow:
            if len(axs) != 1:
                msg = "Expected a single Axes for a non-multirow node."
                raise ValueError(msg)
            for child in self.children:
                child(axs, plot_mode=plot_mode, **kwargs)
            return axs
        if len(axs) != len(self.multirow):
            msg = (
                f"Expected {len(self.multirow)} Axes for a multirow node, "
                f"but got {len(axs)}."
            )
            raise ValueError(msg)
        for ax, child in zip(axs, self.children, strict=True):
            child([ax], plot_mode=plot_mode, **kwargs)
        return axs


def classify(
    obj: AxPlottable | MultChan, /, label_ctx: LabelContext = trivial_ctx
) -> PlotNode:
    if isinstance(obj, (TimeSeries, TimePhasor)):
        _guard_shape(obj)
        xaxis, xunit = _get_times_axis(obj.times)
        _ary = obj.entries if isinstance(obj, TimeSeries) else obj.phases
        children = [
            PlotNode(
                payload=SingleLine(
                    xaxis=xaxis,
                    line=_ary[i].squeeze(),
                    xunit=xunit,
                    xname="Time",
                    yname=False,
                ),
                label_ctx=label_ctx,
                plot_mode="plot",
            )
            for i in range(obj.entries.shape[0])
        ]
        return PlotNode(
            children=children,
            label_ctx=label_ctx,
        ).normalize()
    if isinstance(obj, (FrequencySeries, FrequencyPhasor)):
        _guard_shape(obj)
        xaxis, xunit = _get_freqs_axis(obj.frequencies)
        _ary = obj.abs().entries if isinstance(obj, FrequencySeries) else obj.phases
        _plt_mode = "semilogx" if isinstance(obj, FrequencyPhasor) else "loglog"
        children = [
            PlotNode(
                payload=SingleLine(
                    xaxis=xaxis,
                    line=_ary[i].squeeze(),
                    xunit=xunit,
                    xname="Frequency",
                    yname=False,
                ),
                label_ctx=label_ctx,
                plot_mode=_plt_mode,
            )
            for i in range(obj.entries.shape[0])
        ]
        return PlotNode(
            children=children,
            label_ctx=label_ctx,
        ).normalize()
    if isinstance(obj, (STFT, WDM)):
        _guard_shape(obj)
        if isinstance(obj.grid, Grid2DSparse):
            msg = "Not implemented yet for sparse grid."
            raise NotImplementedError(msg)
        xaxis, xunit = _get_times_axis(obj.times)
        yaxis, yunit = _get_freqs_axis(obj.frequencies)
        if isinstance(obj, STFT):
            extent = (
                cast("float", xaxis[0]),
                cast("float", xaxis[-1]),
                cast("float", yaxis[0]),
                cast("float", yaxis[-1]),
            )
        else:
            extent = (
                cast("float", xaxis[0]),
                cast("float", xaxis[-1] + obj.dT),
                cast("float", yaxis[0]),
                cast("float", yaxis[-1] + obj.dF),
            )

        if obj.entries.shape[0] != 1:
            msg = "STFT and WDM with n_channels > 1 are not supported yet."
            raise NotImplementedError(msg)

        children = [
            PlotNode(
                payload=SingleImage(
                    toshow=obj.abs().entries[i].squeeze(),
                    extent=extent,
                    xunit=xunit,
                    yunit=yunit,
                    xname="Time",
                    yname="Frequency",
                ),
                label_ctx=label_ctx,
                plot_mode="imshow",
            )
            for i in range(obj.entries.shape[0])
        ]
        return PlotNode(
            children=children,
            label_ctx=label_ctx,
        ).normalize()

    if isinstance(obj, HarmonicWaveform):
        children: list[PlotNode] = [
            classify(obj[mode], label_ctx=LabelContext(prefix=str(tuple(mode))))
            for mode in obj.harmonics
        ]
        return PlotNode(
            children=children,
            label_ctx=label_ctx,
        ).normalize()
    if isinstance(obj, _mixins.ChannelMapping):
        children = [
            classify(obj[chn], label_ctx=label_ctx) for chn in obj.channel_names
        ]
        return PlotNode(
            children=children,
            label_ctx=label_ctx,
            multirow=list(obj.channel_names),
        ).normalize()
    if not isinstance(obj, HarmonicProjectedWaveform):  # pyright: ignore[reportUnnecessaryIsInstance]
        msg = f"Object of type {type(obj)!r} is not plottable."  # pyright: ignore[reportUnreachable]
        raise TypeError(msg)
    _obj = _swap_hpw(obj)
    children = [classify(_obj[chn], label_ctx=label_ctx) for chn in obj.channel_names]
    return PlotNode(
        children=children,
        label_ctx=label_ctx,
        multirow=list(obj.channel_names),
    ).normalize()


def _dispatch_plot(
    obj: AxPlottable | MultChan,
    /,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
    *,
    plot_mode: LINE_PLOT_MODES | None = None,
    set_legend: bool = True,
    **kwargs: Unpack[PlotKwargs],
):
    node = classify(obj)
    multirow = node.multirow or [""]
    nrows = len(multirow) if multirow else 1
    fig, axs = _get_fig_ax(fig, ax, nrows=nrows, ncols=1, **kwargs)
    _axs = listify_axs(np.asarray(axs))  # Single column
    node(_axs[0], plot_mode=plot_mode, **kwargs)
    _axs[0][-1].set_xlabel(node.xlabel)
    for idx, row_label in enumerate(multirow):
        _axs[0][idx].set_ylabel(row_label)
    if set_legend:
        legend_kwargs = sieve_kwargs(LEGEND_KWARGS, kwargs)
        for _ax in _axs[0]:
            _ax.legend(**legend_kwargs)
    if nrows == 1:
        _axs[0][0].set_ylabel(node.ylabel)
    else:
        fig.supylabel(node.ylabel)
    return fig, axs


def _dispatch_compare[T: AxPlottable | MultChan](
    obj1: T,
    obj2: T,
    /,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
    *,
    plot_mode: LINE_PLOT_MODES | None = None,
    set_legend: bool = True,
    showdiff: bool = False,
    **kwargs: Unpack[PlotKwargs],
):
    node1 = classify(obj1)
    node2 = classify(obj2)
    if node1.multirow != node2.multirow:
        msg = "Both objects must have the same multirow structure for comparison."
        raise ValueError(msg)
    multirow = node1.multirow or [""]
    nrows = len(multirow) if multirow else 1
    ncols = 2 if showdiff else 1
    fig, axs = _get_fig_ax(fig, ax, nrows=nrows, ncols=ncols, **kwargs)
    _axs = listify_axs(np.asarray(axs).T)
    node1(_axs[0], plot_mode=plot_mode, **kwargs)
    node2(_axs[0], plot_mode=plot_mode, **kwargs)
    _axs[0][-1].set_xlabel(node1.xlabel)
    for idx, row_label in enumerate(multirow):
        _axs[0][idx].set_ylabel(row_label)
    if showdiff:
        if isinstance(obj1, (TimePhasor, FrequencyPhasor)) or isinstance(
            obj2, (TimePhasor, FrequencyPhasor)
        ):
            msg = (
                "Showing the difference is not "
                f"supported for {type(obj1).__name__} objects."
            )
            raise ValueError(msg)
        node_diff = classify(obj1 - obj2)
        node_diff(_axs[1], plot_mode=plot_mode, **kwargs)
        _axs[1][0].set_title("Difference")
        _axs[1][-1].set_xlabel(node1.xlabel)
    if nrows == 1:
        _axs[0][0].set_ylabel(node1.ylabel)
    else:
        fig.supylabel(node1.ylabel)
    if set_legend:
        legend_kwargs = sieve_kwargs(LEGEND_KWARGS, kwargs)
        for _ax in _axs[0] + _axs[1]:
            _ax.legend(**legend_kwargs)
    return fig, axs


def _get_fig_ax(
    fig: matplotlib.figure.Figure | None,
    ax: matplotlib.axes.Axes | None,
    /,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, list[list[matplotlib.axes.Axes]]]:
    if fig is None:
        if ax is not None:
            msg = "If `ax` is provided, `fig` must also be provided."
            raise ValueError(msg)
        fig_kwargs = sieve_kwargs(FIGURE_KWARGS, kwargs)
        fig = plt.figure(**fig_kwargs)
    if ax is not None:
        if nrows != 1 or ncols != 1:
            msg = (
                "The `ax` argument is only supported when nrows=1 and ncols=1. "
                f"Got nrows={nrows} and ncols={ncols}."
            )
            raise ValueError(msg)
        return fig, listify_axs([[ax]])
    subplot_kwargs = sieve_kwargs(SUBPLOT_KWARGS, kwargs)
    subplot_kwargs["sharex"] = subplot_kwargs.get("sharex", "col")
    subplot_kwargs["sharey"] = subplot_kwargs.get("sharey", "row")
    axs = fig.subplots(nrows, ncols, **subplot_kwargs)
    if nrows == 1 and ncols == 1:
        axs = listify_axs([[axs]])
    elif nrows == 1 or ncols == 1:
        axs = listify_axs([axs])
    return fig, listify_axs(axs)


def plot(
    obj: Any,
    /,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
    *,
    plot_mode: LINE_PLOT_MODES | None = None,
    set_legend: bool = True,
    **kwargs: Unpack[PlotKwargs],
):
    """Plot a single object with customizable visualization options.

    This function provides a unified interface for plotting various types
    The function automatically handles axis creation, unit normalization,
    and label formatting.

    Parameters
    ----------
    obj :
        The object to plot. Can be a :ref:`representation <representation_types>`,
        a :ref:`data <data_types>`, or a :ref:`waveform <waveform_types>`.
        Positional-only.
    fig : matplotlib.figure.Figure, optional
        The figure to plot into. If not provided a new
        figure will be created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot into. If provided, `fig` must also be provided.
        Can not be specified when plotting multi-channel objects.
    plot_mode : {'loglog', 'semilogx', 'semilogy', 'plot', 'imshow'}, optional
        The plotting mode to use. If not specified, a default mode is selected
        based on the object type.
    set_legend : bool, default=True
        Whether to display a legend on the plot.
    **kwargs :
        Additional keyword arguments passed to matplotlib or figure creation.
        Common options include:

        - Figure kwargs: figsize, dpi, constrained_layout
        - Subplot kwargs: sharex, sharey, wspace, hspace
        - Line kwargs: color, linewidth, linestyle, label
        - Image kwargs: cmap, vmin, vmax

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.

    axs : list[list[matplotlib.axes.Axes]]
        The axes containing the plot, organized as a list of rows of axes, where
        a row is itself a list of axes.

    Raises
    ------
    TypeError
        If `obj` is not a plottable type.
    ValueError
        If the object has an unsupported shape or if `ax` is provided without `fig`.
    NotImplementedError
        If `obj` has a sparse grid or other configuration yet to be supported.

    Notes
    -----
    - Units are automatically normalized across all data in the plot for
      consistency and readability.
    - For :class:`~types.TimeSeries`, the time unit is automatically selected based on
      the duration (seconds, hours, or days).
    - For :class:`~types.FrequencySeries` and :class:`~types.Phasor`, the frequency
      unit is automatically selected based on the maximum frequency (Hz, kHz, or mHz).
    - When plotting multi-channel objects, subplots are created with shared
      axes.

    See Also
    --------
    :func:`plot_compare`
        For comparing two compatible objects.
    """
    return _dispatch_plot(
        obj, fig=fig, ax=ax, plot_mode=plot_mode, set_legend=set_legend, **kwargs
    )


def plot_compare(
    obj1: Any,
    obj2: Any,
    /,
    fig: matplotlib.figure.Figure | None = None,
    ax: matplotlib.axes.Axes | None = None,
    *,
    plot_mode: LINE_PLOT_MODES | None = None,
    showdiff: bool = False,
    **kwargs: Unpack[PlotKwargs],
):
    """Plot two objects for comparison, with optional difference view.

    This function provides a unified interface for comparing two compatible
    objects. Both objects are plotted with consistent scaling and units. Optionally,
    the difference between the two objects can be displayed in a side column.

    Parameters
    ----------
    obj1 :
        The first object to plot. Positional-only.
    obj2 :
        The second object to plot. Must be of the same type as `obj1`. Positional-only.
    fig : matplotlib.figure.Figure, optional
        The figure to plot into. If not provided, a new
        figure will be created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot into. If provided, `fig` must also be provided.
        Can not be specified when plotting multi-channel objects or
        when `showdiff=True`.
    plot_mode : {'loglog', 'semilogx', 'semilogy', 'plot', 'imshow'}, optional
        The plotting mode to use. If not specified, a default mode is selected
        based on the object type. The same mode is applied to both objects
        and the difference plot (if shown).
    showdiff : bool, default=False
        Whether to display the difference (`obj1 - obj2`).
        If True, the difference is shown in a side column. Difference display is not
        supported for :class:`~types.Phasor` objects.
    set_legend : bool, default=True
        Whether to display a legend on the plots.
    **kwargs : PlotKwargs
        Additional keyword arguments passed to matplotlib or figure creation.
        Common options include:

        - Figure kwargs: figsize, dpi, constrained_layout
        - Subplot kwargs: sharex, sharey, wspace, hspace
        - Line kwargs: color, linewidth, linestyle, label
        - Image kwargs: cmap, vmin, vmax

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plots.

    axs : list[list[matplotlib.axes.Axes]]
        The axes containing the plots, organized as a list of rows of axes, where
        a row is itself a list of axes.

    Raises
    ------
    TypeError
        If either `obj1` or `obj2` is not a plottable type.
    ValueError
        If the objects are incompatible for comparison,
        or if `ax` is provided without `fig`.
    NotImplementedError
        If the objects have a sparse grid or other configuration yet to be supported.

    See Also
    --------
    :func:`plot`
        More details on plotting behavior, unit normalization, and supported types.
    """
    return _dispatch_compare(
        obj1,
        obj2,
        fig=fig,
        ax=ax,
        plot_mode=plot_mode,
        showdiff=showdiff,
        **kwargs,
    )
