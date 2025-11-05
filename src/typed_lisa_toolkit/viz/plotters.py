"""Module for plotting tools.

We provide a set of classes to plot data and representations. The classes
implemented here are used by classes in :mod:`typed_lisa_toolkit.containers.data`
and :mod:`typed_lisa_toolkit.containers.representations` to visualize signals.
It is also possible to call methods from these classes to obtain figures and axes
to customize the plots.

.. currentmodule:: typed_lisa_toolkit.viz.plotters

Entities
--------

.. autoclass:: TSPlotter
    :members:
    :member-order: groupwise
    :inherited-members:

.. autoclass:: FSPlotter
    :members:
    :member-order: groupwise
    :inherited-members:

.. autoclass:: PhasorPlotter
    :members:
    :member-order: groupwise
    :inherited-members:

.. autoclass:: TimeFrequencyPlotter
    :members:
    :member-order: groupwise
    :inherited-members:

.. autoclass:: TSDataPlotter
    :members:
    :member-order: groupwise
    :inherited-members:

.. autoclass:: FSDataPlotter
    :members:
    :member-order: groupwise
    :inherited-members:

"""

from __future__ import annotations
import abc
import logging
import copy
from typing import TYPE_CHECKING, Self, Literal
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..containers import data, representations

logger = logging.getLogger(__name__)

figure_kwargs = [
    "dpi",
    "edgecolor",
    "facecolor",
    "figsize",
    "frameon",
    "num",
    "tight_layout",
]

add_subplot_kwargs = ["polar", "projection", "sharex", "sharey"]

legend_kwargs = [
    "bbox_to_anchor",
    "borderaxespad",
    "borderpad",
    "columnspacing",
    "edgecolor",
    "facecolor",
    "fancybox",
    "fontsize",
    "framealpha",
    "frameon",
    "handlelength",
    "handler_map",
    "handletextpad",
    "labelspacing",
    "loc",
    "markerfirst",
    "markerscale",
    "mode",
    "ncol",
    "numpoints",
    "prop",
    "reverse",
    "scatterpoints",
    "scatteryoffsets",
    "shadow",
    "title",
    "title_fontsize",
]

plot_kwargs = [
    "alpha",
    "c",
    "color",
    "label",
    "linestyle",
    "linewidth",
    "ls",
    "lw",
    "marker",
    "markersize",
    "markevery",
    "ms",
    "zorder",
    "rasterized",
]

imshow_kwargs = [
    "cmap",
    "norm",
    "aspect",
    "interpolation",
    "alpha",
    "vmin",
    "vmax",
    "origin",
    "extent",
    "filternorm",
    "filterrad",
    "resample",
    "url",
]

savefig_kwargs = [
    "transparent",
    "dpi",
    "format",
    "metadata",
    "bbox_inches",
    "pad_inches",
    "facecolor",
    "edgecolor",
    "backend",
    "orientation",
    "papertype",
    "bbox_extra_artists",
    "pil_kwargs",
]


def sieve_kwargs(keys_to_accept: list[str], **kwargs):
    """Sieve the keyword arguments to accept."""
    return {k: v for k, v in kwargs.items() if k in keys_to_accept}


class _SeriesPlotter(abc.ABC):
    series: representations._Series

    def __init__(self, series: representations._Series) -> None:
        self.series = copy.deepcopy(series)

    def _get_fig_ax(
        self, **kwargs
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Get the figure and axes."""
        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None)
        if fig is None:
            if ax is not None:
                raise ValueError("If ax is provided, fig must also be provided.")
            _fig_kwargs = sieve_kwargs(figure_kwargs, **kwargs)
            fig = plt.figure(**_fig_kwargs)
        if ax is None:
            _add_subplot_kwargs = sieve_kwargs(add_subplot_kwargs, **kwargs)
            ax = fig.add_subplot(**_add_subplot_kwargs)
        return fig, ax

    @abc.abstractmethod
    def plot(
        self,
        ax: matplotlib.axes.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes: ...

    def draw(self, **kwargs) -> matplotlib.figure.Figure:
        """Draw the series on a figure."""
        fig, ax = self._get_fig_ax(**kwargs)
        self.plot(ax, set_xlabel=True, set_ylabel=True, **kwargs)
        return fig


class TSPlotter(_SeriesPlotter):
    """Plotter for :class:`.containers.representations.TimeSeries`."""

    series: (  # pyright: ignore[reportIncompatibleVariableOverride]
        representations.TimeSeries
    )

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        time_unit: Literal["hrs", "days"] = "hrs",
        ylabel: str = "strain",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the series on the provided Axes."""
        grid = np.array(self.series.grid)
        if not np.issubdtype(grid.dtype, np.datetime64):
            if time_unit == "hrs":
                grid /= 3600
            elif time_unit == "days":
                grid /= 3600 * 24
            else:
                raise ValueError(
                    f"The time unit {time_unit} is not supported. Please use 'hrs' or 'days'."
                )
        _kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        ax.plot(grid, self.series.entries, **_kwargs)
        if set_xlabel:
            if not np.issubdtype(grid.dtype, np.datetime64):
                ax.set_xlabel(f"Time [{time_unit}]")
            else:
                ax.set_xlabel("Time")
        if set_ylabel:
            ax.set_ylabel(ylabel)
        if set_legend:
            _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
            ax.legend(**_legend_kwargs)
        return ax


class FSPlotter(_SeriesPlotter):
    """Plotter for :class:`.containers.representations.FrequencySeries`."""

    series: (  # pyright: ignore[reportIncompatibleVariableOverride]
        representations.FrequencySeries
    )

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        freq_unit: Literal["Hz", "mHz"] = "Hz",
        ylabel: str = "Strain",
        method: Literal["loglog"] | Literal["semilogx"] = "loglog",
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the series on the provided Axes."""
        _kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        series_abs = self.series.abs()
        if freq_unit == "Hz":
            grid = series_abs.grid
            grid_label = "Frequency [Hz]"
        elif freq_unit == "mHz":
            grid = np.array(series_abs.grid) * 1e3
            grid_label = "Frequency [mHz]"
        else:
            raise ValueError(
                f"The frequency unit {freq_unit} is not supported. Please use 'Hz' or 'mHz'."
            )
        if method == "loglog":
            ax.loglog(grid, series_abs.entries, **_kwargs)
        elif method == "semilogx":
            ax.semilogx(grid, series_abs.entries, **_kwargs)
        if set_xlabel:
            ax.set_xlabel(grid_label)
        if set_ylabel:
            ax.set_ylabel(ylabel)
        if set_legend:
            _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
            ax.legend(**_legend_kwargs)
        return ax

    def plot_angle(
        self,
        ax: matplotlib.axes.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the angle of the series on the provided Axes."""
        _kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        angle = self.series.angle().entries
        ax.semilogx(self.series.grid, angle, **_kwargs)
        if set_xlabel:
            ax.set_xlabel("Frequency [Hz]")
        if set_ylabel:
            ax.set_ylabel("Angle [rad]")
        if set_legend:
            _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
            ax.legend(**_legend_kwargs)
        return ax

    def draw(self, **kwargs) -> matplotlib.figure.Figure:
        """Draw the representation on a figure."""
        fig, axs = plt.subplots(2, sharex=True)
        amp_ax, phase_ax = axs[0], axs[1]
        self.plot(amp_ax, set_xlabel=False, set_ylabel=True, **kwargs)
        self.plot_angle(phase_ax, set_xlabel=True, set_ylabel=True, **kwargs)
        return fig


class PhasorPlotter(FSPlotter):
    """Plotter for :class:`.containers.representations.Phasor`."""

    series: representations.Phasor

    def __init__(self, series: representations.Phasor) -> None:
        self.series = series  # pyright: ignore[reportIncompatibleVariableOverride]

    def plot_phase(
        self,
        ax: matplotlib.axes.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the phase of the series on the provided Axes."""
        _kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        phases = self.series.phases
        ax.semilogx(self.series.grid, phases, **_kwargs)
        if set_xlabel:
            ax.set_xlabel("Frequency [Hz]")
        if set_ylabel:
            ax.set_ylabel("Phase [rad]")
        if set_legend:
            _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
            ax.legend(**_legend_kwargs)
        return ax


class TimeFrequencyPlotter:
    """Plotter for :class:`.containers.representations.TimeFrequency`."""

    def __init__(self, representation: representations.TimeFrequency) -> None:
        self.representation = representation

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        **kwargs,
    ) -> matplotlib.axes.Axes:
        """Plot the time-frequency representation on the provided Axes."""
        _kwargs = sieve_kwargs(imshow_kwargs, **kwargs)
        times = np.array(self.representation.times)
        frequencies = np.array(self.representation.frequencies)
        dT, dF = self.representation.dT, self.representation.dF

        times_extent = (
            times[0] / 3600,
            (times[-1] + dT) / 3600,
        )
        freqs_extent = (
            frequencies[0],
            frequencies[-1] + dF,
        )
        extent = times_extent + freqs_extent
        ax.imshow(
            abs(self.representation.entries),
            origin="lower",
            aspect="auto",
            extent=extent,
            **_kwargs,
        )
        if set_xlabel:
            ax.set_xlabel("Time [hrs]")
        if set_ylabel:
            ax.set_ylabel("Frequency [Hz]")
        if set_legend:
            _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
            ax.legend(**_legend_kwargs)
        return ax


class _SeriesDataPlotter(abc.ABC):
    def __init__(self, data: data._SeriesData) -> None:
        self.data = copy.deepcopy(data)

    def _draw(
        self, plotter: type[_SeriesPlotter], set_legend: bool = False, **kwargs
    ) -> matplotlib.figure.Figure:
        chn_num = len(self.data.channel_names)

        fig, axs = plt.subplots(chn_num, sharex=True)

        # If only one channel, axs is not a list
        try:
            axs[0]
        except TypeError:
            axs = [axs]

        label = kwargs.pop("label", self.data.name)

        for idx, chnname in enumerate(self.data.channel_names):
            xlabel_bool = True if idx == chn_num - 1 else False
            plotter(self.data[chnname]).plot(
                axs[idx],
                set_xlabel=xlabel_bool,
                set_ylabel=True,
                set_legend=set_legend,
                label=label,
                ylabel=f"{chnname}",
                **kwargs,
            )
        return fig

    def _compare(
        self,
        plotter: type[_SeriesPlotter],
        other: Self,
        plot_residual: bool = True,
        **kwargs,
    ) -> matplotlib.figure.Figure:
        chn_num = len(self.data.channel_names)
        ylabel_bool = kwargs.pop("set_ylabel", False)
        if plot_residual:
            fig, axs = plt.subplots(2 * chn_num, sharex=True)
        else:
            fig, axs = plt.subplots(chn_num, sharex=True)
        diff_ylabel = kwargs.pop("diff_ylabel", "difference")
        for idx, chnname in enumerate(self.data.channel_names):
            xlabel_bool = True if idx == chn_num - 1 else False
            if plot_residual:
                orig_idx = 2 * idx
                diff_idx = 2 * idx + 1
            else:
                orig_idx = idx
            plotter(self.data[chnname]).plot(
                axs[orig_idx],
                set_xlabel=False,
                set_legend=True,
                set_ylabel=ylabel_bool,
                label=self.data.name,
                ylabel=f"{chnname}",
                **kwargs,
            )
            plotter(other.data[chnname]).plot(
                axs[orig_idx],
                set_legend=True,
                set_xlabel=xlabel_bool and not plot_residual,
                label=other.data.name,
                **kwargs,
            )
            if plot_residual:
                plotter(self.data[chnname] - other.data[chnname]).plot(
                    axs[diff_idx],  # pyright: ignore[reportPossiblyUnboundVariable]
                    set_xlabel=xlabel_bool,
                    set_ylabel=ylabel_bool,
                    set_legend=False,
                    method="semilogx",
                    ylabel=f"{chnname}: {diff_ylabel}",
                    **kwargs,
                )
        return fig

    @abc.abstractmethod
    def draw(self, **kwargs) -> matplotlib.figure.Figure: ...

    @abc.abstractmethod
    def compare(self, other: Self, **kwargs) -> matplotlib.figure.Figure: ...


class TSDataPlotter(_SeriesDataPlotter):
    """Plotter for :class:`.containers.data.TSData`."""

    def draw(self, set_legend: bool = False, **kwargs) -> matplotlib.figure.Figure:
        """Draw the time series data."""
        return self._draw(plotter=TSPlotter, set_legend=set_legend, **kwargs)

    def compare(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Self, **kwargs
    ) -> matplotlib.figure.Figure:
        """Compare two time series data."""
        return self._compare(plotter=TSPlotter, other=other, **kwargs)


class FSDataPlotter(_SeriesDataPlotter):
    """Plotter for :class:`.containers.data.FSData`."""

    def _draw_angle(
        self, set_legend: bool = False, **kwargs
    ) -> matplotlib.figure.Figure:
        chn_num = len(self.data.channel_names)
        fig, axs = plt.subplots(2 * chn_num, sharex=True)
        for idx, chnname in enumerate(self.data.channel_names):
            xlabel_bool = True if idx == chn_num - 1 else False
            amp_idx = 2 * idx
            phase_idx = 2 * idx + 1
            plotter = PhasorPlotter(self.data[chnname])
            plotter.plot(
                axs[amp_idx],
                set_xlabel=False,
                set_ylabel=True,
                set_legend=set_legend,
                label=self.data.name,
                ylabel=f"{chnname} Amplitude",
                **kwargs,
            )
            plotter.plot_angle(
                axs[phase_idx],
                set_xlabel=xlabel_bool,
                set_ylabel=True,
                set_legend=False,
                label=self.data.name,
                **kwargs,
            )
        return fig

    def _compare_angle(self, other: Self, **kwargs) -> matplotlib.figure.Figure:
        chn_num = len(self.data.channel_names)
        fig, axs = plt.subplots(4 * chn_num, sharex=True)
        for idx, chnname in enumerate(self.data.channel_names):
            xlabel_bool = True if idx == chn_num - 1 else False
            orig_amp_idx = 4 * idx
            orig_phase_idx = 4 * idx + 1
            diff_amp_idx = 4 * idx + 2
            diff_phase_idx = 4 * idx + 3
            self_plotter = PhasorPlotter(self.data[chnname])
            other_plotter = PhasorPlotter(other.data[chnname])
            diff_plotter = PhasorPlotter(self.data[chnname] - other.data[chnname])
            self_plotter.plot(
                axs[orig_amp_idx],
                set_xlabel=False,
                set_ylabel=True,
                set_legend=True,
                label=self.data.name,
                ylabel=f"{chnname} Amplitude",
                **kwargs,
            )
            other_plotter.plot(
                axs[orig_amp_idx],
                label=other.data.name,
                **kwargs,
            )
            self_plotter.plot(
                axs[orig_phase_idx],
                set_xlabel=False,
                set_ylabel=True,
                set_legend=True,
                label=self.data.name,
                ylabel=f"{chnname} Phase [rad]",
                **kwargs,
            )
            other_plotter.plot(
                axs[orig_phase_idx],
                label=other.data.name,
                **kwargs,
            )
            diff_plotter.plot(
                axs[diff_amp_idx],
                set_xlabel=False,
                set_ylabel=True,
                set_legend=True,
                label=f"{chnname}: {self.data.name} - {other.data.name}",
                **kwargs,
            )
            diff_plotter.plot(
                axs[diff_phase_idx],
                set_xlabel=xlabel_bool,
                set_ylabel=True,
                set_legend=True,
                label=f"{chnname}: {self.data.name} - {other.data.name}",
                **kwargs,
            )
        return fig

    def draw(
        self, set_legend: bool = False, angle: bool = False, **kwargs
    ) -> matplotlib.figure.Figure:
        """Draw the frequency series data."""
        if not angle:
            return self._draw(plotter=FSPlotter, set_legend=set_legend, **kwargs)
        return self._draw_angle(set_legend=set_legend, **kwargs)

    def compare(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Self, angle: bool = False, **kwargs
    ) -> matplotlib.figure.Figure:
        """Compare two frequency series data."""
        if not angle:
            return self._compare(plotter=FSPlotter, other=other, **kwargs)
        return self._compare_angle(other=other, **kwargs)


class TFDataPlotter:
    """Plotter for :class:`.containers.data.TFData`."""

    def __init__(self, data: data.TFData) -> None:
        self.data = copy.deepcopy(data)

    def draw(self, set_legend: bool = False, **kwargs) -> matplotlib.figure.Figure:
        """Draw the time-frequency data."""
        fig, axs = plt.subplots(len(self.data.channel_names), sharex=True)
        # If only one channel, axs is not a list
        try:
            axs[0]
        except TypeError:
            axs = [axs]
        for idx, chnname in enumerate(self.data.channel_names):
            xlabel_bool = True if idx == len(self.data.channel_names) - 1 else False
            TimeFrequencyPlotter(self.data[chnname]).plot(
                axs[idx],
                set_xlabel=xlabel_bool,
                set_ylabel=True,
                set_legend=set_legend,
                **kwargs,
            )
        return fig
