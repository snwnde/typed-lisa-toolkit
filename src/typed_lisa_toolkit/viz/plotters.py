"""Plotting tools for Data."""

from __future__ import annotations
import abc
import logging
import copy
from typing import TYPE_CHECKING, Self, Literal
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..containers import data
    from ..containers import representations

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
    "ncols",
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

    def _get_fig_ax(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:
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
        ax: plt.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        **kwargs,
    ) -> plt.Axes: ...

    def draw(self, **kwargs) -> plt.Figure:
        """Draw the series on a figure."""
        fig, ax = self._get_fig_ax(**kwargs)
        self.plot(ax, set_xlabel=True, set_ylabel=True, **kwargs)
        return fig


class TSPlotter(_SeriesPlotter):
    """Plotter for :class:`..containers.representations.TimeSeries`."""

    series: representations.TimeSeries

    def plot(
        self,
        ax: plt.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        time_unit: str = "hrs",
        ylabel: str = "strain",
        **kwargs,
    ) -> plt.Axes:
        """Plot the series on the provided Axes."""
        grid = self.series.grid.copy()
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
    """Plotter for :class:`..containers.representations.FrequencySeries`."""

    series: representations.FrequencySeries

    def plot(
        self,
        ax: plt.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        ylabel: str = "Strain",
        method: Literal["loglog"] | Literal["semilogx"] = "loglog",
        **kwargs,
    ) -> plt.Axes:
        """Plot the series on the provided Axes."""
        _kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        series_abs = self.series.abs()
        if method == "loglog":
            ax.loglog(series_abs.grid, series_abs.entries, **_kwargs)
        elif method == "semilogx":
            ax.semilogx(series_abs.grid, series_abs.entries, **_kwargs)
        if set_xlabel:
            ax.set_xlabel("Frequency [Hz]")
        if set_ylabel:
            ax.set_ylabel(ylabel)
        if set_legend:
            _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
            ax.legend(**_legend_kwargs)
        return ax


class PhasorPlotter(FSPlotter):
    """Plotter for :class:`..containers.representations.Phasor`."""

    series: representations.Phasor

    def __init__(self, series: representations.FrequencySeries) -> None:
        self.series = series.to_phasor()

    def plot_phase(
        self,
        ax: plt.Axes,
        set_xlabel: bool = False,
        set_ylabel: bool = False,
        set_legend: bool = False,
        unwrap: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """Plot the phase of the series on the provided Axes."""
        _kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        _phases = self.series.phases
        if unwrap:
            _phases = np.unwrap(_phases, np.pi)
        ax.semilogx(self.series.grid, _phases, **_kwargs)
        if set_xlabel:
            ax.set_xlabel("Frequency [Hz]")
        if set_ylabel:
            ax.set_ylabel("Phase [rad]")
        if set_legend:
            _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
            ax.legend(**_legend_kwargs)
        return ax

    def draw(self, **kwargs) -> plt.Figure:
        """Draw the phasor representation on a figure."""
        fig, axs = plt.subplots(2, sharex=True)
        amp_ax, phase_ax = axs[0], axs[1]
        self.plot(amp_ax, set_xlabel=False, set_ylabel=True, **kwargs)
        self.plot_phase(phase_ax, set_xlabel=True, set_ylabel=True, **kwargs)
        return fig


class _SeriesDataPlotter(abc.ABC):
    def __init__(self, data: data._SeriesData) -> None:
        self.data = copy.deepcopy(data)

    def _draw(
        self, plotter: type[_SeriesPlotter], set_legend: bool = False, **kwargs
    ) -> plt.Figure:
        chn_num = len(self.data.channel_names)
        fig, axs = plt.subplots(chn_num, sharex=True)
        for idx, chnname in enumerate(self.data.channel_names):
            xlabel_bool = True if idx == chn_num - 1 else False
            plotter(self.data[chnname]).plot(
                axs[idx],
                set_xlabel=xlabel_bool,
                set_ylabel=True,
                set_legend=set_legend,
                label=self.data.name,
                ylabel=f"{chnname}",
                **kwargs,
            )
        return fig

    def _compare(
        self, plotter: type[_SeriesPlotter], other: Self, **kwargs
    ) -> plt.Figure:
        chn_num = len(self.data.channel_names)
        fig, axs = plt.subplots(2 * chn_num, sharex=True)
        for idx, chnname in enumerate(self.data.channel_names):
            xlabel_bool = True if idx == chn_num - 1 else False
            orig_idx = 2 * idx
            diff_idx = 2 * idx + 1
            plotter(self.data[chnname]).plot(
                axs[orig_idx],
                set_xlabel=False,
                set_ylabel=True,
                set_legend=True,
                label=self.data.name,
                ylabel=f"{chnname}",
                **kwargs,
            )
            plotter(other.data[chnname]).plot(
                axs[orig_idx],
                label=other.data.name,
                **kwargs,
            )
            plotter(self.data[chnname] - other.data[chnname]).plot(
                axs[diff_idx],
                set_xlabel=xlabel_bool,
                set_ylabel=True,
                set_legend=False,
                method="semilogx",
                ylabel=f"{chnname}: {self.data.name} - {other.data.name}",
                **kwargs,
            )
        return fig

    @abc.abstractmethod
    def draw(self, **kwargs) -> plt.Figure: ...

    @abc.abstractmethod
    def compare(self, other: Self, **kwargs) -> plt.Figure: ...


class TSDataPlotter(_SeriesDataPlotter):
    """Plotter for :class:`..containers.data.TSData`."""

    def draw(self, set_legend: bool = False, **kwargs) -> plt.Figure:
        """Draw the time series data."""
        return self._draw(plotter=TSPlotter, set_legend=set_legend, **kwargs)

    def compare(self, other: Self, **kwargs) -> plt.Figure:
        """Compare two time series data."""
        return self._compare(plotter=TSPlotter, other=other, **kwargs)


class FSDataPlotter(_SeriesDataPlotter):
    """Plotter for :class:`..containers.data.FSData`."""

    def _draw_phasor(self, set_legend: bool = False, **kwargs) -> plt.Figure:
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
            plotter.plot_phase(
                axs[phase_idx],
                set_xlabel=xlabel_bool,
                set_ylabel=True,
                set_legend=False,
                label=self.data.name,
                **kwargs,
            )
        return fig

    def _compare_phasor(self, other: Self, **kwargs) -> plt.Figure:
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
        self, set_legend: bool = False, phasor: bool = False, **kwargs
    ) -> plt.Figure:
        """Draw the frequency series data."""
        if not phasor:
            return self._draw(plotter=FSPlotter, set_legend=set_legend, **kwargs)
        return self._draw_phasor(set_legend=set_legend, **kwargs)

    def compare(self, other: Self, phasor: bool = False, **kwargs) -> plt.Figure:
        """Compare two frequency series data."""
        if not phasor:
            return self._compare(plotter=FSPlotter, other=other, **kwargs)
        return self._compare_phasor(other=other, **kwargs)
