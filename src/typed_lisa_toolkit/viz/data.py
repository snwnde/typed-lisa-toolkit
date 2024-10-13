"""Plotting tools for Data."""

from __future__ import annotations
import abc
import logging
import copy
from typing import TYPE_CHECKING, Literal, Self
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..containers import data as ldcdata

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


class DataPlotter:
    """Base plotter for Data."""

    number_of_subplots: int

    def __init__(self, data: ldcdata._SeriesData) -> None:
        self.data = copy.deepcopy(data)

    @classmethod
    def _render_fig_ax(cls, fig, ax, **kwargs):
        """Render figure and axis if not provided."""
        if fig is None:
            fig_kwargs = sieve_kwargs(figure_kwargs, **kwargs)
            fig = plt.figure(**fig_kwargs)
        if ax is None:
            _add_subplot_kwargs = sieve_kwargs(add_subplot_kwargs, **kwargs)
            ax = fig.add_subplot(**_add_subplot_kwargs)
        return fig, ax

    @classmethod
    @abc.abstractmethod
    def _get_xlabel(cls, **kwargs):
        """Get the label for the x-axis."""
        pass

    @abc.abstractmethod
    def draw(self, **kwargs):
        """Draw the data in a figure, possibly with multiple subplots."""
        pass

    def _render_plot_labels(self, kwargs):
        """Render plot labels."""
        labels = kwargs.pop("label", "auto")
        if labels == "auto":
            labels = list(self.data.channel_names)
        return labels, kwargs

    @classmethod
    def _plot_difference(cls, source_ax, ax, label=None, set_xlabel=False, **kwargs):
        """Plot the difference between two data sets."""
        x_axis = source_ax.lines[0].get_data()[0]
        y_axis = source_ax.lines[0].get_data()[1] - source_ax.lines[1].get_data()[1]
        _plot_kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        ax.plot(x_axis, y_axis, label=label, **_plot_kwargs)
        if set_xlabel:
            ax.set_xlabel(cls._get_xlabel(**kwargs))
        return ax

    def pick_channel(self, chnname: str) -> Self:
        """Get the channel."""
        return type(self)(self.data.pick(chnname))

    def compare(
        self,
        other: ldcdata._SeriesData,
        savefig=None,
        figsize=None,
        sharex: bool | Literal["none", "all", "row", "col"] = "all",
        h_pad=None,
        fig=None,
        **kwargs,
    ):
        """Compare the data with another data."""
        other_ = self.__class__(other)
        if fig is not None:
            raise NotImplementedError(
                "Currently we do not support plot on an existing figure."
            )
        # If both data agree in axis, the difference is also plotted.
        if np.array_equal(self.data.grid, other_.data.grid):
            diff = self.__class__(self.data - other_.data)
            total_number_of_subplots = (
                2 * self.number_of_subplots * len(self.data.channel_names)
            )
        else:
            diff = None
            total_number_of_subplots = self.number_of_subplots * len(
                self.data.channel_names
            )

        fig, axs = plt.subplots(
            total_number_of_subplots, figsize=figsize, sharex=sharex
        )

        fig.tight_layout(h_pad=h_pad)
        labels = kwargs.pop("label", "auto")

        def _render_manual_labels(idx, labels, npergroup):
            """Render manual labels."""
            if len(labels) == npergroup:
                labels_ = labels
            elif len(labels) == npergroup * len(self.data.channel_names):
                labels_ = labels[idx * npergroup : (idx + 1) * npergroup]
            else:
                raise ValueError("The number of labels do not match the representation.")
            return labels_

        if diff is not None:
            _step_size = 2 * self.number_of_subplots
            ax_groups = [
                axs[i * _step_size : (i + 1) * _step_size]
                for i, _ in enumerate(self.data.channel_names)
            ]

            for idx_, axs_ in enumerate(ax_groups):
                chnname = self.data.channel_names[idx_]
                # Determine the labels
                if labels == "auto":
                    labels_ = [
                        f"{chnname}1",
                        f"{chnname}2",
                        f"Diff {chnname}1-{chnname}2",
                    ]
                else:
                    labels_ = _render_manual_labels(idx_, labels, 3)

                fig, axs_[0::2] = self.pick_channel(chnname).draw(
                    fig=fig,
                    axs=axs_[0::2],
                    set_xlabel=False,
                    label=labels_[0],
                    **kwargs,
                )  # type: ignore
                fig, axs_[0::2] = other_.pick_channel(chnname).draw(
                    fig=fig,
                    axs=axs_[0::2],
                    set_xlabel=False,
                    label=labels_[1],
                    **kwargs,
                )  # type: ignore

                for id, ax in enumerate(axs_[1::2]):
                    # Set xlabel at the last ax
                    set_xlabel = (
                        True
                        if idx_ == len(ax_groups) - 1 and id == len(axs_[1::2]) - 1
                        else False
                    )
                    ax = self._plot_difference(
                        axs_[0::2][id],
                        ax,
                        label=labels_[2],
                        set_xlabel=set_xlabel,
                        xlabel=self._get_xlabel(**kwargs),
                        **kwargs,
                    )

        else:
            _step_size = self.number_of_subplots
            ax_groups = [
                axs[i * _step_size : (i + 1) * _step_size]
                for i, _ in enumerate(self.data.channel_names)
            ]

            for idx_, axs_ in enumerate(ax_groups):
                chnname = self.data.channel_names[idx_]
                # Set xlabel at the last ax
                set_xlabel = True if idx_ == len(ax_groups) - 1 else False
                # Determine the labels
                if labels == "auto":
                    labels_ = [f"{chnname}1", f"{chnname}2"]
                else:
                    labels_ = _render_manual_labels(idx_, labels, 2)

                fig, axs_ = self.pick_channel(chnname).draw(  # type: ignore
                    fig=fig, axs=axs_, set_xlabel=set_xlabel, label=labels_[0], **kwargs
                )
                fig, axs_ = other_.pick_channel(chnname).draw(  # type: ignore
                    fig=fig,
                    axs=axs_,
                    set_xlabel=set_xlabel,
                    label=labels_[1],
                    linestyle="--",
                    **kwargs,
                )

        # Set legend
        legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
        for ax in axs:
            ax.legend(**legend_kwargs)

        if savefig is not None:
            _savefig_kwargs = sieve_kwargs(savefig_kwargs, **kwargs)
            fig.savefig(savefig, **_savefig_kwargs)
        return fig, axs


class TimeSeriesPlotter(DataPlotter):
    """Plotting tools for TSData."""

    data: ldcdata.TSData
    number_of_subplots = 1

    @classmethod
    def _render_time_grid(cls, time_grid, time_unit):
        """Render the time grid to the given unit."""
        time_grid = copy.deepcopy(time_grid)
        # Unless time_grid is np.datetime64, convert to time_unit
        if not np.issubdtype(time_grid.dtype, np.datetime64):
            if time_unit == "hrs":
                time_grid /= 3600
            elif time_unit == "days":
                time_grid /= 3600 * 24
            else:
                raise ValueError(
                    f"The time unit {time_unit} is not supported. Please use 'hrs' or 'days'."
                )
            return time_grid
        else:
            return time_grid

    @classmethod
    def _get_xlabel(cls, time_unit="hrs", **kwargs):
        """Get the label for the x-axis."""
        del kwargs
        return f"Time [{time_unit}]"

    def plot(
        self,
        fig=None,
        ax=None,
        time_unit="hrs",
        set_xlabel=True,
        **kwargs,
    ):
        """Plot the time series data."""
        # Render fig and ax
        fig, ax = self._render_fig_ax(fig, ax, **kwargs)
        # Render time grid
        time_grid = self._render_time_grid(self.data.time, time_unit=time_unit)
        # Set x label
        if set_xlabel:
            time_label = self._get_xlabel(time_unit=time_unit)
            ax.set_xlabel(time_label)
        # Plot
        _plot_kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        labels, plot_kwargs = self._render_plot_labels(_plot_kwargs)

        for idx, chname in enumerate(self.data.channel_names):
            label = labels if labels is None or isinstance(labels, str) else labels[idx]
            ax.plot(time_grid, self.data[chname].signal, label=label, **plot_kwargs)

        # Set legend
        _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
        ax.legend(**_legend_kwargs)
        return fig, ax

    def draw(
        self,
        savefig=None,
        fig=None,
        axs=None,
        time_unit="hrs",
        **kwargs,
    ):
        """Draw the time series data."""
        try:
            ax = axs[0]  # type: ignore
        except TypeError:
            ax = axs
        if fig is None and ax is None:
            fig, ax = plt.subplots(self.number_of_subplots)
        fig, ax = self.plot(fig=fig, ax=ax, time_unit=time_unit, **kwargs)
        # Save fig
        if savefig is not None:
            _savefig_kwargs = sieve_kwargs(savefig_kwargs, **kwargs)
            fig.savefig(savefig, **_savefig_kwargs)
        return fig, axs


class FrequencySeriesPlotter(DataPlotter):
    """Plotting tools for FSData."""

    data: ldcdata.FSData
    number_of_subplots = 2

    @classmethod
    def _get_xlabel(cls, **kwargs):
        """Get the label for the x-axis."""
        del kwargs
        return "Frequency [Hz]"

    def plot_amplitude(self, labels, ax, ylabel="Amplitude", **plot_kwargs):
        """Plot the amplitude of the frequency series data."""
        if ax is None:
            _, ax = plt.subplots()
        # Plot amplitude
        for idx, chname in enumerate(self.data.channel_names):
            label = labels if labels is None or isinstance(labels, str) else labels[idx]
            ax.set_ylabel(ylabel)
            ax.loglog(
                self.data.frequencies,
                np.abs(self.data[chname].signal),
                label=label,
                **plot_kwargs,
            )
        return ax

    def _plot_phase(self, labels, ax, unwrap, **plot_kwargs):
        """Plot the phase of the frequency series data."""
        for idx, chname in enumerate(self.data.channel_names):
            label = labels if labels is None or isinstance(labels, str) else labels[idx]
            ax.set_ylabel("Phase [rad]")
            phase = np.angle(self.data[chname].signal)
            if unwrap:
                phase_ = np.unwrap(phase, np.pi)
            else:
                phase_ = phase
            ax.semilogx(
                self.data.frequencies,
                phase_,
                label=label,
                **plot_kwargs,
            )

    def plot_phase(self, labels, ax, **plot_kwargs):
        """Plot the unwrapped phase of the frequency series data."""
        self._plot_phase(labels, ax, unwrap=True, **plot_kwargs)

    def plot_nonunwrap_phase(self, labels, ax, **plot_kwargs):
        """Plot the non-unwrapped phase of the frequency series data."""
        self._plot_phase(labels, ax, unwrap=False, **plot_kwargs)

    def plot_real(self, labels, ax, **plot_kwargs):
        """Plot the real part of the frequency series data."""
        for idx, chname in enumerate(self.data.channel_names):
            label = labels if labels is None or isinstance(labels, str) else labels[idx]
            ax.set_ylabel("Real part")
            ax.set_yscale("symlog")
            ax.semilogx(
                self.data.frequencies,
                np.real(self.data[chname].signal),
                label=label,
                **plot_kwargs,
            )

    def plot_imag(self, labels, ax, **plot_kwargs):
        """Plot the imaginary part of the frequency series data."""
        for idx, chname in enumerate(self.data.channel_names):
            label = labels if labels is None or isinstance(labels, str) else labels[idx]
            ax.set_ylabel("Imaginary part")
            ax.set_yscale("symlog")
            ax.semilogx(
                self.data.frequencies,
                np.imag(self.data[chname].signal),
                label=label,
                **plot_kwargs,
            )

    def plot(
        self,
        what_to_plot: str,
        fig=None,
        ax=None,
        set_xlabel=True,
        **kwargs,
    ):
        """Plot the specified aspect of the frequency series data."""
        # Render fig and ax
        fig, ax = self._render_fig_ax(fig, ax, **kwargs)
        # Set x label
        if set_xlabel:
            ax.set_xlabel(self._get_xlabel(**kwargs))
        # Plot
        _plot_kwargs = sieve_kwargs(plot_kwargs, **kwargs)
        labels, _plot_kwargs = self._render_plot_labels(_plot_kwargs)
        getattr(self, f"plot_{what_to_plot}")(labels, ax, **_plot_kwargs)
        # Set legend
        _legend_kwargs = sieve_kwargs(legend_kwargs, **kwargs)
        ax.legend(**_legend_kwargs)
        return fig, ax

    def draw(
        self,
        plot_real_imag=False,
        savefig=None,
        fig=None,
        axs=None,
        h_pad=None,
        **kwargs,
    ):
        """Draw the frequency series data."""
        if fig is None or axs is None:
            fig, axs = plt.subplots(self.number_of_subplots, sharex=True)
        fig.tight_layout(h_pad=h_pad)
        set_xlabel = kwargs.pop("set_xlabel", True)
        unwrap = kwargs.pop("unwrap", True)
        phase_str = "phase" if unwrap else "nonunwrap_phase"
        if not plot_real_imag:
            fig, axs[0] = self.plot(
                "amplitude", fig=fig, ax=axs[0], set_xlabel=False, **kwargs
            )
            fig, axs[1] = self.plot(
                phase_str, fig=fig, ax=axs[1], set_xlabel=set_xlabel, **kwargs
            )
        else:
            fig, axs[0] = self.plot(
                "real", fig=fig, ax=axs[0], set_xlabel=False, **kwargs
            )
            fig, axs[1] = self.plot(
                "imag", fig=fig, ax=axs[1], set_xlabel=set_xlabel, **kwargs
            )
        # Save fig
        if savefig is not None:
            _savefig_kwargs = sieve_kwargs(savefig_kwargs, **kwargs)
            fig.savefig(savefig, **_savefig_kwargs)
        return fig, axs