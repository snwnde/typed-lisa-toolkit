"""Plot option definitions and parameter handling."""

from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Literal,
    TypedDict,
    cast,
)

import matplotlib.colors
import matplotlib.figure
import matplotlib.typing
import numpy.typing as npt


class FigureKwargs(TypedDict, total=False):
    dpi: float | None
    edgecolor: matplotlib.typing.ColorType | None
    facecolor: matplotlib.typing.ColorType | None
    figsize: npt.ArrayLike | None
    frameon: bool
    num: int | str | matplotlib.figure.Figure | matplotlib.figure.SubFigure | None
    tight_layout: object


class AddSubplotKwargs(TypedDict, total=False):
    polar: object
    projection: object
    sharex: object
    sharey: object


class SubplotKwargs(TypedDict, total=False):
    sharex: bool | Literal["none", "all", "row", "col"]
    sharey: bool | Literal["none", "all", "row", "col"]
    squeeze: bool
    width_ratios: Sequence[float] | None
    height_ratios: Sequence[float] | None
    subplot_kw: dict[str, Any] | None
    gridspec_kw: dict[str, Any] | None


class LegendKwargs(TypedDict, total=False):
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


class LineKwargs(TypedDict, total=False):
    alpha: npt.ArrayLike | float | None
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


class ImshowKwargs(TypedDict, total=False):
    cmap: str | matplotlib.colors.Colormap | None
    norm: str | matplotlib.colors.Normalize | None
    aspect: float | Literal["equal", "auto"] | None
    interpolation: str | None
    alpha: npt.ArrayLike | float | None
    vmin: float | None
    vmax: float | None
    origin: Literal["upper", "lower"] | None
    extent: tuple[float, float, float, float] | None
    filternorm: bool
    filterrad: float
    resample: bool | None
    url: str | None


class PlotKwargs(
    FigureKwargs,
    AddSubplotKwargs,
    LegendKwargs,
    LineKwargs,
    ImshowKwargs,
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
        FigureKwargs,
        AddSubplotKwargs,
        SubplotKwargs,
        LegendKwargs,
        LineKwargs,
        ImshowKwargs,
    )
]:
    def __init__(self, kwargs_cls: type[T], /) -> None:
        self.keys: frozenset[str] = kwargs_cls.__optional_keys__


FIGURE_KWARGS = _Keys(FigureKwargs)
ADD_SUBPLOT_KWARGS = _Keys(AddSubplotKwargs)
SUBPLOT_KWARGS = _Keys(SubplotKwargs)
LEGEND_KWARGS = _Keys(LegendKwargs)
LINE_KWARGS = _Keys(LineKwargs)
IMSHOW_KWARGS = _Keys(ImshowKwargs)

LINE_PLOT_MODES = Literal["loglog", "semilogx", "plot", "semilogy"]


def sieve_kwargs[
    T: (
        FigureKwargs,
        AddSubplotKwargs,
        SubplotKwargs,
        LegendKwargs,
        LineKwargs,
        ImshowKwargs,
    )
](
    to_accept: _Keys[T],
    /,
    kwargs: Mapping[str, Any],
) -> T:
    """Filter keyword arguments to the accepted subset."""
    _dict = {key: value for key, value in kwargs.items() if key in to_accept.keys}
    return cast("T", cast("object", _dict))
