import matplotlib.pyplot as plt

import typed_lisa_toolkit as tlt
from typed_lisa_toolkit.types import (
    STFT,
    AnyArray,
    Axis,
    Grid2DSparse,
    Linspace,
    STFTData,
)


def test_plot_tsdata(tsdata: tlt.types.TSData):
    _ = tlt.plot(tsdata.set_name("Test"))
    _ = tlt.plot(tsdata.pick("X"), set_legend=False)
    plt.close("all")


def test_plot_compare_tsdata(tsdata: tlt.types.TSData):
    _ = tlt.plot_compare(tsdata.set_name("Test 1"), tsdata.set_name("Test 2"))
    _ = tlt.plot_compare(tsdata.pick("X"), tsdata.pick("X"), set_legend=False)
    _ = tlt.plot_compare(
        tsdata.pick("X"), tsdata.pick("X"), set_legend=False, showdiff=True
    )
    plt.close("all")


def test_plot_time_series(tsdata: tlt.types.TSData):
    _ = tlt.plot(tsdata["X"], set_legend=False)
    _ = tlt.plot_compare(tsdata["X"], tsdata["Y"], set_legend=False)
    _ = tlt.plot_compare(tsdata["X"], tsdata["Y"], set_legend=False, showdiff=True)
    plt.close("all")


def test_plot_uni_uni_sparse_stft(
    uni_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    _ = tlt.plot(uni_uni_sparse_stft)
    plt.close("all")


def test_plot_compare_uni_uni_sparse_stft(
    uni_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    _ = tlt.plot_compare(uni_uni_sparse_stft, uni_uni_sparse_stft)
    plt.close("all")


def test_plot_stftdata_sparse(
    stftdata_sparse: STFTData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    _ = tlt.plot(stftdata_sparse)
    plt.close("all")
