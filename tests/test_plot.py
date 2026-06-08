import typed_lisa_toolkit as tlt


def test_plot_tsdata(tsdata: tlt.types.TSData):
    _ = tlt.plot(tsdata.set_name("Test"))
    _ = tlt.plot(tsdata.pick("X"), set_legend=False)


def test_plot_compare_tsdata(tsdata: tlt.types.TSData):
    _ = tlt.plot_compare(tsdata.set_name("Test 1"), tsdata.set_name("Test 2"))
    _ = tlt.plot_compare(tsdata.pick("X"), tsdata.pick("X"), set_legend=False)
    _ = tlt.plot_compare(
        tsdata.pick("X"), tsdata.pick("X"), set_legend=False, showdiff=True
    )


def test_plot_time_series(tsdata: tlt.types.TSData):
    _ = tlt.plot(tsdata["X"], set_legend=False)
    _ = tlt.plot_compare(tsdata["X"], tsdata["Y"], set_legend=False)
    _ = tlt.plot_compare(tsdata["X"], tsdata["Y"], set_legend=False, showdiff=True)
