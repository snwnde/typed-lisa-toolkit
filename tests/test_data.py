# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportCallIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false

import pathlib

import h5py
import pytest

from typed_lisa_toolkit import (
    fsdata,
    load_data,
    stftdata,
    tsdata,
    wdmdata,
)
from typed_lisa_toolkit.types import (
    Array,
    Axis,
    FrequencySeries,
    FSData,
    Grid2DCartesian,
    Grid2DSparse,
    Linspace,
    STFTData,
    TSData,
    WDMData,
)

## Data


def test_tsdata(tsdata: TSData, long_time_grid1d: tuple[Axis[Array]]):
    with pytest.raises(ValueError, match="num must be"):
        tsdata.get_subset(interval=(0.1, 0.5))
    subset = tsdata.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.get_kernel().shape == (1, 3, 1, 1, len(subset.grid[0]))
    subset = tsdata.get_subset(slice=slice(0, 2))
    assert len(subset.grid[0]) == 2
    assert subset.get_kernel().shape == (1, 3, 1, 1, len(subset.grid[0]))
    embed = tsdata.get_embedded(embedding_grid=long_time_grid1d)
    assert embed.grid == long_time_grid1d
    embed = tsdata.get_embedded(
        embedding_grid=long_time_grid1d, known_slices=(slice(0, 6),)
    )
    assert embed.grid == long_time_grid1d
    with pytest.warns(DeprecationWarning, match="to_fsdata"):
        _ = tsdata.to_fsdata()
    with pytest.warns(DeprecationWarning, match="to_fsdata"):
        _ = tsdata.to_fsdata(keep_times=False)


def test_fsdata(
    fsdata: FSData, long_freq_grid1d: tuple[Axis[Array]], uni_ary_time_axis: Axis[Array]
):
    with pytest.raises(ValueError, match="num must be"):
        fsdata.get_subset(interval=(0.1, 0.5))
    subset = fsdata.get_subset(interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 2
    assert subset.get_kernel().shape == (1, 3, 1, 1, len(subset.grid[0]))
    subset = fsdata.get_subset(slice=slice(2, 7))
    assert len(subset.grid[0]) == 1
    assert subset.get_kernel().shape == (1, 3, 1, 1, len(subset.grid[0]))
    embed = fsdata.get_embedded(embedding_grid=long_freq_grid1d)
    assert embed.grid == long_freq_grid1d
    embed = fsdata.get_embedded(
        embedding_grid=long_freq_grid1d, known_slices=(slice(0, 3),)
    )
    assert embed.grid == long_freq_grid1d
    with pytest.warns(DeprecationWarning, match="to_tsdata"):
        _ = fsdata.to_tsdata(uni_ary_time_axis)


def test_stftdata_cartesian(
    stftdata_cartesian: STFTData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        stftdata_cartesian.get_subset(time_interval=(0.1, 0.5))
    subset = stftdata_cartesian.get_subset(time_interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid[0]),
        len(subset.grid[1]),
    )
    subset = stftdata_cartesian.get_subset(slices=(slice(None), slice(0, 2)))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid[0]),
        len(subset.grid[1]),
    )


def test_stftdata_sparse(
    stftdata_sparse: STFTData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        stftdata_sparse.get_subset(time_interval=(0.1, 0.5))
    subset = stftdata_sparse.get_subset(time_interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid.indices),
    )
    subset = stftdata_sparse.get_subset(slices=(slice(None), slice(0, 2)))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid.indices),
    )


def test_wdmdata_cartesian(
    wdmdata_cartesian: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        wdmdata_cartesian.get_subset(time_interval=(0.1, 0.5))
    subset = wdmdata_cartesian.get_subset(time_interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid[0]),
        len(subset.grid[1]),
    )
    subset = wdmdata_cartesian.get_subset(slices=(slice(None), slice(0, 2)))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid[0]),
        len(subset.grid[1]),
    )


def test_wdmdata_sparse(
    wdmdata_sparse: WDMData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    with pytest.raises(ValueError, match="num must be"):
        wdmdata_sparse.get_subset(time_interval=(0.1, 0.5))
    subset = wdmdata_sparse.get_subset(time_interval=(1.0, 2.5))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid.indices),
    )
    subset = wdmdata_sparse.get_subset(slices=(slice(None), slice(0, 2)))
    assert len(subset.grid[0]) == 3
    assert subset.get_kernel().shape == (
        1,
        3,
        1,
        1,
        len(subset.grid.indices),
    )


def test_timed_fsdata_roundtrip(
    fsdata: FSData,
    lin_time_axis: Axis[Linspace],
    tmp_path,
):
    timed = fsdata.set_times(lin_time_axis)
    assert timed.kind == "timed"
    assert timed.dt == lin_time_axis.ax.step

    subset = timed.get_subset(slice=slice(0, 2))
    assert subset.times == timed.times

    embedded = timed.get_embedded(embedding_grid=timed.grid)
    assert embedded.times == timed.times

    dropped = timed.drop_times()
    assert isinstance(dropped, FSData)

    file_path = tmp_path / "timed_fsdata_roundtrip.h5"
    timed.save(file_path)
    with pytest.raises(TypeError, match="unexpected keyword argument 'times'"):
        _ = load_data(file_path, domain="frequency", kind="timed")


def test_load_data_validation(tsdata: TSData, tmp_path):
    file_path = tmp_path / "tsdata_load_validation.h5"
    tsdata.save(file_path)

    with pytest.warns(FutureWarning, match="will be required"):
        inferred = load_data(file_path, kind=None)
    assert isinstance(inferred, TSData)

    with pytest.raises(ValueError, match="domain"):
        load_data(file_path, domain="frequency", kind=None)

    with pytest.raises(ValueError, match="kind"):
        load_data(file_path, domain="time", kind="timed")

    with pytest.raises(ValueError, match="Sparse grid is not supported"):
        load_data(file_path, domain="time", kind=None, sparse=True)


def test_load_data_rejects_unknown_grid_format(tmp_path):
    file_path = tmp_path / "unknown_grid_format.h5"
    with h5py.File(file_path, "w") as f:
        f.attrs["domain"] = "time"
        f.attrs["kind"] = "None"
        f.attrs["channels"] = ("X",)
        data_group = f.create_group("data")
        grid_group = data_group.create_group("grid")
        grid_group.attrs["dim"] = 99
        data_group.create_dataset("entries", data=[[[[[0.0]]]]])

    with pytest.raises(ValueError, match="Unknown grid serialization format"):
        load_data(file_path, domain="time", kind=None)


def test_tsdata_factory_validation(
    lin_time_series,
    lin_time_axis: Axis[Linspace],
):
    with pytest.raises(
        ValueError, match="Cannot specify `times`, `entries`, or `channels`"
    ):
        tsdata({"X": lin_time_series}, times=lin_time_axis)

    with pytest.raises(
        ValueError, match="Must specify `times`, `entries`, and `channels`"
    ):
        tsdata()


def test_data_mapping_shape_validation(lin_freq_axis: Axis[Linspace], tsdata: TSData):
    xp = tsdata.xp
    bad_rep = FrequencySeries(
        (lin_freq_axis,),
        xp.ones((1, 2, 1, 1, len(lin_freq_axis)), dtype=xp.float64),
    )

    with pytest.raises(ValueError, match="representation entries must have shape"):
        fsdata({"X": bad_rep})


def test_stftdata_and_wdmdata_factory_validation(stftdata_cartesian: STFTData):
    with pytest.raises(
        ValueError,
        match=(
            "Cannot specify `frequencies`, `times`, `entries`, `channels`, "
            "or `sparse_indices`"
        ),
    ):
        stftdata({"X": stftdata_cartesian["X"]}, frequencies=stftdata_cartesian.grid[0])

    with pytest.raises(
        ValueError,
        match="Must specify `frequencies`, `times`, `entries`, and `channels`",
    ):
        stftdata()

    with pytest.raises(
        ValueError,
        match="Must specify `frequencies`, `times`, `entries`, and `channels`",
    ):
        wdmdata()


def test_tsdata_zero_padding(tsdata: TSData):
    padded = tsdata.get_zero_padded((1.0, 2.0))
    assert isinstance(padded, TSData)
    assert len(padded.times) == len(tsdata.times) + 3
    assert padded.t_start < tsdata.t_start
    assert padded.t_end > tsdata.t_end


def test_roundtrip_load_data_dense_and_sparse(
    tsdata: TSData,
    fsdata: FSData,
    stftdata_cartesian: STFTData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
    stftdata_sparse: STFTData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
    wdmdata_cartesian: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
    wdmdata_sparse: WDMData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
    tmp_path,
):
    paths = {
        "ts": tmp_path / "roundtrip_ts.h5",
        "fs": tmp_path / "roundtrip_fs.h5",
        "stft": tmp_path / "roundtrip_stft.h5",
        "stft_sparse": tmp_path / "roundtrip_stft_sparse.h5",
        "wdm": tmp_path / "roundtrip_wdm.h5",
        "wdm_sparse": tmp_path / "roundtrip_wdm_sparse.h5",
    }

    tsdata.save(paths["ts"])
    fsdata.save(paths["fs"])
    stftdata_cartesian.save(paths["stft"])
    stftdata_sparse.save(paths["stft_sparse"])
    wdmdata_cartesian.save(paths["wdm"])
    wdmdata_sparse.save(paths["wdm_sparse"])

    loaded_ts = load_data(paths["ts"], domain="time", kind=None)
    loaded_fs = load_data(paths["fs"], domain="frequency", kind=None)
    loaded_stft = load_data(paths["stft"], domain="time-frequency", kind="stft")
    loaded_stft_sparse = load_data(
        paths["stft_sparse"],
        domain="time-frequency",
        kind="stft",
        sparse=True,
    )
    loaded_wdm = load_data(paths["wdm"], domain="time-frequency", kind="wdm")
    loaded_wdm_sparse = load_data(
        paths["wdm_sparse"],
        domain="time-frequency",
        kind="wdm",
        sparse=True,
    )

    assert isinstance(loaded_ts, TSData)
    assert isinstance(loaded_fs, FSData)
    assert loaded_stft.kind == "stft"
    assert isinstance(loaded_stft_sparse.grid, Grid2DSparse)
    assert loaded_wdm.kind == "wdm"
    assert isinstance(loaded_wdm_sparse.grid, Grid2DSparse)


def test_load_data_legacy_and_unsupported_combination(
    tsdata: TSData,
    tmp_path: pathlib.Path,
):
    legacy_path = tmp_path / "legacy_ts.h5"
    with h5py.File(str(legacy_path), "w") as f:
        f.attrs["type"] = "TSData"
        f.create_group("X")
        f["X"].create_dataset("grid", data=tsdata.times.asarray())  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        f["X"].create_dataset("entries", data=tsdata["X"].entries.squeeze())  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.warns(DeprecationWarning, match="legacy"):
        loaded_legacy = load_data(legacy_path, legacy=True)
    assert isinstance(loaded_legacy, TSData)

    unsupported_path = tmp_path / "legacy_unknown.h5"
    with h5py.File(str(unsupported_path), "w") as f:
        f.attrs["type"] = "NotSupportedData"
    with (
        pytest.raises(ValueError, match="Unsupported data type"),
        pytest.warns(DeprecationWarning, match="legacy"),
    ):
        load_data(unsupported_path, legacy=True)

    bad_combo_path = tmp_path / "bad_combo.h5"
    tsdata.save(bad_combo_path)
    with h5py.File(str(bad_combo_path), "a") as f:
        f.attrs["kind"] = "custom"
    with pytest.raises(ValueError, match="Unsupported combination"):
        load_data(bad_combo_path, domain="time", kind="custom")


def test_stft_and_wdm_data_reject_nonuniform_axes(
    ary_freq_axis: Axis[Array],
    ary_time_axis: Axis[Array],
    tsdata: TSData,
):
    xp = tsdata.xp
    dense_entries = xp.ones(
        (1, 1, 1, 1, len(ary_freq_axis), len(ary_time_axis)),
        dtype=xp.float64,
    )
    with pytest.raises(ValueError, match="must be uniform"):
        stftdata(
            frequencies=ary_freq_axis,
            times=ary_time_axis,
            entries=dense_entries,
            channels=("X",),
        )
    with pytest.raises(ValueError, match="must be uniform"):
        wdmdata(
            frequencies=ary_freq_axis,
            times=ary_time_axis,
            entries=dense_entries,
            channels=("X",),
        )
