from types import ModuleType

from typed_lisa_toolkit.types import (
    STFT,
    WDM,
    AnyArray,
    Axis,
    FrequencySeries,
    FSData,
    Grid2DCartesian,
    Grid2DSparse,
    Harmonic,
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    HomogeneousHarmonicProjectedWaveform,
    Linspace,
    Phasor,
    ProjectedWaveform,
    STFTData,
    TimeSeries,
    TSData,
    UniformFrequencySeries,
    UniformTimeSeries,
    WDMData,
)

## Axis


def test_lin_time_axis(lin_time_axis: Axis[Linspace]):
    assert lin_time_axis.start == lin_time_axis.ax.start
    assert lin_time_axis.stop == lin_time_axis.ax.stop


def test_lin_freq_axis(lin_freq_axis: Axis[Linspace]):
    assert lin_freq_axis.start == lin_freq_axis.ax.start
    assert lin_freq_axis.stop == lin_freq_axis.ax.stop


def test_uni_ary_time_axis(uni_ary_time_axis: Axis[AnyArray]):
    assert uni_ary_time_axis.start == uni_ary_time_axis.ax[0]
    assert uni_ary_time_axis.stop == uni_ary_time_axis.ax[-1]


def test_uni_ary_freq_axis(uni_ary_freq_axis: Axis[AnyArray]):
    assert uni_ary_freq_axis.start == uni_ary_freq_axis.ax[0]
    assert uni_ary_freq_axis.stop == uni_ary_freq_axis.ax[-1]


def test_ary_time_axis(ary_time_axis: Axis[AnyArray]):
    assert ary_time_axis.start == ary_time_axis.ax[0]
    assert ary_time_axis.stop == ary_time_axis.ax[-1]


def test_ary_freq_axis(ary_freq_axis: Axis[AnyArray]):
    assert ary_freq_axis.start == ary_freq_axis.ax[0]
    assert ary_freq_axis.stop == ary_freq_axis.ax[-1]


## Grids


def test_lin_time_grid1d(lin_time_grid1d: tuple[Axis[Linspace]]):
    assert len(lin_time_grid1d) == 1
    assert isinstance(lin_time_grid1d[0], Axis)


def test_lin_freq_grid1d(lin_freq_grid1d: tuple[Axis[Linspace]]):
    assert len(lin_freq_grid1d) == 1
    assert isinstance(lin_freq_grid1d[0], Axis)


def test_uni_ary_time_grid1d(uni_ary_time_grid1d: tuple[Axis[AnyArray]]):
    assert len(uni_ary_time_grid1d) == 1
    assert isinstance(uni_ary_time_grid1d[0], Axis)


def test_uni_ary_freq_grid1d(uni_ary_freq_grid1d: tuple[Axis[AnyArray]]):
    assert len(uni_ary_freq_grid1d) == 1
    assert isinstance(uni_ary_freq_grid1d[0], Axis)


def test_ary_time_grid1d(ary_time_grid1d: tuple[Axis[AnyArray]]):
    assert len(ary_time_grid1d) == 1
    assert isinstance(ary_time_grid1d[0], Axis)


def test_ary_freq_grid1d(ary_freq_grid1d: tuple[Axis[AnyArray]]):
    assert len(ary_freq_grid1d) == 1
    assert isinstance(ary_freq_grid1d[0], Axis)


def test_lin_lin_cartesian(lin_lin_cartesian: tuple[Axis[Linspace], Axis[Linspace]]):
    assert len(lin_lin_cartesian) == 2
    assert isinstance(lin_lin_cartesian[0], Axis)
    assert isinstance(lin_lin_cartesian[1], Axis)


def test_lin_ary_cartesian(lin_ary_cartesian: tuple[Axis[Linspace], Axis[AnyArray]]):
    assert len(lin_ary_cartesian) == 2
    assert isinstance(lin_ary_cartesian[0], Axis)
    assert isinstance(lin_ary_cartesian[1], Axis)


def test_ary_lin_cartesian(ary_lin_cartesian: tuple[Axis[AnyArray], Axis[Linspace]]):
    assert len(ary_lin_cartesian) == 2
    assert isinstance(ary_lin_cartesian[0], Axis)
    assert isinstance(ary_lin_cartesian[1], Axis)


def test_ary_ary_cartesian(ary_ary_cartesian: tuple[Axis[AnyArray], Axis[AnyArray]]):
    assert len(ary_ary_cartesian) == 2
    assert isinstance(ary_ary_cartesian[0], Axis)
    assert isinstance(ary_ary_cartesian[1], Axis)


def test_lin_uni_cartesian(lin_uni_cartesian: tuple[Axis[Linspace], Axis[AnyArray]]):
    assert len(lin_uni_cartesian) == 2
    assert isinstance(lin_uni_cartesian[0], Axis)
    assert isinstance(lin_uni_cartesian[1], Axis)


def test_uni_lin_cartesian(uni_lin_cartesian: tuple[Axis[AnyArray], Axis[Linspace]]):
    assert len(uni_lin_cartesian) == 2
    assert isinstance(uni_lin_cartesian[0], Axis)
    assert isinstance(uni_lin_cartesian[1], Axis)


def test_uni_uni_cartesian(uni_uni_cartesian: tuple[Axis[AnyArray], Axis[AnyArray]]):
    assert len(uni_uni_cartesian) == 2
    assert isinstance(uni_uni_cartesian[0], Axis)
    assert isinstance(uni_uni_cartesian[1], Axis)


def test_ary_uni_cartesian(ary_uni_cartesian: tuple[Axis[AnyArray], Axis[AnyArray]]):
    assert len(ary_uni_cartesian) == 2
    assert isinstance(ary_uni_cartesian[0], Axis)
    assert isinstance(ary_uni_cartesian[1], Axis)


def test_uni_ary_cartesian(uni_ary_cartesian: tuple[Axis[AnyArray], Axis[AnyArray]]):
    assert len(uni_ary_cartesian) == 2
    assert isinstance(uni_ary_cartesian[0], Axis)
    assert isinstance(uni_ary_cartesian[1], Axis)


def test_lin_lin_sparse(
    xp: ModuleType, lin_lin_sparse: Grid2DSparse[Axis[Linspace], Axis[Linspace]]
):
    assert len(lin_lin_sparse) == 2
    assert isinstance(lin_lin_sparse[0], Axis)
    assert isinstance(lin_lin_sparse[1], Axis)
    assert xp.all(lin_lin_sparse.indices[:, 0] < len(lin_lin_sparse[0].ax))
    assert xp.all(lin_lin_sparse.indices[:, 1] < len(lin_lin_sparse[1].ax))
    assert lin_lin_sparse.indices.shape[1] == 2


def test_lin_uni_sparse(
    xp: ModuleType, lin_uni_sparse: Grid2DSparse[Axis[Linspace], Axis[AnyArray]]
):
    assert len(lin_uni_sparse) == 2
    assert isinstance(lin_uni_sparse[0], Axis)
    assert isinstance(lin_uni_sparse[1], Axis)
    assert xp.all(lin_uni_sparse.indices[:, 0] < len(lin_uni_sparse[0].ax))
    assert xp.all(lin_uni_sparse.indices[:, 1] < len(lin_uni_sparse[1].ax))
    assert lin_uni_sparse.indices.shape[1] == 2


def test_lin_ary_sparse(
    xp: ModuleType, lin_ary_sparse: Grid2DSparse[Axis[Linspace], Axis[AnyArray]]
):
    assert len(lin_ary_sparse) == 2
    assert isinstance(lin_ary_sparse[0], Axis)
    assert isinstance(lin_ary_sparse[1], Axis)
    assert xp.all(lin_ary_sparse.indices[:, 0] < len(lin_ary_sparse[0].ax))
    assert xp.all(lin_ary_sparse.indices[:, 1] < len(lin_ary_sparse[1].ax))
    assert lin_ary_sparse.indices.shape[1] == 2


def test_uni_lin_sparse(
    xp: ModuleType, uni_lin_sparse: Grid2DSparse[Axis[AnyArray], Axis[Linspace]]
):
    assert len(uni_lin_sparse) == 2
    assert isinstance(uni_lin_sparse[0], Axis)
    assert isinstance(uni_lin_sparse[1], Axis)
    assert xp.all(uni_lin_sparse.indices[:, 0] < len(uni_lin_sparse[0].ax))
    assert xp.all(uni_lin_sparse.indices[:, 1] < len(uni_lin_sparse[1].ax))
    assert uni_lin_sparse.indices.shape[1] == 2


def test_uni_uni_sparse(
    xp: ModuleType, uni_uni_sparse: Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]
):
    assert len(uni_uni_sparse) == 2
    assert isinstance(uni_uni_sparse[0], Axis)
    assert isinstance(uni_uni_sparse[1], Axis)
    assert xp.all(uni_uni_sparse.indices[:, 0] < len(uni_uni_sparse[0].ax))
    assert xp.all(uni_uni_sparse.indices[:, 1] < len(uni_uni_sparse[1].ax))
    assert uni_uni_sparse.indices.shape[1] == 2


def test_uni_ary_sparse(
    xp: ModuleType, uni_ary_sparse: Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]
):
    assert len(uni_ary_sparse) == 2
    assert isinstance(uni_ary_sparse[0], Axis)
    assert isinstance(uni_ary_sparse[1], Axis)
    assert xp.all(uni_ary_sparse.indices[:, 0] < len(uni_ary_sparse[0].ax))
    assert xp.all(uni_ary_sparse.indices[:, 1] < len(uni_ary_sparse[1].ax))
    assert uni_ary_sparse.indices.shape[1] == 2


def test_ary_lin_sparse(
    xp: ModuleType, ary_lin_sparse: Grid2DSparse[Axis[AnyArray], Axis[Linspace]]
):
    assert len(ary_lin_sparse) == 2
    assert isinstance(ary_lin_sparse[0], Axis)
    assert isinstance(ary_lin_sparse[1], Axis)
    assert xp.all(ary_lin_sparse.indices[:, 0] < len(ary_lin_sparse[0].ax))
    assert xp.all(ary_lin_sparse.indices[:, 1] < len(ary_lin_sparse[1].ax))
    assert ary_lin_sparse.indices.shape[1] == 2


def test_ary_uni_sparse(
    xp: ModuleType, ary_uni_sparse: Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]
):
    assert len(ary_uni_sparse) == 2
    assert isinstance(ary_uni_sparse[0], Axis)
    assert isinstance(ary_uni_sparse[1], Axis)
    assert xp.all(ary_uni_sparse.indices[:, 0] < len(ary_uni_sparse[0].ax))
    assert xp.all(ary_uni_sparse.indices[:, 1] < len(ary_uni_sparse[1].ax))
    assert ary_uni_sparse.indices.shape[1] == 2


def test_ary_ary_sparse(
    xp: ModuleType, ary_ary_sparse: Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]
):
    assert len(ary_ary_sparse) == 2
    assert isinstance(ary_ary_sparse[0], Axis)
    assert isinstance(ary_ary_sparse[1], Axis)
    assert xp.all(ary_ary_sparse.indices[:, 0] < len(ary_ary_sparse[0].ax))
    assert xp.all(ary_ary_sparse.indices[:, 1] < len(ary_ary_sparse[1].ax))
    assert ary_ary_sparse.indices.shape[1] == 2


## Representations


def test_lin_freq_series(lin_freq_series: UniformFrequencySeries):
    assert lin_freq_series.domain == "frequency"
    assert lin_freq_series.kind is None
    assert isinstance(lin_freq_series.grid, tuple)
    assert lin_freq_series.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_freq_series.grid[0].ax),
    )


def test_uni_ary_freq_series(
    uni_ary_freq_series: FrequencySeries[Axis[AnyArray]],
):
    assert uni_ary_freq_series.domain == "frequency"
    assert uni_ary_freq_series.kind is None
    assert isinstance(uni_ary_freq_series.grid, tuple)
    assert uni_ary_freq_series.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_ary_freq_series.grid[0].ax),
    )


def test_ary_freq_series(
    ary_freq_series: FrequencySeries[Axis[AnyArray]],
):
    assert ary_freq_series.domain == "frequency"
    assert ary_freq_series.kind is None
    assert isinstance(ary_freq_series.grid, tuple)
    assert ary_freq_series.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_freq_series.grid[0].ax),
    )


def test_lin_time_series(lin_time_series: UniformTimeSeries):
    assert lin_time_series.domain == "time"
    assert lin_time_series.kind is None
    assert isinstance(lin_time_series.grid, tuple)
    assert lin_time_series.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_time_series.grid[0].ax),
    )


def test_uni_ary_time_series(
    uni_ary_time_series: TimeSeries[Axis[AnyArray]],
):
    assert uni_ary_time_series.domain == "time"
    assert uni_ary_time_series.kind is None
    assert isinstance(uni_ary_time_series.grid, tuple)
    assert uni_ary_time_series.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_ary_time_series.grid[0].ax),
    )


def test_ary_time_series(
    ary_time_series: TimeSeries[Axis[AnyArray]],
):
    assert ary_time_series.domain == "time"
    assert ary_time_series.kind is None
    assert isinstance(ary_time_series.grid, tuple)
    assert ary_time_series.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_time_series.grid[0].ax),
    )


def test_lin_phasor(xp: ModuleType, lin_freq_phasor: Phasor[Axis[Linspace]]):
    assert lin_freq_phasor.domain == "frequency"
    assert lin_freq_phasor.kind == "phasor"
    assert not xp.isreal(lin_freq_phasor.amplitudes).any()
    assert xp.isreal(lin_freq_phasor.phases).all()
    assert isinstance(lin_freq_phasor.grid, tuple)
    assert lin_freq_phasor.entries.shape == (
        1,
        1,
        1,
        2,
        len(lin_freq_phasor.grid[0].ax),
    )


def test_uni_ary_phasor(xp: ModuleType, uni_ary_phasor: Phasor[Axis[AnyArray]]):
    assert uni_ary_phasor.domain == "frequency"
    assert uni_ary_phasor.kind == "phasor"
    assert not xp.isreal(uni_ary_phasor.amplitudes).any()
    assert xp.isreal(uni_ary_phasor.phases).all()
    assert isinstance(uni_ary_phasor.grid, tuple)
    assert uni_ary_phasor.entries.shape == (
        1,
        1,
        1,
        2,
        len(uni_ary_phasor.grid[0].ax),
    )


def test_ary_phasor(xp: ModuleType, ary_phasor: Phasor[Axis[AnyArray]]):
    assert ary_phasor.domain == "frequency"
    assert ary_phasor.kind == "phasor"
    assert not xp.isreal(ary_phasor.amplitudes).any()
    assert xp.isreal(ary_phasor.phases).all()
    assert isinstance(ary_phasor.grid, tuple)
    assert ary_phasor.entries.shape == (
        1,
        1,
        1,
        2,
        len(ary_phasor.grid[0].ax),
    )


def test_lin_lin_cartesian_stft(
    lin_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    assert lin_lin_cartesian_stft.domain == "time-frequency"
    assert lin_lin_cartesian_stft.kind == "stft"
    assert isinstance(lin_lin_cartesian_stft.grid, tuple)
    assert lin_lin_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_lin_cartesian_stft.grid[0].ax),
        len(lin_lin_cartesian_stft.grid[1].ax),
    )


def test_lin_uni_cartesian_stft(
    lin_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[AnyArray]]],
):
    assert lin_uni_cartesian_stft.domain == "time-frequency"
    assert lin_uni_cartesian_stft.kind == "stft"
    assert isinstance(lin_uni_cartesian_stft.grid, tuple)
    assert lin_uni_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_uni_cartesian_stft.grid[0].ax),
        len(lin_uni_cartesian_stft.grid[1].ax),
    )


def test_lin_ary_cartesian_stft(
    lin_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[Linspace], Axis[AnyArray]]],
):
    assert lin_ary_cartesian_stft.domain == "time-frequency"
    assert lin_ary_cartesian_stft.kind == "stft"
    assert isinstance(lin_ary_cartesian_stft.grid, tuple)
    assert lin_ary_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_ary_cartesian_stft.grid[0].ax),
        len(lin_ary_cartesian_stft.grid[1].ax),
    )


def test_ary_lin_cartesian_stft(
    ary_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[Linspace]]],
):
    assert ary_lin_cartesian_stft.domain == "time-frequency"
    assert ary_lin_cartesian_stft.kind == "stft"
    assert isinstance(ary_lin_cartesian_stft.grid, tuple)
    assert ary_lin_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_lin_cartesian_stft.grid[0].ax),
        len(ary_lin_cartesian_stft.grid[1].ax),
    )


def test_ary_uni_cartesian_stft(
    ary_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    assert ary_uni_cartesian_stft.domain == "time-frequency"
    assert ary_uni_cartesian_stft.kind == "stft"
    assert isinstance(ary_uni_cartesian_stft.grid, tuple)
    assert ary_uni_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_uni_cartesian_stft.grid[0].ax),
        len(ary_uni_cartesian_stft.grid[1].ax),
    )


def test_uni_lin_cartesian_stft(
    uni_lin_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[Linspace]]],
):
    assert uni_lin_cartesian_stft.domain == "time-frequency"
    assert uni_lin_cartesian_stft.kind == "stft"
    assert isinstance(uni_lin_cartesian_stft.grid, tuple)
    assert uni_lin_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_lin_cartesian_stft.grid[0].ax),
        len(uni_lin_cartesian_stft.grid[1].ax),
    )


def test_uni_uni_cartesian_stft(
    uni_uni_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    assert uni_uni_cartesian_stft.domain == "time-frequency"
    assert uni_uni_cartesian_stft.kind == "stft"
    assert isinstance(uni_uni_cartesian_stft.grid, tuple)
    assert uni_uni_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_uni_cartesian_stft.grid[0].ax),
        len(uni_uni_cartesian_stft.grid[1].ax),
    )


def test_uni_ary_cartesian_stft(
    uni_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    assert uni_ary_cartesian_stft.domain == "time-frequency"
    assert uni_ary_cartesian_stft.kind == "stft"
    assert isinstance(uni_ary_cartesian_stft.grid, tuple)
    assert uni_ary_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_ary_cartesian_stft.grid[0].ax),
        len(uni_ary_cartesian_stft.grid[1].ax),
    )


def test_ary_ary_cartesian_stft(
    ary_ary_cartesian_stft: STFT[Grid2DCartesian[Axis[AnyArray], Axis[AnyArray]]],
):
    assert ary_ary_cartesian_stft.domain == "time-frequency"
    assert ary_ary_cartesian_stft.kind == "stft"
    assert isinstance(ary_ary_cartesian_stft.grid, tuple)
    assert ary_ary_cartesian_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_ary_cartesian_stft.grid[0].ax),
        len(ary_ary_cartesian_stft.grid[1].ax),
    )


def test_lin_lin_sparse_stft(
    lin_lin_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    assert lin_lin_sparse_stft.domain == "time-frequency"
    assert lin_lin_sparse_stft.kind == "stft"
    assert isinstance(lin_lin_sparse_stft.grid, Grid2DSparse)
    assert lin_lin_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_lin_sparse_stft.grid.indices),
    )


def test_lin_uni_sparse_stft(
    lin_uni_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[AnyArray]]],
):
    assert lin_uni_sparse_stft.domain == "time-frequency"
    assert lin_uni_sparse_stft.kind == "stft"
    assert isinstance(lin_uni_sparse_stft.grid, Grid2DSparse)
    assert lin_uni_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_uni_sparse_stft.grid.indices),
    )


def test_lin_ary_sparse_stft(
    lin_ary_sparse_stft: STFT[Grid2DSparse[Axis[Linspace], Axis[AnyArray]]],
):
    assert lin_ary_sparse_stft.domain == "time-frequency"
    assert lin_ary_sparse_stft.kind == "stft"
    assert isinstance(lin_ary_sparse_stft.grid, Grid2DSparse)
    assert lin_ary_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_ary_sparse_stft.grid.indices),
    )


def test_uni_lin_sparse_stft(
    uni_lin_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[Linspace]]],
):
    assert uni_lin_sparse_stft.domain == "time-frequency"
    assert uni_lin_sparse_stft.kind == "stft"
    assert isinstance(uni_lin_sparse_stft.grid, Grid2DSparse)
    assert uni_lin_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_lin_sparse_stft.grid.indices),
    )


def test_uni_uni_sparse_stft(
    uni_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    assert uni_uni_sparse_stft.domain == "time-frequency"
    assert uni_uni_sparse_stft.kind == "stft"
    assert isinstance(uni_uni_sparse_stft.grid, Grid2DSparse)
    assert uni_uni_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_uni_sparse_stft.grid.indices),
    )


def test_uni_ary_sparse_stft(
    uni_ary_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    assert uni_ary_sparse_stft.domain == "time-frequency"
    assert uni_ary_sparse_stft.kind == "stft"
    assert isinstance(uni_ary_sparse_stft.grid, Grid2DSparse)
    assert uni_ary_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(uni_ary_sparse_stft.grid.indices),
    )


def test_ary_lin_sparse_stft(
    ary_lin_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[Linspace]]],
):
    assert ary_lin_sparse_stft.domain == "time-frequency"
    assert ary_lin_sparse_stft.kind == "stft"
    assert isinstance(ary_lin_sparse_stft.grid, Grid2DSparse)
    assert ary_lin_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_lin_sparse_stft.grid.indices),
    )


def test_ary_uni_sparse_stft(
    ary_uni_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    assert ary_uni_sparse_stft.domain == "time-frequency"
    assert ary_uni_sparse_stft.kind == "stft"
    assert isinstance(ary_uni_sparse_stft.grid, Grid2DSparse)
    assert ary_uni_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_uni_sparse_stft.grid.indices),
    )


def test_ary_ary_sparse_stft(
    ary_ary_sparse_stft: STFT[Grid2DSparse[Axis[AnyArray], Axis[AnyArray]]],
):
    assert ary_ary_sparse_stft.domain == "time-frequency"
    assert ary_ary_sparse_stft.kind == "stft"
    assert isinstance(ary_ary_sparse_stft.grid, Grid2DSparse)
    assert ary_ary_sparse_stft.entries.shape == (
        1,
        1,
        1,
        1,
        len(ary_ary_sparse_stft.grid.indices),
    )


def test_lin_lin_cartesian_wdm(
    lin_lin_cartesian_wdm: WDM[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    assert lin_lin_cartesian_wdm.domain == "time-frequency"
    assert lin_lin_cartesian_wdm.kind == "wdm"
    assert isinstance(lin_lin_cartesian_wdm.grid, tuple)
    assert lin_lin_cartesian_wdm.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_lin_cartesian_wdm.grid[0].ax),
        len(lin_lin_cartesian_wdm.grid[1].ax),
    )


def test_lin_lin_sparse_wdm(
    lin_lin_sparse_wdm: WDM[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    assert lin_lin_sparse_wdm.domain == "time-frequency"
    assert lin_lin_sparse_wdm.kind == "wdm"
    assert isinstance(lin_lin_sparse_wdm.grid, Grid2DSparse)
    assert lin_lin_sparse_wdm.entries.shape == (
        1,
        1,
        1,
        1,
        len(lin_lin_sparse_wdm.grid.indices),
    )


## Data


def test_tsdata(tsdata: TSData):
    assert tsdata.domain == "time"
    assert tsdata.kind is None
    assert isinstance(tsdata.grid, tuple)
    assert tsdata.entries.shape == (
        1,
        3,
        1,
        1,
        len(tsdata.grid[0].ax),
    )


def test_fsdata(fsdata: FSData):
    assert fsdata.domain == "frequency"
    assert fsdata.kind is None
    assert isinstance(fsdata.grid, tuple)
    assert fsdata.entries.shape == (
        1,
        3,
        1,
        1,
        len(fsdata.grid[0].ax),
    )


def test_stftdata_cartesian(
    stftdata_cartesian: STFTData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    assert stftdata_cartesian.domain == "time-frequency"
    assert stftdata_cartesian.kind == "stft"
    assert isinstance(stftdata_cartesian.grid, tuple)
    assert stftdata_cartesian.entries.shape == (
        1,
        3,
        1,
        1,
        len(stftdata_cartesian.grid[0].ax),
        len(stftdata_cartesian.grid[1].ax),
    )


def test_stftdata_sparse(
    stftdata_sparse: STFTData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    assert stftdata_sparse.domain == "time-frequency"
    assert stftdata_sparse.kind == "stft"
    assert isinstance(stftdata_sparse.grid, Grid2DSparse)
    assert stftdata_sparse.entries.shape == (
        1,
        3,
        1,
        1,
        len(stftdata_sparse.grid.indices),
    )


def test_wdmdata_cartesian(
    wdmdata_cartesian: WDMData[Grid2DCartesian[Axis[Linspace], Axis[Linspace]]],
):
    assert wdmdata_cartesian.domain == "time-frequency"
    assert wdmdata_cartesian.kind == "wdm"
    assert isinstance(wdmdata_cartesian.grid, tuple)
    assert wdmdata_cartesian.entries.shape == (
        1,
        3,
        1,
        1,
        len(wdmdata_cartesian.grid[0].ax),
        len(wdmdata_cartesian.grid[1].ax),
    )


def test_wdmdata_sparse(
    wdmdata_sparse: WDMData[Grid2DSparse[Axis[Linspace], Axis[Linspace]]],
):
    assert wdmdata_sparse.domain == "time-frequency"
    assert wdmdata_sparse.kind == "wdm"
    assert isinstance(wdmdata_sparse.grid, Grid2DSparse)
    assert wdmdata_sparse.entries.shape == (
        1,
        3,
        1,
        1,
        len(wdmdata_sparse.grid.indices),
    )


## Waveforms


def test_harmonic_waveform(
    harmonic_waveform: HarmonicWaveform[Harmonic, UniformFrequencySeries],
):
    assert harmonic_waveform.domain == "frequency"


def test_projected_waveform(
    projected_waveform: ProjectedWaveform[UniformFrequencySeries],
):
    assert projected_waveform.domain == "frequency"
    assert projected_waveform.kind is None


def test_harmonic_projected_waveform(
    harmonic_projected_waveform: HarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
):
    assert harmonic_projected_waveform.domain == "frequency"


def test_homogeneous_harmonic_projected_waveform(
    homogeneous_harmonic_projected_waveform: HomogeneousHarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
):
    assert homogeneous_harmonic_projected_waveform.domain == "frequency"
    assert homogeneous_harmonic_projected_waveform.get_kernel().shape == (
        1,
        len(homogeneous_harmonic_projected_waveform.channel_names),
        len(homogeneous_harmonic_projected_waveform.harmonics),
        1,
        3,
    )
