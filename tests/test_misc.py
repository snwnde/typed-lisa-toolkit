from types import ModuleType

import typed_lisa_toolkit as tlt


def test_axis_getitem(
    xp: ModuleType,
    uni_ary_time_axis: tlt.types.Axis[tlt.types.AnyArray],
    lin_time_axis: tlt.types.Axis[tlt.types.Linspace],
):
    assert uni_ary_time_axis[0] == uni_ary_time_axis.ax[0]
    mask = uni_ary_time_axis.ax > 2.0
    masked = uni_ary_time_axis[mask]
    assert isinstance(masked, tlt.types.Axis)
    assert masked.ax.shape == (xp.sum(mask),)
    assert lin_time_axis[0] == lin_time_axis.ax[0]
    mask = lin_time_axis.asarray(xp) > 2.0
    masked = lin_time_axis[mask]
    assert isinstance(masked, tlt.types.Axis)
    assert masked.ax.shape == (xp.sum(mask),)
