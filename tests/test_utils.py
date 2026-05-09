from types import ModuleType

from typed_lisa_toolkit import utils


def test_get_subset_slice_basic(xp: ModuleType):
    arr = xp.arange(10)
    # Should select indices 2 through 6 (inclusive of 2, exclusive of 7)
    s = utils.get_subset_slice(arr, 2, 6)
    assert arr[s].tolist() == [2, 3, 4, 5, 6]


def test_get_subset_slice_empty(xp: ModuleType):
    arr = xp.arange(10)
    # No values in [20, 30]
    s = utils.get_subset_slice(arr, 20, 30)
    assert arr[s].tolist() == []


def test_get_support_slice(xp: ModuleType):
    arr = xp.asarray([0, 0, 1, 2, 0, 0])
    s = utils.get_support_slice(arr)
    assert s == slice(2, 4)
    assert arr[s].tolist() == [1, 2]


def test_get_support_slice_all_zeros(xp: ModuleType):
    arr = xp.zeros(5)
    s = utils.get_support_slice(arr)
    assert s == slice(0, 0)
    assert arr[s].tolist() == []


def test_extend_to_1d(xp: ModuleType):
    grid = xp.arange(3, 8)
    entries = xp.asarray([1, 2, 3, 4, 5]).reshape(1, 1, 1, 1, 5)
    target_grid = xp.arange(10)
    extended = utils.extend_to(target_grid)(grid, entries)
    # Only indices 3-7 should be filled
    assert extended.shape == (1, 1, 1, 1, 10)
    assert (extended[..., 3:8] == entries).all()
    assert (extended[..., :3] == 0).all()
    assert (extended[..., 8:] == 0).all()


def test_trim_interp_decorator(xp: ModuleType):
    # entries is 1D (support detection); interp must return canonical 5D output
    def interp(x, y):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        return lambda t: xp.interp(t, x, y, left=0, right=0).reshape(1, 1, 1, 1, -1)  # type: ignore[return-value]

    decorated = utils.trim_interp(interp)  # pyright: ignore[reportUnknownArgumentType]
    arr = xp.asarray([0, 0, 1, 2, 0, 0])  # 1D
    grid = xp.arange(6)
    f = decorated(grid, arr)
    out = f(xp.arange(6))  # shape (1,1,1,1,6)
    assert out.shape == (1, 1, 1, 1, 6)
    assert (out[..., :2] == 0).all()
    assert (out[..., 4:] == 0).all()
    assert (out[..., 2:4] != 0).all()
