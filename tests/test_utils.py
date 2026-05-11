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


def test_promote_slice():
    s = utils.promote_slice(slice(2, 5))
    assert s == (slice(None),) * 4 + (slice(2, 5),)


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
