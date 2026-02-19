import pytest
import numpy as np
from scipy.signal import windows # type: ignore[import]
from typed_lisa_toolkit.containers.tapering import ldc_window, planck_window, get_tapering_func

def test_ldc_window():
    grid = np.linspace(0, 10000, 1000)
    taper = ldc_window(margin=1000.0, kap=0.005)
    result = taper(grid)

    assert result.shape == grid.shape
    assert np.all(result[:100] < 1)  # Tapering at the start
    assert np.all(result[-100:] < 1)  # Tapering at the end
    assert np.all(result[200:-200] == pytest.approx(1.0, rel=1e-2))  # Flat region


def test_planck_window():
    grid = np.linspace(0, 10000, 1000)
    taper = planck_window(left_margin=1000.0, right_margin=1000.0)
    result = taper(grid)

    assert result.shape == grid.shape
    assert result[0] == 0  # First element is zero
    assert result[-1] == 0  # Last element is zero
    assert np.all(result[:90] < 1)  # Tapering at the start
    assert np.all(result[-90:] < 1)  # Tapering at the end
    assert np.all(result[100:-100] == 1)  # Flat region


def test_get_tapering_func_with_callable():
    def custom_window(length, alpha=0.5):
        return np.hanning(length) * alpha

    tapering_func = get_tapering_func(custom_window, alpha=0.8)
    grid = np.linspace(0, 10000, 1000)
    result = tapering_func(grid)

    assert result.shape == grid.shape
    expected = custom_window(1000, alpha=0.8)  # Check if the result matches the expected window
    assert np.array_equal(result, expected)  # Compare the result with the expected window


def test_get_tapering_func_with_string():
    tapering_func = get_tapering_func("hann", sym=False)
    grid = np.linspace(0, 10000, 1000)
    result = tapering_func(grid)

    assert result.shape == grid.shape
    expected = windows.hann(1000, sym=False)  # Check if the result matches the expected window
    assert np.array_equal(result, expected)  # Compare the result with the expected window


def test_get_tapering_func_invalid_string():
    with pytest.raises(ValueError, match="Unknown window type."):
        get_tapering_func("invalid_window")