"""Module for utilities.

.. currentmodule:: typed_lisa_toolkit.utils

Types
-----

.. autoclass:: ArrayFunc
.. autoclass:: Interpolator

Functions
---------

.. autofunction:: get_subset_slice
.. autofunction:: get_support_slice

Decorators
----------

.. autofunction:: trim_interp

"""

from __future__ import annotations
from collections.abc import Callable
import functools
import logging

import numpy as np
import numpy.typing as npt


log = logging.getLogger(__name__)


ArrayFunc = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
Interpolator = Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating]], ArrayFunc]


def get_subset_slice(
    increasing_array: npt.NDArray[np.floating], min: float, max: float
):
    """Return the index slice for the subset [min, max] of the increasing array.

    Examples
    --------
    >>> get_subset_slice(0.5 * np.arange(10), 1.0, 3.0)
    slice(2, 7, None)
    """
    start_idx = np.searchsorted(increasing_array, min, side="left")
    end_idx = np.searchsorted(increasing_array, max, side="right")
    return slice(start_idx, end_idx)


def get_support_slice(array: npt.NDArray[np.number]):
    """Return the index slice for the support of the array.

    If the array is all zeros, the slice returned is empty.

    Examples
    --------
    >>> get_support_slice(np.array([0, 0, 1, 2, 0, 0]))
    slice(2, 4, None)
    """
    non_zero_indices = np.flatnonzero(array)
    if non_zero_indices.size == 0:
        return slice(0, 0)
    return slice(non_zero_indices[0], non_zero_indices[-1] + 1)


def trim_interp(interpolator: Interpolator):
    """Return decorated interpolator.

    The decorated interpolate function will first trim the input
    array to its support before calling the original interpolator.
    Outside the support, the interpolated values are set to zero.
    """

    @functools.wraps(interpolator)
    def _interpolator(
        grid: npt.NDArray[np.floating],
        entries: npt.NDArray[np.floating],
    ) -> ArrayFunc:
        support_slice = get_support_slice(entries)
        try:
            min, max = grid[support_slice][[0, -1]]
        except IndexError:
            log.warning("Empty array. Returning a zero interpolator.")
            return lambda target_entries: np.zeros_like(target_entries)

        def _interpolated(
            target_grid: npt.NDArray[np.floating],
        ) -> npt.NDArray[np.floating]:
            """Return the interpolated entries at the target grid."""
            target_support_slice = get_subset_slice(target_grid, min, max)
            target_entries = np.zeros_like(target_grid)
            target_entries[target_support_slice] = interpolator(
                grid[support_slice], entries[support_slice]
            )(target_grid[target_support_slice])
            return target_entries

        return _interpolated

    return _interpolator
