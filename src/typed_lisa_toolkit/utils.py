"""Module for utilities.

.. currentmodule:: typed_lisa_toolkit.utils

Functions
---------

.. autofunction:: get_subset_slice
.. autofunction:: get_support_slice
.. autofunction:: extend_to

Decorators
----------

.. autofunction:: trim_interp

"""

from __future__ import annotations
from collections.abc import Callable
import functools
import logging
from typing import TYPE_CHECKING
import array_api_compat as xpc

if TYPE_CHECKING:
    import jax
    import jax.typing as jpt
    import numpy as np
    import numpy.typing as npt

    ArrayLike = jpt.ArrayLike | npt.ArrayLike
    Array = jax.Array | npt.NDArray[np.number]
    ArrayFunc = Callable[[Array], Array]
    Interpolator = Callable[[Array, Array], ArrayFunc]


log = logging.getLogger(__name__)


def get_subset_slice(increasing_array: Array, min: float, max: float):
    """Return the index slice for the subset [min, max] of the increasing array.

    Examples
    --------
    >>> get_subset_slice(0.5 * np.arange(10), 1.0, 3.0)
    slice(2, 7, None)
    """
    xp = xpc.get_namespace(increasing_array)
    start_idx = xp.searchsorted(increasing_array, min, side="left")
    end_idx = xp.searchsorted(increasing_array, max, side="right")
    return slice(int(start_idx), int(end_idx))


def get_support_slice(array: Array):
    """Return the index slice for the support of the array.

    If the array is all zeros, the slice returned is empty.

    Examples
    --------
    >>> get_support_slice(np.array([0, 0, 1, 2, 0, 0]))
    slice(2, 4, None)
    """
    xp = xpc.get_namespace(array)
    non_zero_indices = xp.flatnonzero(array)
    if non_zero_indices.size == 0:
        return slice(0, 0)
    return slice(non_zero_indices[0], non_zero_indices[-1] + 1)


def extend_to[ArrayT: Array](target_grid: tuple[ArrayT, ...] | ArrayT):
    """Return a function that extends the entries to the target grid.

    The returned function has the signature:

    .. code-block:: python

            def get_extension(grid: tuple[Array, ...], entries: Array) -> Array:
                ...

    The function extends the entries to the target grid by setting the entries
    outside the input grid to zero. Both the input grid and the target grid are
    assumed to be increasing, and the input grid is assumed to be a connected
    subset of the target grid. Both grids should be tuples of arrays.

    The entries are assumed to have canonical shape:
    ``(n_batches, n_channels, n_harmonics, n_features, *grid_dims)``

    Examples
    --------
    >>> grid = (2 + np.arange(5),)
    >>> entries = np.array([1, 2, 3, 4, 5]).reshape(1, 1, 1, 1, 5)
    >>> target_grid = (np.arange(10),)
    >>> result = extend_to(target_grid)(grid, entries)
    >>> result.shape
    (1, 1, 1, 1, 10)

    """
    # One might consider pre-allocate memory for extended_entries, i.e. designing
    # a class that stores extended_entries and reinitializes it with zeros before
    # each call to get_extension. This would accelerate the function, but I fail
    # to guarantee the correctness of the implementation, probably due to some
    # issue with np.memmap.fill.
    _target_grid = target_grid if isinstance(target_grid, tuple) else (target_grid,)

    def get_extension(
        grid: tuple[ArrayT, ...] | ArrayT,
        entries: ArrayT,
    ) -> ArrayT:
        _grid = grid if isinstance(grid, tuple) else (grid,)
        xp = xpc.get_namespace(entries)
        # Handle canonical shape: (n_batches, n_channels, n_harmonics, n_features, *grid_dims)
        # Create extended array with target_grid lengths in the grid dimensions
        extended_shape = entries.shape[:4] + tuple(len(g) for g in _target_grid)
        extended_entries = xp.zeros(extended_shape, dtype=entries.dtype)
        support_slices = tuple(
            get_subset_slice(target_g, float(g[0]), float(g[-1]))
            for target_g, g in zip(_target_grid, _grid)
        )
        # Build full indexing tuple for canonical shape
        index = (slice(None),) * 4 + support_slices

        try:
            extended_entries[index] = entries
        except TypeError:
            # Fall back to functional update (JAX immutable arrays)
            extended_entries = extended_entries.at[index].set(entries)

        return extended_entries

    return get_extension


def trim_interp(interpolator: Interpolator):
    """Return decorated interpolator.

    The decorated interpolate function will first trim the input
    array to its support before calling the original interpolator.
    Outside the support, the interpolated values are set to zero.
    """

    @functools.wraps(interpolator)
    def _interpolator(
        grid: Array,
        entries: Array,
    ) -> ArrayFunc:
        support_slice = get_support_slice(entries)
        xp = xpc.get_namespace(entries)
        trimmed_grid = grid[support_slice]
        if trimmed_grid.size == 0:
            log.warning("Empty array. Returning a zero interpolator.")
            return lambda target_entries: xp.zeros_like(target_entries)
        min = float(trimmed_grid[0])
        max = float(trimmed_grid[-1])


        def _interpolated(target_grid: Array) -> Array:
            """Return the interpolated entries at the target grid, zero outside support."""
            target_support_slice = get_subset_slice(target_grid, min, max)
            interp_grid = target_grid[target_support_slice]
            interpolated = interpolator(grid[support_slice], entries[support_slice])(interp_grid)
            return extend_to(target_grid)(interp_grid, interpolated)

        return _interpolated

    return _interpolator
