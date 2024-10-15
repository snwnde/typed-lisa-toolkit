"""Module for utilities.

.. currentmodule:: typed_lisa_toolkit.utils

Functions
---------

.. autofunction:: get_subset_slice

"""

from __future__ import annotations
import logging

import numpy as np
import numpy.typing as npt


log = logging.getLogger(__name__)


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
