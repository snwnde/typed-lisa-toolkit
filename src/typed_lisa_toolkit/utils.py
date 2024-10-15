from __future__ import annotations
import logging

import numpy as np
import numpy.typing as npt


log = logging.getLogger(__name__)


def get_subset_slice(
    increasing_array: npt.NDArray[np.floating], min: float, max: float
):
    """Return the slice for the subset [min, max] of the increasing array."""
    start_idx = np.searchsorted(increasing_array, min, side="left")
    end_idx = np.searchsorted(increasing_array, max, side="right")
    return slice(start_idx, end_idx)
