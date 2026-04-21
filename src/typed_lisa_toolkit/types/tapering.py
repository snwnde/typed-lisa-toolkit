"""
Module for tapering functions.

This module provides a function to create tapering functions compatible with
the transformations defined in the :mod:`.representations` module,
for the window functions
from the :mod:`scipy.signal.windows` module. For this purpose, use the
:func:`get_tapering_func` function.

The module also provides custom tapering functions, ready to use upon
choosing the parameters. Refer to the section "Tapering Functions" for a list.

.. currentmodule:: typed_lisa_toolkit.types.tapering

Protocols
---------

.. autoclass:: Tapering
.. autoclass:: LenWindow

Utility Functions
-----------------
.. autofunction:: get_tapering_func

Tapering Functions
------------------
.. autoclass:: ldc_window
.. autoclass:: planck_window
"""

import logging
from typing import TYPE_CHECKING, ParamSpec, Protocol, final

import numpy as np
import numpy.typing as npt
import scipy.special
from scipy.signal.windows import _windows  # type: ignore[import]

if TYPE_CHECKING:
    import jax
    import jax.typing as jpt
    import numpy as np
    import numpy.typing as npt

    ArrayLike = jpt.ArrayLike | npt.ArrayLike
    Array = jax.Array | npt.NDArray[np.number]

log = logging.getLogger(__name__)

P = ParamSpec("P")
"""ParamSpec for the window functions."""


class Tapering(Protocol):
    """Protocol for tapering functions."""

    def __call__(self, __array: "Array") -> "Array":  # noqa: PYI063
        """Return the tapering window to apply on the array."""
        ...


class LenWindow[**P](Protocol):
    """Protocol for window functions with a length argument."""

    def __call__(  # noqa: D102
        self,
        __len: int,  # noqa: PYI063
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> "Array": ...


# class _ScipyWindow(Protocol, Generic[P]):
#     """Protocol for scipy window functions."""

#     def __call__(
#         self, M: int, *args: P.args, sym: bool = True, **kwargs: P.kwargs
#     ) -> npt.NDArray[np.floating]: ...


# def _scipyfy(window: LenWindow[P]) -> _ScipyWindow[P]:
#     """Return a callable window function."""

#     @functools.wraps(window)
#     def scipy_window(M: int, *args: P.args, sym: bool = True, **kwargs: P.kwargs):
#         if _windows._len_guards(M):
#             return np.ones(M)
#         M, needs_trunc = _windows._extend(M, sym)
#         w = window(M, *args, **kwargs)
#         return _windows._truncate(w, needs_trunc)

#     return scipy_window


@final
class ldc_window(Tapering):  # noqa: N801
    """A window function that tapers a margin at both ends.

    Parameters
    ----------
    margin
        The margin to taper at both ends of the array, in the same units as the grid.

    kap
        The steepness of the tapering function. A larger value results in a steeper
        tapering function.
    """

    def __init__(self, margin: float = 1000.0, kap: float = 0.005):
        self.margin = margin
        self.kap = kap

    def __call__(self, grid: "Array") -> npt.NDArray[np.floating]:
        """Return the tapering window to apply on the array."""
        xr = grid[-1] - self.margin
        xl = grid[0] + self.margin
        winl = 0.5 * (1.0 + np.tanh(self.kap * (grid - xl)))
        winr = 0.5 * (1.0 - np.tanh(self.kap * (grid - xr)))
        return winl * winr


@final
class planck_window(Tapering):  # noqa: N801
    """A Planck taper window [1]_.

    Parameters
    ----------
    left_margin :
        The margin to taper at the left end of the array, in the same units as the grid.
    right_margin :
        The margin to taper at the right end of the array,
        in the same units as the grid.

    References
    ----------
    .. [1] McKechan, D.J.A., Robinson, C., and Sathyaprakash, B.S. (April
           2010). "A tapering window for time-domain templates and simulated
           signals in the detection of gravitational waves from coalescing
           compact binaries". Classical and Quantum Gravity 27 (8).
           :doi:`10.1088/0264-9381/27/8/084020`

    """

    def __init__(self, left_margin: float = 0, right_margin: float = 0):
        self.xl = left_margin
        self.xr = right_margin

    def __call__(self, grid: "Array"):
        """Return the tapering window to apply on the array."""
        # https://arxiv.org/abs/1003.2939
        win = np.ones_like(grid)
        win[0] = 0
        win[-1] = 0
        g_min, g_max = grid[0], grid[-1]
        lgrid = grid[grid <= g_min + self.xl]
        rgrid = grid[grid >= g_max - self.xr]
        zl = self.xl * (1 / (lgrid[1:] - g_min) + 1 / (lgrid[1:] - g_min - self.xl))
        zr = -self.xr * (1 / (rgrid[:-1] - g_max) + 1 / (rgrid[:-1] - g_max + self.xr))
        # pylint: disable=no-member
        scipy.special.expit(-zl, out=win[1 : len(lgrid)])
        scipy.special.expit(-zr, out=win[-len(rgrid) : -1])
        return win


def get_tapering_func[**P](  # pyright: ignore[reportUnknownParameterType]
    window: LenWindow[P] | str, *args: P.args, **kwargs: P.kwargs
):
    """Return a callable tapering function.

    This function wraps the given window function with their arguments (if any)
    and returns a callable function that takes the length of the array as input
    and returns the tapering window to apply on the array.

    If `window` is a string, it will be looked up in the
    :mod:`scipy.signal.windows` module.

    Example
    -------
    >>> tapering_func1 = get_tapering_func(scipy.signal.windows.tukey, alpha=0.5, sym=False)
    >>> tapering_func2 = get_tapering_func("hann", sym=False)

    """  # noqa: E501
    _msg = (
        "Unknown window type: {window}. "
        "Supported window types are those in `scipy.signal.windows`."
    )
    if not isinstance(window, str):
        return lambda x: window(len(x), *args, **kwargs)  # pyright: ignore[reportUnknownVariableType, reportUnknownLambdaType, reportUnknownArgumentType]
    try:
        winfunc = _windows._win_equiv[window]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
    except AttributeError:
        try:
            winfunc = _windows._WIN_FUNCS[window][0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportPrivateUsage]
        except KeyError as e:
            msg = _msg.format(window=window)
            raise ValueError(msg) from e
    except KeyError as e:
        msg = _msg.format(window=window)
        raise ValueError(msg) from e
    return get_tapering_func(winfunc, *args, **kwargs)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
