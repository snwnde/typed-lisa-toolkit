"""Module for waveform representation.

The module provides formatting and transformation functions, as well as
a protocol for waveform templates.

.. currentmodule:: typed_lisa_toolkit.containers.waveforms

Types
-----

.. autoclass:: NPFloatingT
.. autoclass:: NPNumberT
.. autoclass:: WaveformInMode
.. autoclass:: WaveformInChannel
.. autoclass:: Waveform
.. autoclass:: FormattedWaveform
.. autoclass:: FrequencyModeDict
.. autoprotocol:: FSWaveformGen

Functions
---------

.. autofunction:: format
.. autofunction:: to_fsdata
.. autofunction:: get_dense_maker

"""

from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import TypeVar, Protocol

import numpy as np
import numpy.typing as npt

from . import representations as reps, arithdicts, modes, data
from .. import utils

log = logging.getLogger(__name__)

NPFloatingT = TypeVar("NPFloatingT", bound=np.floating)
"""Numpy floating dtype."""

NPNumberT = TypeVar("NPNumberT", bound=np.number)
"""Numpy dtype."""

WaveformInMode = TypeVar("WaveformInMode", bound=reps.Representation)
"""Invariant type variable for either :class:`.series.FrequencySeries`,
:class:`.series.TimeSeries`, or :class:`.PhasorSequence`. Instances of
this type are used to represent the signal of a waveform in a specific
mode."""

_ModeT = TypeVar(
    "_ModeT", tuple[int, int], tuple[int, int, int], modes.Harmonic, modes.QNM
)
_FModeT = TypeVar("_FModeT", bound=modes.Harmonic | modes.QNM)
_PhasorT = TypeVar("_PhasorT", bound=reps.Phasor)

WaveformInChannel = Mapping[_ModeT, WaveformInMode]
Waveform = Mapping[arithdicts.ChnName, WaveformInChannel[_ModeT, WaveformInMode]]
FormattedWaveformInChannel = arithdicts.ModeDict[_FModeT, WaveformInMode]
FormattedWaveform = arithdicts.ChannelDict[
    FormattedWaveformInChannel[_FModeT, WaveformInMode]
]
FrequencyModeDict = arithdicts.ModeDict[_FModeT, npt.NDArray[np.floating]]


def _format_waveform_in_channel(wf: WaveformInChannel[_ModeT, WaveformInMode]):
    return arithdicts.ModeDict({modes.cast_mode(k): v for k, v in wf.items()})


def format(
    wf: Waveform[_ModeT, WaveformInMode],
):
    """Format a waveform.

    This function converts :class:`.Waveform` to :class:`.arithdicts.ChannelDict`
    of :class:`.arithdicts.ModeDict`.
    """
    return arithdicts.ChannelDict(
        {k: _format_waveform_in_channel(v) for k, v in wf.items()}
    )


def to_fsdata(
    wf: Waveform[_ModeT, reps.FrequencySeries[NPFloatingT, NPNumberT]],
) -> data.FSData[NPFloatingT, NPNumberT]:
    """Convert :class:`.Waveform` to :class:`.data.FSData`.

    This function accepts an instance of :class:`.Waveform` with
    signal represented as :class:`.series.FrequencySeries` and
    returns an instance of :class:`.data.FSData`.
    """

    def _sum_modes(wf: WaveformInChannel[_ModeT, WaveformInMode]):
        # Sum the contributions of all modes in each channel
        return _format_waveform_in_channel(wf).sum()

    return data.FSData({k: _sum_modes(v) for k, v in wf.items()})


class FSWaveformGen(Protocol):
    """Protocol for frequency domain waveform generators."""

    def __call__(
        self, *args, **kwargs
    ) -> FormattedWaveform[modes.Harmonic | modes.QNM, reps.FrequencySeries]:
        """Get the frequency domain waveform at the given frequencies."""


def get_dense_maker(
    interpolator: reps.Interpolator,
):
    """Return a function to convert a phasor waveform to a dense representation.

    This function is to be used with `pass_through` to convert a :class:`.arithdicts.ModeDict`
    or :class:`.arithdicts.ChannelDict` of :class:`.representations.Phasor` to dictionaries of
    dense representations.

    The returned function has the signature:

    .. code-block:: python

        def make(
            frequencies: npt.NDArray[np.floating],
            embed: bool = False,
        ) -> Callable[[reps.Phasor], reps.Phasor]

    The function takes a list of frequencies and an optional boolean
    `embed` argument. If `embed` is True, the returned function will
    return a waveform with the same frequencies as the input. If `embed`
    is False, the returned function will return a waveform with the
    frequencies truncated to the lowest and highest frequencies of the
    input waveform.
    """

    def make(
        frequencies: npt.NDArray[np.floating],
        embed: bool = False,
    ):
        def do(wf: _PhasorT):
            fmin, fmax = wf.frequencies[0], wf.frequencies[-1]
            freqs = frequencies[utils.get_subset_slice(frequencies, fmin, fmax)]
            nwf = wf.get_interpolated(freqs, interpolator)
            if not embed:
                return nwf
            return nwf.get_embedded(frequencies)

        return do

    return make
