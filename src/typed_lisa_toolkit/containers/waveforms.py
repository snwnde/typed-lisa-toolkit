"""Module for waveform representation formatting and transformation tool.

.. currentmodule:: typed_lisa_toolkit.containers.waveforms

Types
-----

.. autoclass:: NPFloatingT
.. autoclass:: NPNumberT
.. autoclass:: WaveformInMode
.. autoclass:: WaveformInChannel
.. autoclass:: Waveform
.. autoclass:: FormattedWaveform

Functions
---------

.. autofunction:: format
.. autofunction:: to_fsdata

"""

from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import TypeVar

import numpy as np

from . import representations, arithdicts, modes, data

log = logging.getLogger(__name__)

NPFloatingT = TypeVar("NPFloatingT", bound=np.floating)
"""Numpy floating dtype."""

NPNumberT = TypeVar("NPNumberT", bound=np.number)
"""Numpy dtype."""

WaveformInMode = TypeVar("WaveformInMode", bound=representations.Representation)
"""Invariant type variable for either :class:`.series.FrequencySeries`,
:class:`.series.TimeSeries`, or :class:`.PhasorSequence`. Instances of
this type are used to represent the signal of a waveform in a specific
mode."""

WaveformInChannel = Mapping[tuple[int, ...], WaveformInMode]
Waveform = Mapping[arithdicts.ChnName, WaveformInChannel[WaveformInMode]]
FormattedWaveform = arithdicts.ChannelDict[
    arithdicts.ModeDict[modes.Harmonic | modes.QNM, WaveformInMode]
]


def _format_waveform_in_channel(wf: WaveformInChannel[WaveformInMode]):
    return arithdicts.ModeDict({modes.cast_mode(k): v for k, v in wf.items()})


def format(wf: Waveform[WaveformInMode]) -> FormattedWaveform:
    """Format a waveform.

    This function converts :class:`.Waveform` to :class:`.arithdicts.ChannelDict`
    of :class:`.arithdicts.ModeDict`.
    """
    return arithdicts.ChannelDict(
        {k: _format_waveform_in_channel(v) for k, v in wf.items()}
    )


def to_fsdata(
    wf: Waveform[representations.FrequencySeries[NPFloatingT, NPNumberT]],
) -> data.FSData[NPFloatingT, NPNumberT]:
    """Convert :class:`.Waveform` to :class:`.data.FSData`.

    This function accepts an instance of :class:`.Waveform` with
    signal represented as :class:`.series.FrequencySeries` and
    returns an instance of :class:`.data.FSData`.
    """

    def _sum_modes(wf: WaveformInChannel[WaveformInMode]):
        # Sum the contributions of all modes in each channel
        return _format_waveform_in_channel(wf).sum()

    return data.FSData({k: _sum_modes(v) for k, v in wf.items()})
