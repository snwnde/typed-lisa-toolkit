"""Module for waveform representation.

We model LISA (TDI) waveforms as an encapsulation of :class:`.series.FrequencySeries`
or :class:`PhasorSequence`
(in the future we can extend this to time-frequency matrices) in :class:`.arithdicts.ModeDict`
wrapped in :class:`.arithdicts.ChannelDict`.
"""

from __future__ import annotations
from collections.abc import Mapping, Callable
import logging
from typing import Self, TypeVar

import numpy as np
import numpy.typing as npt

from . import series, arithdicts, modes, data

log = logging.getLogger(__name__)

ArrayFunc = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
WaveformInMode = TypeVar(
    "WaveformInMode",
    "series.FrequencySeries[np.complexfloating]",
    "series.TimeSeries[np.complexfloating]",
    "PhasorSequence",
)
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
    wf: Waveform[series.FrequencySeries[np.complexfloating]],
) -> data.FSData[np.complexfloating]:
    """Convert :class:`.Waveform` to :class:`.data.FSData`.

    This function accepts an instance of :class:`.Waveform` with
    signal represented as :class:`.series.FrequencySeries` and
    returns an instance of :class:`.data.FSData`.
    """

    def _sum_modes(wf: WaveformInChannel[WaveformInMode]):
        # Sum the contributions of all modes in each channel
        return _format_waveform_in_channel(wf).sum()

    return data.FSData({k: _sum_modes(v) for k, v in wf.items()})


class PhasorSequence:
    """A sequence of phasors.

    A phasor is a couple of amplitude and phase that represent a complex number.
    This class encapsulates a sequence of phasors at different frequencies, which
    can be used to represent a waveform. This representation is useful for
    interpolating waveforms generated on a sparse grid of frequencies to a dense
    grid of frequencies.

    This class also follows :class:`.arithdicts.SupportsArithmetic` so that it is
    a valid value for :class:`.arithdicts.ModeDict`.
    """

    def __init__(
        self,
        frequencies: npt.NDArray[np.floating],
        amplitudes: npt.NDArray[np.floating],
        phases: npt.NDArray[np.floating],
    ):
        self.frequencies = frequencies
        self.amplitudes = amplitudes
        self.phases = phases

    @staticmethod
    def reim_to_cplx(
        real_parts: npt.NDArray[np.floating],
        imag_parts: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.complexfloating]:
        """Convert real and imaginary parts to complex numbers."""
        return real_parts + 1j * imag_parts

    @staticmethod
    def cplx_to_phasor(
        complex_numbers: npt.NDArray[np.complexfloating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Convert complex numbers to phasors."""
        return np.abs(complex_numbers), np.angle(complex_numbers)

    @staticmethod
    def phasor_to_cplx(
        amplitudes: npt.NDArray[np.floating],
        phases: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.complexfloating]:
        """Convert phasors to complex numbers."""
        return amplitudes * np.exp(1j * phases)

    @staticmethod
    def reim_to_phasor(
        real_parts: npt.NDArray[np.floating],
        imag_parts: npt.NDArray[np.floating],
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Convert real and imaginary parts to phasors."""
        return PhasorSequence.cplx_to_phasor(
            PhasorSequence.reim_to_cplx(real_parts, imag_parts)
        )

    def get_interpolated(
        self,
        frequencies: npt.NDArray[np.floating],
        interpolator: Callable[..., ArrayFunc],
    ) -> Self:
        """Get the phasors interpolated to the given frequencies."""
        amplitudes = interpolator(self.frequencies, self.amplitudes)(frequencies)
        phases = interpolator(self.frequencies, self.phases)(frequencies)
        return type(self)(frequencies, amplitudes, phases)

    def to_freq_series(self) -> series.FrequencySeries[np.complexfloating]:
        """Get the complex representation of the waveform."""
        return series.FrequencySeries(
            self.frequencies,
            self.phasor_to_cplx(self.amplitudes, self.phases),
        )

    @classmethod
    def from_freq_series(
        cls, freq_series: series.FrequencySeries[np.complexfloating]
    ) -> PhasorSequence:
        """Create a :class:`.PhasorSequence` from a :class:`.series.FrequencySeries`."""
        amplitudes, phases = cls.cplx_to_phasor(freq_series.signal)
        return cls(freq_series.grid, amplitudes, phases)

    def __add__(self, other: PhasorSequence):
        """Add two phasor sequences."""
        return self.from_freq_series(self.to_freq_series() + other.to_freq_series())

    def __neg__(self):
        """Negate a phasor sequence."""
        return self.from_freq_series(-self.to_freq_series())

    def __sub__(self, other: PhasorSequence):
        """Subtract two phasor sequences."""
        return self.from_freq_series(self.to_freq_series() - other.to_freq_series())

    def __mul_phasor_sequence__(self, other: PhasorSequence):
        """Multiply two phasor sequences."""
        return self.from_freq_series(self.to_freq_series() * other.to_freq_series())

    def __mul_num__(self, other: series.Numeric):
        """Multiply a phasor sequence by a number."""
        return self.from_freq_series(self.to_freq_series() * other)

    def __mul__(self, other: PhasorSequence | series.Numeric):
        """Multiply a phasor sequence by another phasor sequence or a number."""
        if isinstance(other, PhasorSequence):
            return self.__mul_phasor_sequence__(other)
        return self.__mul_num__(other)

    def __rmul__(self, other: series.Numeric):
        """Multiply a number by a phasor sequence."""
        return self.__mul_num__(other)

    def __truediv__(self, other: PhasorSequence | series.Numeric):
        """Divide a phasor sequence by another phasor sequence or a number."""
        if isinstance(other, PhasorSequence):
            return self.from_freq_series(self.to_freq_series() / other.to_freq_series())
        return self.from_freq_series(self.to_freq_series() / other)

    def __rtruediv__(self, other: series.Numeric):
        """Divide a number by a phasor sequence."""
        return self.from_freq_series(other / self.to_freq_series())
