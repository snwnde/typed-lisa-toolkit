"""Module for waveform containers."""

from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import Callable

import numpy as np
import numpy.typing as npt

from . import series, arithdicts, modes, data

log = logging.getLogger(__name__)

ArrayFunc = Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]]
WaveformInChannel = Mapping[tuple[int, ...], series.FrequencySeries[np.complexfloating]]
Waveform = Mapping[arithdicts.ChnName, WaveformInChannel]


class PhasorSequence:
    """A sequence of phasors.

    Notes
    -----
    A phasor is a couple of (amplitude, phase) that represents a complex number.
    This class encapsulates a sequence of phasors at different frequencies, which
    can be used to represent a waveform.

    This representation is useful for interpolating waveforms generated on a sparse
    grid of frequencies to a dense grid of frequencies.
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
    ):
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
    ):
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
    ):
        """Get the phasors interpolated to the given frequencies."""
        amplitudes = interpolator(self.frequencies, self.amplitudes)(frequencies)
        phases = interpolator(self.frequencies, self.phases)(frequencies)
        return type(self)(frequencies, amplitudes, phases)

    def to_freq_series(self):
        """Get the complex representation of the waveform."""
        return series.FrequencySeries(
            self.frequencies,
            self.phasor_to_cplx(self.amplitudes, self.phases),
        )

    @classmethod
    def from_freq_series(
        cls, freq_series: series.FrequencySeries[np.complexfloating]
    ) -> PhasorSequence:
        """Create a PhasorSequence from a FrequencySeries."""
        amplitudes, phases = cls.cplx_to_phasor(freq_series.signal)
        return cls(freq_series.grid, amplitudes, phases)


def _format_waveform_in_channel(wf: WaveformInChannel):
    return arithdicts.ModeDict({modes.cast_mode(k): v for k, v in wf.items()})


def format(wf: Waveform):
    """Format a waveform to a ChennlDict of ModeDict."""
    return arithdicts.ChannelDict(
        {k: _format_waveform_in_channel(v) for k, v in wf.items()}
    )


def to_fsdata(wf: Waveform):
    """Convert a waveform to a FrequencySeries data."""

    def _sum_modes(wf: WaveformInChannel):
        # Sum the contributions of all modes in each channel
        return _format_waveform_in_channel(wf).sum()

    return data.FSData({k: _sum_modes(v) for k, v in wf.items()})
