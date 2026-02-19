"""Module for waveform containers.

.. currentmodule:: typed_lisa_toolkit.containers.waveforms

.. autoclass:: HarmonicWaveform
.. autoclass:: ProjectedWaveform
.. autoclass:: HarmonicProjectedWaveform

Functions
---------

.. autofunction:: sum_harmonics
.. autofunction:: get_dense_maker

"""

from __future__ import annotations
from collections.abc import Mapping
import logging
from typing import TYPE_CHECKING, Self, Any, Iterator
import array_api_compat as xpc

from . import representations as reps, data, modes


Reps = reps.TimeSeries | reps.FrequencySeries | reps.STFT | reps.Phasor | reps.WDM
Mode = modes.Harmonic | modes.QNM

from .. import utils

if TYPE_CHECKING:
    Array = reps.Array

log = logging.getLogger(__name__)


class _ModeMapping[ModeT: Mode, VT: Any](Mapping[ModeT, VT]):
    """Mixin for picking a single mode from a waveform."""

    def __init__(self, mapping: Mapping[ModeT, VT]):
        self._mapping = mapping

    # Implement Mapping protocol
    def __getitem__(self, key: ModeT) -> VT:
        """Get a channel by name as a view with preserved channel dimension (size 1)."""
        return self._mapping[key]

    def __iter__(self) -> Iterator[ModeT]:
        """Iterate over harmonic modes."""
        return iter(self.harmonics)

    def __len__(self) -> int:
        """Return the number of harmonic modes."""
        return len(self.harmonics)

    def __repr__(self):
        items = {key: self[key] for key in self}
        return f"{self.__class__.__name__}({items!r})"

    def pick(self, modes: ModeT | tuple[ModeT, ...]) -> Self:
        """Pick specific modes.

        Parameters
        ----------
        modes : Mode or tuple[Mode, ...]
            Mode(s) to pick.

        Returns
        -------
        Self
            New instance with only the specified mode(s).
        """
        try:
            return self._pick(modes)  # type: ignore[arg-type]
        except KeyError:
            return self._pick((modes,))  # type: ignore[arg-type]

    def _pick(self, modes: tuple[ModeT, ...]) -> Self:
        new_mapping = {mode: self[mode] for mode in modes}
        return type(self)(new_mapping)

    @property
    def harmonics(self):
        """All harmonic modes and their order."""
        return tuple(self.keys())


class HarmonicWaveform[ModeT: Mode, RepT: Reps](_ModeMapping[ModeT, RepT]):
    """Waveform in different modes.

    This class is a mapping from modes to representations.
    """

    def __xp__(self, api_version: str | None = None):
        """Get the array API namespace of the entries."""
        return xpc.get_namespace(next(iter(self.values())), api_version=api_version)

    @property
    def domain(self):
        """Physical domain shared by all harmonics."""
        return self[next(iter(self))].domain


class ProjectedWaveform[RepT: Reps](data._ChannelMapping[RepT]):  # pyright: ignore[reportPrivateUsage]
    """Projected waveform in a single mode or all modes summed together.

    This class is a mapping from channel names to representations. It is
    used to represent the detector response of a single harmonic waveform in different
    channels.
    """


class HarmonicProjectedWaveform[ModeT: Mode, RepT: Reps](
    _ModeMapping[ModeT, ProjectedWaveform[RepT]]
):
    """Projected waveform in different modes.

    This class is a mapping from modes to :class:`.ProjectedWaveform` instances.
    """

    def __xp__(self, api_version: str | None = None):
        """Get the array API namespace of the entries."""
        return self._first.__xp__(api_version=api_version)

    @property
    def _first(self):
        return self[next(iter(self))]

    @property
    def domain(self):
        """Physical domain shared by all harmonics."""
        return self._first.domain

    @property
    def channel_names(self) -> tuple[str, ...]:
        """All channel names."""
        return tuple(self._first.keys())

    @property
    def is_homogeneous(self) -> bool:
        """Whether the waveform is homogeneous across channels."""
        return all(
            self[harmonic].grid == self[next(iter(self))].grid
            for harmonic in self.harmonics
        )

    def get_kernel(self):
        """Return an array of the conventional shape.

        The shape is ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        The returned array is suitable for downstream processing (e.g., by noise models to compute inner products).
        This method is only valid when `is_homogeneous` is ``True``. Otherwise, this method should raise an error.
        """
        if not self.is_homogeneous:
            raise ValueError("The waveform is not homogeneous across channels.")
        xp = xpc.get_namespace(self._first.entries)
        return xp.stack([self[harmonic].entries for harmonic in self.harmonics], axis=2)


def sum_harmonics[ModeT: Mode, RepT: reps.FrequencySeries](
    wf: HarmonicProjectedWaveform[ModeT, RepT],
) -> ProjectedWaveform[RepT]:
    """Sum over modes."""
    entries = wf.get_kernel().sum(axis=2)  # c.f. shape convention
    return wf._first.create_like(entries, wf.channel_names)  # pyright: ignore[reportPrivateUsage]


def get_dense_maker[ModeT: Mode](
    interpolator: utils.Interpolator,
):
    """Return a function to convert a sparse phasor projected waveform to a dense phasor projected waveform.

    The returned function has the signature:

    .. code-block:: python

        def make(
            frequencies: npt.NDArray[np.floating],
            embed: bool = False,
        ) -> Callable[HarmonicProjectedWaveform[ModeT, reps.Phasor], HarmonicProjectedWaveform[ModeT, reps.Phasor]]:
            ...

    The function takes a list of frequencies and an optional boolean
    `embed` argument. If `embed` is True, the returned function will
    return a waveform with the same frequencies as the input. If `embed`
    is False, the returned function will return a waveform with the
    frequencies truncated to the lowest and highest frequencies of the
    input waveform.
    """

    def make(
        frequencies: "Array",
        embed: bool = False,
    ):

        def do_phasor(wf: reps.Phasor):
            freqs = frequencies[utils.get_subset_slice(frequencies, wf.f_min, wf.f_max)]
            nwf = wf.get_interpolated(freqs, interpolator)
            if not embed:
                return nwf
            return nwf.get_embedded(frequencies)

        def do_response(resp: ProjectedWaveform[reps.Phasor]):
            return ProjectedWaveform[reps.Phasor].from_dict(
                {chnname: do_phasor(resp[chnname]) for chnname in resp.channel_names}
            )

        def do(wf: HarmonicProjectedWaveform[ModeT, reps.Phasor]):
            return type(wf)({mode: do_response(wf[mode]) for mode in wf.harmonics})

        return do

    return make
