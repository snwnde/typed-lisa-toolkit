"""Module for waveform containers.

.. currentmodule:: typed_lisa_toolkit.containers.waveforms

.. autoclass:: HarmonicWaveform
.. autoclass:: ProjectedWaveform
.. autoclass:: HarmonicProjectedWaveform

Functions
---------

.. autofunction:: harmonic_waveform
.. autofunction:: projected_waveform
.. autofunction:: harmonic_projected_waveform
.. autofunction:: homogeneous_harmonic_projected_waveform
.. autofunction:: sum_harmonics
.. autofunction:: densify_phasor
.. autofunction:: densify_phasor_hw
.. autofunction:: densify_phasor_pw
.. autofunction:: densify_phasor_hpw
.. autofunction:: phasor_to_fs_hw
.. autofunction:: phasor_to_fs_pw
.. autofunction:: phasor_to_fs_hpw
.. autofunction:: combine_phasor_hw
.. autofunction:: combine_phasor_pw
.. autofunction:: combine_phasor_hpw
.. autofunction:: get_dense_maker

Alias Constructors
------------------
Use these as convenience shorthands; canonical names remain preferred in library code.

.. autofunction:: hw
.. autofunction:: hhw
.. autofunction:: pw
.. autofunction:: hpw
.. autofunction:: hhpw


"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Protocol,
    Self,
    cast,
    overload,
)

import array_api_compat as xpc

from . import data, modes
from . import representations as reps

Mode = modes.Harmonic | modes.QNM

from .. import utils

if TYPE_CHECKING:
    Array = reps.Array
    Axis = reps.Axis
    AnyGrid = reps.AnyGrid
    from .representations import Representation

    AnyReps = Representation[AnyGrid]

log = logging.getLogger(__name__)


class _ModeMapping[ModeT: Mode, VT: Any](Mapping[ModeT, VT]):
    """Mixin for picking a single mode from a waveform."""

    def __init__(self, mapping: Mapping[ModeT, Any]):
        self._mapping: Mapping[ModeT, Any] = mapping

    # Implement Mapping protocol
    def __getitem__(self, key: ModeT) -> VT:
        """Get a channel by name as a view with preserved channel dimension (size 1)."""
        return self._mapping[key]

    def __iter__(self) -> Iterator[ModeT]:
        """Iterate over harmonic modes."""
        return iter(self._mapping)

    def __len__(self) -> int:
        """Return the number of harmonic modes."""
        return len(self._mapping)

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
        return tuple(self._mapping.keys())


class HarmonicWaveform[ModeT: Mode, RepT: AnyReps](_ModeMapping[ModeT, RepT]):
    """Waveform in different modes.

    This class is a mapping from modes to representations.
    """

    def __xp__(self, api_version: str | None = None):
        """Get the array API namespace of the entries."""
        return xpc.get_namespace(
            next(iter(self.values())).entries,
            api_version=api_version,
        )

    @property
    def domain(self):
        """Physical domain shared by all harmonics."""
        return self[next(iter(self))].domain


class HomogeneousHarmonicWaveform[ModeT: Mode, RepT: AnyReps](
    HarmonicWaveform[ModeT, RepT]
):
    """Harmonic waveform where all harmonics have the same grid."""

    def get_kernel(self):
        """Return an array of the conventional shape.

        The shape is ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        """
        xp = self.__xp__()
        return xp.concat(
            [self[harmonic].entries for harmonic in self.harmonics], axis=2
        )


class ProjectedWaveform[RepT: AnyReps](data._ChannelMapping[RepT]):  # pyright: ignore[reportPrivateUsage]
    """Projected waveform in a single mode or all modes summed together.

    This class is a mapping from channel names to representations. It is
    used to represent the detector response of a single harmonic waveform in different
    channels.
    """


class HarmonicProjectedWaveform[ModeT: Mode, RepT: AnyReps](
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


class HomogeneousHarmonicProjectedWaveform[ModeT: Mode, RepT: AnyReps](
    HarmonicProjectedWaveform[ModeT, RepT]
):
    """Harmonic projected waveform where all harmonics have the same grid."""

    def get_kernel(self):
        """Return an array of the conventional shape.

        The shape is ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        The returned array is suitable for downstream processing (e.g., by noise models to compute inner products).
        """
        xp = xpc.get_namespace(self._first.entries)
        return xp.concat(
            [self[harmonic].entries for harmonic in self.harmonics], axis=2
        )


def harmonic_waveform[ModeT: Mode, RepT: AnyReps](
    modes_to_representations: Mapping[ModeT, RepT],
) -> HarmonicWaveform[ModeT, RepT]:
    """Build a harmonic waveform from mode-indexed representations."""
    return HarmonicWaveform(modes_to_representations)


def homogeneous_harmonic_waveform[ModeT: Mode, RepT: AnyReps](
    modes_to_representations: Mapping[ModeT, RepT],
) -> HomogeneousHarmonicWaveform[ModeT, RepT]:
    """Build a homogeneous harmonic waveform from mode-indexed representations."""
    return HomogeneousHarmonicWaveform(modes_to_representations)


def projected_waveform[RepT: AnyReps](
    channels: Mapping[str, RepT],
) -> ProjectedWaveform[RepT]:
    """Build a projected waveform from channel-indexed representations."""
    return ProjectedWaveform[RepT].from_dict(channels)


def harmonic_projected_waveform[ModeT: Mode, RepT: AnyReps](
    modes_to_projected_waveforms: Mapping[ModeT, ProjectedWaveform[RepT]],
) -> HarmonicProjectedWaveform[ModeT, RepT]:
    """Build a harmonic projected waveform from mode-indexed projected waveforms."""
    return HarmonicProjectedWaveform(modes_to_projected_waveforms)


def homogeneous_harmonic_projected_waveform[ModeT: Mode, RepT: AnyReps](
    modes_to_projected_waveforms: Mapping[ModeT, ProjectedWaveform[RepT]],
) -> HomogeneousHarmonicProjectedWaveform[ModeT, RepT]:
    """Build a homogeneous harmonic projected waveform from mode-indexed projected waveforms."""
    return HomogeneousHarmonicProjectedWaveform(modes_to_projected_waveforms)


# Convenience aliases
hw = harmonic_waveform
hhw = homogeneous_harmonic_waveform
pw = projected_waveform
hpw = harmonic_projected_waveform
hhpw = homogeneous_harmonic_projected_waveform


def sum_harmonics[ModeT: Mode, AxisT: "Axis"](
    wf: HomogeneousHarmonicProjectedWaveform[ModeT, reps.FrequencySeries[AxisT]],
) -> ProjectedWaveform[reps.FrequencySeries[AxisT]]:
    """Sum over modes."""
    entries = wf.get_kernel().sum(axis=2, keepdims=True)  # c.f. shape convention
    return wf._first.create_like(entries, wf.channel_names)  # pyright: ignore[reportPrivateUsage]


def densify_phasor[AT: "Axis"](
    wf: reps.Phasor["Axis"],
    interpolator: utils.Interpolator,
    frequencies: AT,
    embed: bool = False,
) -> reps.Phasor[AT]:
    """Densify a phasor representation by interpolation.

    Parameters
    ----------
    wf : Phasor[Axis]
        The phasor representation to densify.
    interpolator : Interpolator
        The interpolator to use for densification.
    frequencies : Axis
        The frequencies at which to evaluate the densified phasor.
    embed : bool, optional
        Whether to embed the densified phasor on the original frequency grid.
    """
    _frequencies = reps.to_array(frequencies, xpc.get_namespace(wf.entries))

    _slice = utils.get_subset_slice(_frequencies, wf.f_min, wf.f_max)
    freqs = cast(
        AT,
        frequencies[utils.get_subset_slice(_frequencies, wf.f_min, wf.f_max)],
    )
    nwf = wf.get_interpolated(freqs, interpolator)
    if not embed:
        return nwf
    return nwf.get_embedded((frequencies,), known_slices=(_slice,))


def densify_phasor_hw[ModeT: Mode, AxisT: "Axis"](
    wf: HarmonicWaveform[ModeT, reps.Phasor["Axis"]],
    interpolator: utils.Interpolator,
    frequencies: AxisT,
    embed: bool = False,
) -> HomogeneousHarmonicWaveform[ModeT, reps.Phasor[AxisT]]:
    """Densify a harmonic waveform with phasor representations by interpolation.

    Parameters
    ----------
    wf : HarmonicWaveform[Mode, Phasor[Axis]]
        The harmonic waveform to densify.
    interpolator : Interpolator
        The interpolator to use for densification.
    frequencies : Axis
        The frequencies at which to evaluate the densified phasor.
    embed : bool, optional
        Whether to embed the densified phasor on the original frequency grid.
    """
    return homogeneous_harmonic_waveform(
        {
            mode: densify_phasor(wf[mode], interpolator, frequencies, embed)
            for mode in wf
        }
    )


def densify_phasor_pw[RepT: reps.Phasor["Axis"], AxisT: "Axis"](
    wf: ProjectedWaveform[RepT],
    interpolator: utils.Interpolator,
    frequencies: AxisT,
    embed: bool = False,
) -> ProjectedWaveform[reps.Phasor[AxisT]]:
    """Densify a projected waveform with phasor representations by interpolation.

    Parameters
    ----------
    wf : ProjectedWaveform[Phasor[Axis]]
        The projected waveform to densify.
    interpolator : Interpolator
        The interpolator to use for densification.
    frequencies : Axis
        The frequencies at which to evaluate the densified phasor.
    embed : bool, optional
        Whether to embed the densified phasor on the original frequency grid.
    """
    return projected_waveform(
        {
            chnname: densify_phasor(wf[chnname], interpolator, frequencies, embed)
            for chnname in wf.channel_names
        }
    )


def densify_phasor_hpw[ModeT: Mode, AxisT: "Axis"](
    wf: HarmonicProjectedWaveform[ModeT, reps.Phasor["Axis"]],
    interpolator: utils.Interpolator,
    frequencies: AxisT,
    embed: bool = False,
) -> HomogeneousHarmonicProjectedWaveform[ModeT, reps.Phasor[AxisT]]:
    """Densify a harmonic projected waveform with phasor representations by interpolation.

    Parameters
    ----------
    wf : HarmonicProjectedWaveform[Mode, Phasor[Axis]]
        The harmonic projected waveform to densify.
    interpolator : Interpolator
        The interpolator to use for densification.
    frequencies : Axis
        The frequencies at which to evaluate the densified phasor.
    embed : bool, optional
        Whether to embed the densified phasor on the original frequency grid.
    """
    return homogeneous_harmonic_projected_waveform(
        {
            mode: densify_phasor_pw(wf[mode], interpolator, frequencies, embed)
            for mode in wf
        }
    )


class _HWLike[ModeT: "Mode", RepT: "AnyReps"](Protocol):
    def __getitem__(self, key: ModeT) -> RepT: ...
    def __iter__(self) -> Iterator[ModeT]: ...


class _HHWLike[ModeT: "Mode", RepT: "AnyReps"](_HWLike[ModeT, RepT], Protocol):
    def get_kernel(self) -> Array: ...


class _PWLike[RepT: "AnyReps"](Protocol):
    @property
    def channel_names(self) -> tuple[str, ...]: ...

    def __getitem__(self, key: str) -> RepT: ...


class _HPWLike[ModeT: "Mode", RepT: "AnyReps"](Protocol):
    def __getitem__(self, key: ModeT) -> ProjectedWaveform[RepT]: ...
    def __iter__(self) -> Iterator[ModeT]: ...


class _HHPWLike[ModeT: "Mode", RepT: "AnyReps"](_HPWLike[ModeT, RepT], Protocol):
    def get_kernel(self) -> Array: ...


@overload
def phasor_to_fs_hw[MT: Mode, AxisT: "Axis"](
    wf: _HHWLike[MT, reps.Phasor[AxisT]],
) -> HomogeneousHarmonicWaveform[MT, reps.FrequencySeries[AxisT]]: ...


@overload
def phasor_to_fs_hw[MT: Mode, AxisT: "Axis"](
    wf: _HWLike[MT, reps.Phasor[AxisT]],
) -> HarmonicWaveform[MT, reps.FrequencySeries[AxisT]]: ...


def phasor_to_fs_hw[MT: Mode, AxisT: "Axis"](
    wf: _HWLike[MT, reps.Phasor[AxisT]],
):
    """Convert phasor leaves in a harmonic waveform to frequency-series leaves."""
    _mapping = {mode: wf[mode].to_frequency_series() for mode in wf}
    if isinstance(wf, HomogeneousHarmonicWaveform):
        return homogeneous_harmonic_waveform(_mapping)
    return harmonic_waveform(_mapping)


def phasor_to_fs_pw[AxisT: "Axis"](
    wf: _PWLike[reps.Phasor[AxisT]],
):
    """Convert phasor leaves in a projected waveform to frequency-series leaves."""
    return projected_waveform(
        {chnname: wf[chnname].to_frequency_series() for chnname in wf.channel_names}
    )


@overload
def phasor_to_fs_hpw[MT: Mode, AxisT: "Axis"](
    wf: _HHPWLike[MT, reps.Phasor[AxisT]],
) -> HomogeneousHarmonicProjectedWaveform[MT, reps.FrequencySeries[AxisT]]: ...


@overload
def phasor_to_fs_hpw[MT: Mode, AxisT: "Axis"](
    wf: _HPWLike[MT, reps.Phasor[AxisT]],
) -> HomogeneousHarmonicProjectedWaveform[MT, reps.FrequencySeries[AxisT]]: ...


def phasor_to_fs_hpw[MT: Mode, AxisT: "Axis"](
    wf: _HPWLike[MT, reps.Phasor[AxisT]],
):
    """Convert phasor leaves in a harmonic projected waveform to frequency-series leaves."""
    _mapping = {mode: phasor_to_fs_pw(wf[mode]) for mode in wf}

    if isinstance(wf, HomogeneousHarmonicProjectedWaveform):
        return homogeneous_harmonic_projected_waveform(_mapping)
    return harmonic_projected_waveform(_mapping)


# To deprecate
def get_dense_maker(
    interpolator: utils.Interpolator,
):
    """Return a function to convert a sparse phasor projected waveform to a dense phasor projected waveform.

    The returned function has the signature:

    .. code-block:: python

        def make(
            frequencies: npt.NDArray[np.floating],
            embed: bool = False,
        ) -> Callable[HarmonicProjectedWaveform[Mode, Phasor], HarmonicProjectedWaveform[Mode, Phasor]]:
            ...

    The function takes a list of frequencies and an optional boolean
    `embed` argument. If `embed` is True, the returned function will
    return a waveform with the same frequencies as the input. If `embed`
    is False, the returned function will return a waveform with the
    frequencies truncated to the lowest and highest frequencies of the
    input waveform.
    """

    def make[MT: Mode, AxisT: "Axis"](
        frequencies: AxisT,
        embed: bool = False,
    ) -> Callable[
        [HarmonicProjectedWaveform[MT, reps.Phasor["Axis"]]],
        HarmonicProjectedWaveform[MT, reps.Phasor[AxisT]],
    ]:

        def do_phasor(wf: reps.Phasor["Axis"]):
            _frequencies = reps.to_array(frequencies, xpc.get_namespace(wf.entries))

            _slice = utils.get_subset_slice(_frequencies, wf.f_min, wf.f_max)
            freqs = cast(
                AxisT,
                frequencies[utils.get_subset_slice(_frequencies, wf.f_min, wf.f_max)],
            )
            nwf = wf.get_interpolated(freqs, interpolator)
            if not embed:
                return nwf
            return nwf.get_embedded((frequencies,), known_slices=(_slice,))

        def do_response(resp: ProjectedWaveform[reps.Phasor["Axis"]]):
            return ProjectedWaveform[reps.Phasor[AxisT]].from_dict(
                {chnname: do_phasor(resp[chnname]) for chnname in resp.channel_names}
            )

        def do[ModeT: Mode](wf: HarmonicProjectedWaveform[ModeT, reps.Phasor["Axis"]]):
            return HarmonicProjectedWaveform[ModeT, reps.Phasor[AxisT]](
                {mode: do_response(wf[mode]) for mode in wf.harmonics}
            )

        return do

    return make
