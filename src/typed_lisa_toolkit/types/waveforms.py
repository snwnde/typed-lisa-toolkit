"""Waveform types."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Protocol,
    cast,
    overload,
)

import array_api_compat as xpc

from . import _mixins, modes
from . import representations as reps
from .misc import AnyGrid, Array, Axis, Interpolator

Mode = modes.Harmonic | modes.QNM

if TYPE_CHECKING:
    AnyReps = reps.Representation[AnyGrid]

from .. import utils

log = logging.getLogger(__name__)


def _validate_maps_to_pws(mapping: Mapping[Any, ProjectedWaveform[AnyReps]]):
    """Validate that a mapping maps to projected waveforms."""
    for key, pw in mapping.items():
        try:
            _ = _mixins.validate_maps_to_reps(pw)
        except ValueError as error:
            raise ValueError(f"Invalid projected waveform for key {key!r}.") from error


class HarmonicWaveform[ModeT: Mode, RepT: "AnyReps"](_mixins.ModeMapping[ModeT, RepT]):
    """Multi-mode waveform.

    Note
    ----
    To build a :class:`~types.HarmonicWaveform`, use the factory functions:
    :func:`~typed_lisa_toolkit.harmonic_waveform` or :func:`~typed_lisa_toolkit.hw`.
    """


class HomogeneousHarmonicWaveform[ModeT: Mode, RepT: "AnyReps"](
    HarmonicWaveform[ModeT, RepT]
):
    """Multi-mode waveform where all modes share the same grid.

    Note
    ----
    To build a :class:`~types.HomogeneousHarmonicWaveform`, use the factory functions:
    :func:`~typed_lisa_toolkit.homogeneous_harmonic_waveform`
    or :func:`~typed_lisa_toolkit.hhw`.
    """

    def get_kernel(self):
        """Return an array of the conventional shape.

        The shape is ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        """
        xp = self.__xp__()
        return xp.concat(
            [self[harmonic].entries for harmonic in self.harmonics], axis=2
        )


class PlusCrossWaveform[RepT: "AnyReps"](_mixins.ChannelMapping[RepT]):
    """Waveform in plus and cross polarizations.

    Note
    ----
    To build a :class:`~types.PlusCrossWaveform`, use the factory function
    :func:`~typed_lisa_toolkit.plus_cross_waveform`
    or :func:`~typed_lisa_toolkit.pcw`.
    """

    @property
    def plus(self) -> RepT:
        """Plus channel as a view with the channel dimension."""
        return self["plus"]

    @property
    def cross(self) -> RepT:
        """Cross channel as a view with the channel dimension."""
        return self["cross"]

    def __getitem__(self, key: str) -> RepT:
        """Get the plus or cross channel as a view with the chosen channel dimension."""
        return self._mapping[key]

    @property
    def kind(self) -> str | None:
        """Semantic kind."""
        return self[next(iter(self))].kind


class ProjectedWaveform[RepT: "AnyReps"](_mixins.ChannelMapping[RepT]):
    """Single-mode or mode-summed waveform projected onto the detector response in different channels.

    Note
    ----
    To build a :class:`~types.ProjectedWaveform`, use the factory function:
    :func:`~typed_lisa_toolkit.projected_waveform`
    or :func:`~typed_lisa_toolkit.pw`.
    """

    def __getitem__(self, key: str) -> RepT:
        """Get a channel by name as a view with the chosen channel dimension."""
        return self._mapping[key]

    @property
    def kind(self) -> str | None:
        """Semantic kind."""
        return self[next(iter(self))].kind


class HarmonicProjectedWaveform[ModeT: Mode, RepT: "AnyReps"](
    _mixins.ModeMapping[ModeT, ProjectedWaveform[RepT]]
):
    """Multi-mode waveform projected onto the detector response in different channels.

    Note
    ----
    To build a :class:`~types.HarmonicProjectedWaveform`, use the factory function:
    :func:`~typed_lisa_toolkit.harmonic_projected_waveform`
    or :func:`~typed_lisa_toolkit.hpw`.
    """

    @property
    def _first(self):
        return self[next(iter(self))]

    @property
    def channel_names(self) -> tuple[str, ...]:
        """All channel names."""
        return tuple(self._first.keys())


class HomogeneousHarmonicProjectedWaveform[ModeT: Mode, RepT: "AnyReps"](
    HarmonicProjectedWaveform[ModeT, RepT]
):
    """Multi-mode waveform where all modes share the same grid, projected onto the detector response in different channels.

    Note
    ----
    To build a :class:`~types.HomogeneousHarmonicProjectedWaveform`, use the factory function:
    :func:`~typed_lisa_toolkit.homogeneous_harmonic_projected_waveform`
    or :func:`~typed_lisa_toolkit.hhpw`.
    """

    def get_kernel(self):
        """Return an array of the conventional shape.

        The shape is ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        The returned array is suitable for downstream processing (e.g., by noise models to compute inner products).
        """
        xp = xpc.get_namespace(self._first.get_kernel())
        return xp.concat(
            [self[harmonic].get_kernel() for harmonic in self.harmonics], axis=2
        )


def harmonic_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_reps: Mapping[ModeT, RepT],
) -> HarmonicWaveform[ModeT, RepT]:
    """Build a :class:`~types.HarmonicWaveform`.

    Parameters
    ----------
    modes_to_reps :
        A mapping from :ref:`modes <mode_types>`
        to :ref:`representations <representation_types>`.
    """
    _ = _mixins.validate_maps_to_reps(modes_to_reps)
    return HarmonicWaveform(modes_to_reps)


def homogeneous_harmonic_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_reps: Mapping[ModeT, RepT],
) -> HomogeneousHarmonicWaveform[ModeT, RepT]:
    """Build a :class:`~types.HomogeneousHarmonicWaveform`.

    Parameters
    ----------
    modes_to_reps :
        A mapping from :ref:`modes <mode_types>`
        to :ref:`representations <representation_types>`.
    """
    _ = _mixins.validate_maps_to_reps(modes_to_reps)
    return HomogeneousHarmonicWaveform(modes_to_reps)


def plus_cross_waveform[RepT: "AnyReps"](
    pol_to_reps: Mapping[str, RepT],
) -> PlusCrossWaveform[RepT]:
    """Build a :class:`~types.PlusCrossWaveform`.

    Parameters
    ----------
    pol_to_reps :
        A mapping from polarization names (:py:class:`str`) to
        :ref:`representations <representation_types>`.
    """
    _ = _mixins.validate_maps_to_reps(pol_to_reps)
    return PlusCrossWaveform[RepT].from_dict(pol_to_reps)


def projected_waveform[RepT: "AnyReps"](
    channels_to_reps: Mapping[str, RepT],
) -> ProjectedWaveform[RepT]:
    """Build a :class:`~types.ProjectedWaveform`.

    Parameters
    ----------
    channels_to_reps :
        A mapping from channel names (:py:class:`str`) to
        :ref:`representations <representation_types>`.
    """
    _ = _mixins.validate_maps_to_reps(channels_to_reps)
    return ProjectedWaveform[RepT].from_dict(channels_to_reps)


def harmonic_projected_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_pws: Mapping[ModeT, ProjectedWaveform[RepT]],
) -> HarmonicProjectedWaveform[ModeT, RepT]:
    """Build a :class:`~types.HarmonicProjectedWaveform`.

    Parameters
    ----------
    modes_to_pws :
        A mapping from :ref:`modes <mode_types>`
        to :class:`~types.ProjectedWaveform` instances.
    """
    _ = _validate_maps_to_pws(modes_to_pws)
    return HarmonicProjectedWaveform(modes_to_pws)


def homogeneous_harmonic_projected_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_pws: Mapping[ModeT, ProjectedWaveform[RepT]],
) -> HomogeneousHarmonicProjectedWaveform[ModeT, RepT]:
    """Build a :class:`~types.HomogeneousHarmonicProjectedWaveform`.

    Parameters
    ----------
    modes_to_pws :
        A mapping from :ref:`modes <mode_types>`
        to :class:`~types.ProjectedWaveform` instances.
    """
    _ = _validate_maps_to_pws(modes_to_pws)
    return HomogeneousHarmonicProjectedWaveform(modes_to_pws)


# Convenience aliases
hw = harmonic_waveform
"""Alias for :func:`~harmonic_waveform`."""
hhw = homogeneous_harmonic_waveform
"""Alias for :func:`~homogeneous_harmonic_waveform`."""
pcw = plus_cross_waveform
"""Alias for :func:`~plus_cross_waveform`."""
pw = projected_waveform
"""Alias for :func:`~projected_waveform`."""
hpw = harmonic_projected_waveform
"""Alias for :func:`~harmonic_projected_waveform`."""
hhpw = homogeneous_harmonic_projected_waveform
"""Alias for :func:`~homogeneous_harmonic_projected_waveform`."""


def sum_harmonics[ModeT: Mode, AxisT: "Axis"](
    wf: HomogeneousHarmonicProjectedWaveform[ModeT, reps.FrequencySeries[AxisT]],
) -> ProjectedWaveform[reps.FrequencySeries[AxisT]]:
    """Sum over modes."""
    entries = wf.get_kernel().sum(axis=2, keepdims=True)  # c.f. shape convention
    _first = wf._first  # pyright: ignore[reportPrivateUsage]
    return type(_first)(
        _first.grid,
        entries,
        wf.channel_names,
        _rep_type=_first._rep_type,  # pyright: ignore[reportPrivateUsage]
    )


def densify_phasor_hw[ModeT: Mode, AxisT: "Axis"](
    wf: HarmonicWaveform[ModeT, reps.Phasor["Axis"]],
    interpolator: Interpolator,
    frequencies: AxisT,
    embed: bool = False,
) -> HomogeneousHarmonicWaveform[ModeT, reps.Phasor[AxisT]]:
    """Densify :class:`~types.HarmonicWaveform` with sparse :class:`~types.Phasor` by interpolation.

    Parameters
    ----------
    wf :
        The harmonic waveform to densify.
    interpolator :
        The interpolator to use for densification.
    frequencies :
        The frequencies at which to evaluate the densified phasor.
    embed :
        Whether to embed the densified phasor on the original frequency grid.
    """
    return homogeneous_harmonic_waveform(
        {
            mode: reps.densify_phasor(wf[mode], interpolator, frequencies, embed)
            for mode in wf
        }
    )


def densify_phasor_pw[RepT: reps.Phasor["Axis"], AxisT: "Axis"](
    wf: ProjectedWaveform[RepT],
    interpolator: Interpolator,
    frequencies: AxisT,
    embed: bool = False,
) -> ProjectedWaveform[reps.Phasor[AxisT]]:
    """Densify :class:`~types.ProjectedWaveform` with sparse :class:`~types.Phasor` representations by interpolation.

    Parameters
    ----------
    wf :
        The projected waveform to densify.
    interpolator :
        The interpolator to use for densification.
    frequencies :
        The frequencies at which to evaluate the densified phasor.
    embed :
        Whether to embed the densified phasor on the original frequency grid.
    """
    return projected_waveform(
        {
            chnname: reps.densify_phasor(wf[chnname], interpolator, frequencies, embed)
            for chnname in wf.channel_names
        }
    )


def densify_phasor_hpw[ModeT: Mode, AxisT: "Axis"](
    wf: HarmonicProjectedWaveform[ModeT, reps.Phasor["Axis"]],
    interpolator: Interpolator,
    frequencies: AxisT,
    embed: bool = False,
) -> HomogeneousHarmonicProjectedWaveform[ModeT, reps.Phasor[AxisT]]:
    """Densify :class:`~types.HarmonicProjectedWaveform` with sparse :class:`~types.Phasor` representations by interpolation."""
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
    """Convert :class:`~types.Phasor`-valued :class:`~types.HarmonicWaveform` to :class:`~types.FrequencySeries`-valued :class:`~types.HarmonicWaveform`."""
    _mapping = {mode: wf[mode].to_frequency_series() for mode in wf}
    if isinstance(wf, HomogeneousHarmonicWaveform):
        return homogeneous_harmonic_waveform(_mapping)
    return harmonic_waveform(_mapping)


def phasor_to_fs_pw[AxisT: "Axis"](
    wf: _PWLike[reps.Phasor[AxisT]],
):
    """Convert :class:`~types.Phasor`-valued :class:`~types.ProjectedWaveform` to :class:`~types.FrequencySeries`-valued :class:`~types.ProjectedWaveform`."""
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
    """Convert :class:`~types.Phasor`-valued :class:`~types.HarmonicProjectedWaveform` to :class:`~types.FrequencySeries`-valued :class:`~types.HarmonicProjectedWaveform`."""
    _mapping = {mode: phasor_to_fs_pw(wf[mode]) for mode in wf}

    if isinstance(wf, HomogeneousHarmonicProjectedWaveform):
        return homogeneous_harmonic_projected_waveform(_mapping)
    return harmonic_projected_waveform(_mapping)


# To deprecate
def get_dense_maker(
    interpolator: Interpolator,
):
    """Return a function to convert a sparse phasor projected waveform to a dense phasor projected waveform (*Deprecated*).

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

    .. warning::

        This function is deprecated in v0.6.0 and will be removed in v0.8.0.
        Use :func:`~densify_phasor`, :func:`~densify_phasor_hw`, :func:`~densify_phasor_pw`,
        or :func:`~densify_phasor_hpw` instead.
    """

    def make[MT: Mode, AxisT: "Axis"](
        frequencies: AxisT,
        embed: bool = False,
    ) -> Callable[
        [HarmonicProjectedWaveform[MT, reps.Phasor["Axis"]]],
        HarmonicProjectedWaveform[MT, reps.Phasor[AxisT]],
    ]:

        def do_phasor(wf: reps.Phasor["Axis"]):
            _frequencies = _mixins.to_array(frequencies, xpc.get_namespace(wf.entries))

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
