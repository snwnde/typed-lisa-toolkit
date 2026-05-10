"""Waveform types."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    overload,
)

import array_api_compat as xpc

from .. import utils
from . import _mixins, modes
from . import representations as reps
from .misc import AnyAxis, AnyGrid, Interpolator
from .representations import FrequencyPhasor, FrequencySeries, TimePhasor, TimeSeries

Mode = tuple[int, int] | tuple[int, int, int]

if TYPE_CHECKING:
    AnyReps = reps.Representation[AnyGrid]


log = logging.getLogger(__name__)


def _validate_maps_to_pws(mapping: Mapping[Any, ProjectedWaveform[AnyReps]]):
    """Validate that a mapping maps to projected waveforms."""
    for key, pw in mapping.items():
        try:
            _ = _mixins.validate_maps_to_reps(pw)
        except ValueError as error:
            msg = f"Invalid projected waveform for key {key!r}. "
            raise ValueError(msg) from error


class HarmonicWaveform[ModeT: Mode, RepT: "AnyReps"](_mixins.ModeMapping[ModeT, RepT]):
    """Multi-mode waveform.

    Note
    ----
    To build a :class:`.HarmonicWaveform`, use the factory functions:
    :func:`~typed_lisa_toolkit.harmonic_waveform` or :func:`~typed_lisa_toolkit.hw`.
    """


class HomogeneousHarmonicWaveform[ModeT: Mode, RepT: "AnyReps"](
    HarmonicWaveform[ModeT, RepT],
):
    """Multi-mode waveform where all modes share the same grid.

    Note
    ----
    To build a :class:`.HomogeneousHarmonicWaveform`, use the factory functions:
    :func:`~typed_lisa_toolkit.homogeneous_harmonic_waveform`
    or :func:`~typed_lisa_toolkit.hhw`.
    """

    def get_kernel(self):
        """Return an array of the conventional shape.

        The shape is ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        """
        xp = self.__xp__()
        return xp.concat(
            [self[harmonic].entries for harmonic in self.harmonics],
            axis=2,
        )


class PlusCrossWaveform[RepT: "AnyReps"](_mixins.ChannelMapping[RepT]):
    """Waveform in plus and cross polarizations.

    Note
    ----
    To build a :class:`.PlusCrossWaveform`, use the factory function
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
    To build a :class:`.ProjectedWaveform`, use the factory function:
    :func:`~typed_lisa_toolkit.projected_waveform`
    or :func:`~typed_lisa_toolkit.pw`.
    """  # noqa: E501

    def __getitem__(self, key: str) -> RepT:
        """Get a channel by name as a view with the chosen channel dimension."""
        return self._mapping[key]

    @property
    def kind(self) -> str | None:
        """Semantic kind."""
        return self[next(iter(self))].kind


class HarmonicProjectedWaveform[ModeT: Mode, RepT: "AnyReps"](
    _mixins.ModeMapping[ModeT, ProjectedWaveform[RepT]],
):
    """Multi-mode waveform projected onto the detector response in different channels.

    Note
    ----
    To build a :class:`.HarmonicProjectedWaveform`, use the factory function:
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
    HarmonicProjectedWaveform[ModeT, RepT],
):
    """Multi-mode waveform where all modes share the same grid, projected onto the detector response in different channels.

    Note
    ----
    To build a :class:`.HomogeneousHarmonicProjectedWaveform`, use the factory function:
    :func:`~typed_lisa_toolkit.homogeneous_harmonic_projected_waveform`
    or :func:`~typed_lisa_toolkit.hhpw`.
    """  # noqa: E501

    def get_kernel(self):
        """Return an array of the conventional shape.

        The shape is ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        The returned array is suitable for downstream processing
        (e.g., by noise models to compute inner products).
        """
        xp = xpc.get_namespace(self._first.get_kernel())
        return xp.concat(
            [self[harmonic].get_kernel() for harmonic in self.harmonics],
            axis=2,
        )


@overload
def harmonic_waveform[RepT: "AnyReps"](
    modes_to_reps: Mapping[tuple[int, int], RepT],
) -> HarmonicWaveform[modes.Harmonic, RepT]: ...


@overload
def harmonic_waveform[RepT: "AnyReps"](
    modes_to_reps: Mapping[tuple[int, int, int], RepT],
) -> HarmonicWaveform[modes.QuasiNormalMode, RepT]: ...


@overload
def harmonic_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_reps: Mapping[ModeT, RepT],
) -> HarmonicWaveform[ModeT, RepT]: ...


def harmonic_waveform[RepT: "AnyReps"](
    modes_to_reps: Mapping[Any, RepT],
):
    """Build a :class:`~types.HarmonicWaveform`.

    Parameters
    ----------
    modes_to_reps :
        A mapping from :ref:`modes <mode_types>`
        to :ref:`representations <representation_types>`.


    See Also
    --------
    :func:`~typed_lisa_toolkit.hw`
        A convenience alias.


    Example
    -------
    .. code-block:: python

        import jax.numpy as jnp
        import typed_lisa_toolkit as tlt

        ts = tlt.time_series(
            tlt.linspace(0, 1, 10), jnp.ones((1, 1, 1, 1, 10))
        )  # UniformTimeSeries
        hw = tlt.harmonic_waveform(
            {(2, 2): ts}
        )  # HarmonicWaveform[Harmonic, UniformTimeSeries]

    """
    _ = _mixins.validate_maps_to_reps(modes_to_reps)
    return HarmonicWaveform[Any, RepT](modes_to_reps, cast_mode=True)


@overload
def homogeneous_harmonic_waveform[RepT: "AnyReps"](
    modes_to_reps: Mapping[tuple[int, int], RepT],
) -> HomogeneousHarmonicWaveform[modes.Harmonic, RepT]: ...


@overload
def homogeneous_harmonic_waveform[RepT: "AnyReps"](
    modes_to_reps: Mapping[tuple[int, int, int], RepT],
) -> HomogeneousHarmonicWaveform[modes.QuasiNormalMode, RepT]: ...


@overload
def homogeneous_harmonic_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_reps: Mapping[ModeT, RepT],
) -> HomogeneousHarmonicWaveform[ModeT, RepT]: ...


def homogeneous_harmonic_waveform[RepT: "AnyReps"](
    modes_to_reps: Mapping[Any, RepT],
):
    """Build a :class:`~types.HomogeneousHarmonicWaveform`.

    Parameters
    ----------
    modes_to_reps :
        A mapping from :ref:`modes <mode_types>`
        to :ref:`representations <representation_types>`.
    """
    _ = _mixins.validate_maps_to_reps(modes_to_reps)
    return HomogeneousHarmonicWaveform[Any, RepT](modes_to_reps, cast_mode=True)


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


@overload
def harmonic_projected_waveform[RepT: "AnyReps"](
    modes_to_pws: Mapping[tuple[int, int], ProjectedWaveform[RepT]],
) -> HarmonicProjectedWaveform[modes.Harmonic, RepT]: ...


@overload
def harmonic_projected_waveform[RepT: "AnyReps"](
    modes_to_pws: Mapping[tuple[int, int, int], ProjectedWaveform[RepT]],
) -> HarmonicProjectedWaveform[modes.QuasiNormalMode, RepT]: ...


@overload
def harmonic_projected_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_pws: Mapping[ModeT, ProjectedWaveform[RepT]],
) -> HarmonicProjectedWaveform[ModeT, RepT]: ...


def harmonic_projected_waveform[RepT: "AnyReps"](
    modes_to_pws: Mapping[Any, ProjectedWaveform[RepT]],
):
    """Build a :class:`~types.HarmonicProjectedWaveform`.

    Parameters
    ----------
    modes_to_pws :
        A mapping from :ref:`modes <mode_types>`
        to :class:`~types.ProjectedWaveform` instances.
    """
    _ = _validate_maps_to_pws(modes_to_pws)
    return HarmonicProjectedWaveform[Any, RepT](modes_to_pws, cast_mode=True)


@overload
def homogeneous_harmonic_projected_waveform[RepT: "AnyReps"](
    modes_to_pws: Mapping[tuple[int, int], ProjectedWaveform[RepT]],
) -> HomogeneousHarmonicProjectedWaveform[modes.Harmonic, RepT]: ...


@overload
def homogeneous_harmonic_projected_waveform[RepT: "AnyReps"](
    modes_to_pws: Mapping[tuple[int, int, int], ProjectedWaveform[RepT]],
) -> HomogeneousHarmonicProjectedWaveform[modes.QuasiNormalMode, RepT]: ...


@overload
def homogeneous_harmonic_projected_waveform[ModeT: Mode, RepT: "AnyReps"](
    modes_to_pws: Mapping[ModeT, ProjectedWaveform[RepT]],
) -> HomogeneousHarmonicProjectedWaveform[ModeT, RepT]: ...


def homogeneous_harmonic_projected_waveform[RepT: "AnyReps"](
    modes_to_pws: Mapping[Any, ProjectedWaveform[RepT]],
):
    """Build a :class:`~types.HomogeneousHarmonicProjectedWaveform`.

    Parameters
    ----------
    modes_to_pws :
        A mapping from :ref:`modes <mode_types>`
        to :class:`~types.ProjectedWaveform` instances.
    """
    _ = _validate_maps_to_pws(modes_to_pws)
    return HomogeneousHarmonicProjectedWaveform[Any, RepT](modes_to_pws, cast_mode=True)


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


def sum_harmonics[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HomogeneousHarmonicProjectedWaveform[ModeT, FrequencySeries[AxisT]],
) -> ProjectedWaveform[FrequencySeries[AxisT]]:
    """Sum over modes."""
    entries = wf.get_kernel().sum(axis=2, keepdims=True)  # c.f. shape convention
    _first = wf._first  # pyright: ignore[reportPrivateUsage]
    return type(_first)(
        _first.grid,
        entries,
        wf.channel_names,
        _rep_type=_first._rep_type,  # pyright: ignore[reportPrivateUsage]
    )


_PhasorWaveTypes = (
    HomogeneousHarmonicWaveform[Mode, FrequencyPhasor[AnyAxis]]
    | HarmonicWaveform[Mode, FrequencyPhasor[AnyAxis]]
    | HomogeneousHarmonicWaveform[Mode, TimePhasor[AnyAxis]]
    | HarmonicWaveform[Mode, TimePhasor[AnyAxis]]
    | ProjectedWaveform[FrequencyPhasor[AnyAxis]]
    | ProjectedWaveform[TimePhasor[AnyAxis]]
    | HomogeneousHarmonicProjectedWaveform[Mode, FrequencyPhasor[AnyAxis]]
    | HarmonicProjectedWaveform[Mode, FrequencyPhasor[AnyAxis]]
    | HomogeneousHarmonicProjectedWaveform[Mode, TimePhasor[AnyAxis]]
    | HarmonicProjectedWaveform[Mode, TimePhasor[AnyAxis]]
)


@overload
def densify_phasor[AT: "AnyAxis"](
    wf: TimePhasor[AnyAxis],
    /,
    interpolator: Interpolator,
    axis: AT,
    *,
    embed: bool = False,
) -> TimePhasor[AT]: ...
@overload
def densify_phasor[AT: "AnyAxis"](
    wf: FrequencyPhasor[AnyAxis],
    /,
    interpolator: Interpolator,
    axis: AT,
    *,
    embed: bool = False,
) -> FrequencyPhasor[AT]: ...
@overload
def densify_phasor[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[ModeT, TimePhasor[AnyAxis]],
    /,
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicWaveform[ModeT, TimePhasor[AxisT]]: ...
@overload
def densify_phasor[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[ModeT, FrequencyPhasor[AnyAxis]],
    /,
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicWaveform[ModeT, FrequencyPhasor[AxisT]]: ...
@overload
def densify_phasor[RepT: TimePhasor["AnyAxis"], AxisT: "AnyAxis"](
    wf: ProjectedWaveform[RepT],
    /,
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> ProjectedWaveform[TimePhasor[AxisT]]: ...
@overload
def densify_phasor[RepT: FrequencyPhasor["AnyAxis"], AxisT: "AnyAxis"](
    wf: ProjectedWaveform[RepT],
    /,
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> ProjectedWaveform[FrequencyPhasor[AxisT]]: ...
@overload
def densify_phasor[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[ModeT, TimePhasor[AnyAxis]],
    /,
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicProjectedWaveform[ModeT, TimePhasor[AxisT]]: ...
@overload
def densify_phasor[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[ModeT, FrequencyPhasor[AnyAxis]],
    /,
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicProjectedWaveform[ModeT, FrequencyPhasor[AxisT]]: ...


def densify_phasor[AT: "AnyAxis"](
    wf: FrequencyPhasor[AnyAxis] | TimePhasor[AnyAxis] | _PhasorWaveTypes,
    /,
    interpolator: Interpolator,
    axis: AT,
    *,
    embed: bool = False,
    frequencies: AT | None = None,
):
    """Densify a sparse phasor representation by interpolation.

    Parameters
    ----------
    wf :
        The phasor representation or waveform to densify.
    interpolator :
        The interpolator to use for densification.
    axis :
        The axis at which to evaluate the densified phasor.
    embed :
        If False, the returned phasor is restricted to the subset of `axis`
        that overlaps with the support of `wf`.

    Attention
    ---------
    The branch with `embed=False` does not support JIT compilation.
    """
    if frequencies is not None:
        msg = (
            "The `frequencies` argument of `densify_phasor` is deprecated "
            "and will be removed in 0.8.0; "
            "pass the frequencies as the `axis` argument instead."
        )
        utils.warn_external(
            msg,
            DeprecationWarning,
        )
        axis = frequencies

    if isinstance(wf, (TimePhasor, FrequencyPhasor)):
        xp = xpc.get_namespace(wf.entries)
        _axis = _mixins.to_array(axis, xp=xp)
        if not embed:
            _slice = utils.get_subset_slice(_axis, wf.axis_onset, wf.axis_end)
            freqs = axis[_slice]
            return wf.get_interpolated(freqs, interpolator)
        mask = utils.get_subset_mask(_axis, wf.axis_onset, wf.axis_end)
        nwf = wf.get_interpolated(axis, interpolator)
        _amp = xp.where(mask, nwf.amplitudes, 0)
        _phase = xp.where(mask, nwf.phases, 0)
        if isinstance(wf, TimePhasor):
            return reps.time_phasor(times=axis, amplitudes=_amp, phases=_phase)
        return reps.frequency_phasor(frequencies=axis, amplitudes=_amp, phases=_phase)
    if isinstance(wf, HarmonicWaveform):
        return hhw(
            {
                mode: densify_phasor(
                    wf[mode],
                    interpolator,
                    axis,
                    embed=embed,
                )
                for mode in wf
            },
        )
    if isinstance(wf, ProjectedWaveform):
        return pw(
            {
                chnname: densify_phasor(
                    wf[chnname],
                    interpolator,
                    axis,
                    embed=embed,
                )
                for chnname in wf.channel_names
            },
        )
    # Must be a HarmonicProjectedWaveform at this point
    return hhpw(
        {
            mode: projected_waveform(
                {
                    chnname: densify_phasor(
                        wf[mode][chnname],
                        interpolator,
                        axis,
                        embed=embed,
                    )
                    for chnname in wf._first.channel_names  # pyright: ignore[reportPrivateUsage]
                },
            )
            for mode in wf.harmonics
        },
    )

    # if embed:
    #     freqs = xp.where(mask, _frequencies, 0)
    #     nwf = wf.get_interpolated(freqs, interpolator)
    #     return nwf
    # freqs = _frequencies[mask]
    # nwf = wf.get_interpolated(freqs, interpolator)
    # return nwf

    # _slice = utils.get_subset_slice(_frequencies, wf.f_min, wf.f_max)
    # freqs = frequencies[_slice]
    # nwf = wf.get_interpolated(freqs, interpolator)
    # if not embed:
    #     return nwf
    # return nwf.get_embedded((frequencies,), known_slices=(_slice,))


@overload
def densify_phasor_hw[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[ModeT, TimePhasor[AnyAxis]],
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicWaveform[ModeT, TimePhasor[AxisT]]: ...
@overload
def densify_phasor_hw[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[ModeT, FrequencyPhasor[AnyAxis]],
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicWaveform[ModeT, FrequencyPhasor[AxisT]]: ...


@utils.deprecated(
    "densify_phasor_hw", "function", "0.8.0", alternative="densify_phasor"
)
def densify_phasor_hw[ModeT: Mode](
    wf: HarmonicWaveform[ModeT, FrequencyPhasor[AnyAxis]]
    | HarmonicWaveform[ModeT, TimePhasor[AnyAxis]],
    interpolator: Interpolator,
    axis: AnyAxis,
    *,
    embed: bool = False,
):
    """Densify :class:`~types.HarmonicWaveform` with sparse :class:`~types.TimePhasor` or :class:`~types.FrequencyPhasor` by interpolation (*Deprecated*).

    Parameters
    ----------
    wf :
        The harmonic waveform to densify.
    interpolator :
        The interpolator to use for densification.
    axis :
        The axis on which to interpolate.
    embed :
        Whether to embed the densified phasor on the original frequency grid.

    .. deprecated:: 0.6.6
        Will be removed in 0.8.0; use :func:`densify_phasor` instead.
    """  # noqa: E501
    return densify_phasor(wf, interpolator, axis, embed=embed)


@overload
def densify_phasor_pw[RepT: TimePhasor["AnyAxis"], AxisT: "AnyAxis"](
    wf: ProjectedWaveform[RepT],
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> ProjectedWaveform[TimePhasor[AxisT]]: ...
@overload
def densify_phasor_pw[RepT: FrequencyPhasor["AnyAxis"], AxisT: "AnyAxis"](
    wf: ProjectedWaveform[RepT],
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> ProjectedWaveform[FrequencyPhasor[AxisT]]: ...


@utils.deprecated(
    "densify_phasor_pw", "function", "0.8.0", alternative="densify_phasor"
)
def densify_phasor_pw[
    AxisT: "AnyAxis",
](
    wf: ProjectedWaveform[TimePhasor[AnyAxis]]
    | ProjectedWaveform[FrequencyPhasor[AnyAxis]],
    interpolator: Interpolator,
    axis: AnyAxis,
    *,
    embed: bool = False,
):
    """Densify :class:`~types.ProjectedWaveform` with sparse :class:`~types.TimePhasor` or :class:`~types.FrequencyPhasor` representations by interpolation (*Deprecated*).

    Parameters
    ----------
    wf :
        The projected waveform to densify.
    interpolator :
        The interpolator to use for densification.
    axis :
        The axis on which to interpolate.
    embed :
        Whether to embed the densified phasor on the original frequency grid.

    .. deprecated:: 0.6.6
        Will be removed in 0.8.0; use :func:`densify_phasor` instead.
    """  # noqa: E501
    return densify_phasor(wf, interpolator, axis, embed=embed)


@overload
def densify_phasor_hpw[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[ModeT, TimePhasor[AnyAxis]],
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicProjectedWaveform[ModeT, TimePhasor[AxisT]]: ...
@overload
def densify_phasor_hpw[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[ModeT, FrequencyPhasor[AnyAxis]],
    interpolator: Interpolator,
    axis: AxisT,
    *,
    embed: bool = False,
) -> HomogeneousHarmonicProjectedWaveform[ModeT, FrequencyPhasor[AxisT]]: ...


@utils.deprecated(
    "densify_phasor_hpw", "function", "0.8.0", alternative="densify_phasor"
)
def densify_phasor_hpw[ModeT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[ModeT, TimePhasor[AnyAxis]]
    | HarmonicProjectedWaveform[ModeT, FrequencyPhasor[AnyAxis]],
    interpolator: Interpolator,
    axis: AnyAxis,
    *,
    embed: bool = False,
):
    """Densify :class:`~types.HarmonicProjectedWaveform` with sparse :class:`~types.TimePhasor` or :class:`~types.FrequencyPhasor` representations by interpolation (*Deprecated*).

    .. deprecated:: 0.6.6
        Will be removed in 0.8.0; use :func:`densify_phasor` instead.
    """  # noqa: E501
    return densify_phasor(wf, interpolator, axis, embed=embed)


@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HomogeneousHarmonicWaveform[MT, FrequencyPhasor[AxisT]],
    /,
) -> HomogeneousHarmonicWaveform[MT, FrequencySeries[AxisT]]: ...
@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[MT, FrequencyPhasor[AxisT]],
    /,
) -> HarmonicWaveform[MT, FrequencySeries[AxisT]]: ...
@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HomogeneousHarmonicWaveform[MT, TimePhasor[AxisT]],
    /,
) -> HomogeneousHarmonicWaveform[MT, TimeSeries[AxisT]]: ...
@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[MT, TimePhasor[AxisT]],
    /,
) -> HarmonicWaveform[MT, TimeSeries[AxisT]]: ...
@overload
def phasor_to_series[AxisT: "AnyAxis"](
    wf: ProjectedWaveform[FrequencyPhasor[AxisT]],
    /,
) -> ProjectedWaveform[FrequencySeries[AxisT]]: ...
@overload
def phasor_to_series[AxisT: "AnyAxis"](
    wf: ProjectedWaveform[TimePhasor[AxisT]],
    /,
) -> ProjectedWaveform[TimeSeries[AxisT]]: ...
@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HomogeneousHarmonicProjectedWaveform[MT, FrequencyPhasor[AxisT]],
    /,
) -> HomogeneousHarmonicProjectedWaveform[MT, FrequencySeries[AxisT]]: ...
@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[MT, FrequencyPhasor[AxisT]],
    /,
) -> HarmonicProjectedWaveform[MT, FrequencySeries[AxisT]]: ...
@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HomogeneousHarmonicProjectedWaveform[MT, TimePhasor[AxisT]],
    /,
) -> HomogeneousHarmonicProjectedWaveform[MT, TimeSeries[AxisT]]: ...
@overload
def phasor_to_series[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[MT, TimePhasor[AxisT]],
    /,
) -> HarmonicProjectedWaveform[MT, TimeSeries[AxisT]]: ...


def phasor_to_series(
    wf: _PhasorWaveTypes,
    /,
):
    """Convert phasor-valued waveform to series-valued waveform."""
    if isinstance(wf, HarmonicWaveform):
        _mapping = {mode: wf[mode].to_series() for mode in wf}
        if isinstance(wf, HomogeneousHarmonicWaveform):
            return homogeneous_harmonic_waveform(_mapping)
        return harmonic_waveform(_mapping)

    if isinstance(wf, ProjectedWaveform):
        return projected_waveform(
            {chnname: wf[chnname].to_series() for chnname in wf.channel_names},
        )

    _mapping = {
        mode: projected_waveform(
            {
                chnname: wf[mode][chnname].to_series()
                for chnname in wf[mode].channel_names
            },
        )
        for mode in wf
    }
    if isinstance(wf, HomogeneousHarmonicProjectedWaveform):
        return homogeneous_harmonic_projected_waveform(_mapping)
    return harmonic_projected_waveform(_mapping)


@overload
def phasor_to_fs_hw[MT: Mode, AxisT: "AnyAxis"](
    wf: HomogeneousHarmonicWaveform[MT, FrequencyPhasor[AxisT]],
) -> HomogeneousHarmonicWaveform[MT, FrequencySeries[AxisT]]: ...


@overload
def phasor_to_fs_hw[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[MT, FrequencyPhasor[AxisT]],
) -> HarmonicWaveform[MT, FrequencySeries[AxisT]]: ...


@utils.deprecated(
    "phasor_to_fs_hw", "function", "0.8.0", alternative="phasor_to_series"
)
def phasor_to_fs_hw[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicWaveform[MT, FrequencyPhasor[AxisT]],
):
    """Convert :class:`~types.FrequencyPhasor`-valued :class:`~types.HarmonicWaveform` to :class:`~types.FrequencySeries` (*Deprecated*).

    .. deprecated:: 0.6.6
        Will be removed in 0.8.0. Use :func:`phasor_to_series` instead.
    """  # noqa: E501
    return phasor_to_series(wf)


@utils.deprecated(
    "phasor_to_fs_pw", "function", "0.8.0", alternative="phasor_to_series"
)
def phasor_to_fs_pw[AxisT: "AnyAxis"](
    wf: ProjectedWaveform[FrequencyPhasor[AxisT]],
):
    """Convert :class:`~types.FrequencyPhasor`-valued :class:`~types.ProjectedWaveform` to :class:`~types.FrequencySeries` (*Deprecated*).

    .. deprecated:: 0.6.6
        Will be removed in 0.8.0. Use :func:`phasor_to_series` instead.
    """  # noqa: E501
    return phasor_to_series(wf)


@overload
def phasor_to_fs_hpw[MT: Mode, AxisT: "AnyAxis"](
    wf: HomogeneousHarmonicProjectedWaveform[MT, FrequencyPhasor[AxisT]],
) -> HomogeneousHarmonicProjectedWaveform[MT, FrequencySeries[AxisT]]: ...


@overload
def phasor_to_fs_hpw[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[MT, FrequencyPhasor[AxisT]],
) -> HomogeneousHarmonicProjectedWaveform[MT, FrequencySeries[AxisT]]: ...


@utils.deprecated(
    "phasor_to_fs_hpw", "function", "0.8.0", alternative="phasor_to_series"
)
def phasor_to_fs_hpw[MT: Mode, AxisT: "AnyAxis"](
    wf: HarmonicProjectedWaveform[MT, FrequencyPhasor[AxisT]],
):
    """Convert :class:`~types.FrequencyPhasor`-valued :class:`~types.HarmonicProjectedWaveform` to :class:`~types.FrequencySeries` (*Deprecated*).

    .. deprecated:: 0.6.6
        Will be removed in 0.8.0. Use :func:`phasor_to_series` instead.
    """  # noqa: E501
    return phasor_to_series(wf)


@utils.deprecated("get_dense_maker", "function", "0.8.0")
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


    .. deprecated:: 0.6.0
        Will be removed in 0.8.0. Use :func:`densify_phasor`,
        :func:`densify_phasor_hw`, :func:`densify_phasor_pw`,
        or :func:`densify_phasor_hpw` instead.
    """  # noqa: E501

    def make[MT: Mode, AxisT: "AnyAxis"](
        frequencies: AxisT,
        *,
        embed: bool = False,
    ) -> Callable[
        [HarmonicProjectedWaveform[MT, FrequencyPhasor[AnyAxis]]],
        HarmonicProjectedWaveform[MT, FrequencyPhasor[AxisT]],
    ]:

        def do_phasor(wf: FrequencyPhasor[AnyAxis]):
            _frequencies = frequencies.asarray(xpc.get_namespace(wf.entries))

            _slice = utils.get_subset_slice(_frequencies, wf.f_min, wf.f_max)
            freqs = frequencies[
                utils.get_subset_slice(_frequencies, wf.f_min, wf.f_max)
            ]
            nwf = wf.get_interpolated(freqs, interpolator)
            if not embed:
                return nwf
            return nwf.get_embedded((frequencies,), known_slices=(_slice,))

        def do_response(resp: ProjectedWaveform[FrequencyPhasor[AnyAxis]]):
            return ProjectedWaveform[FrequencyPhasor[AxisT]].from_dict(
                {chnname: do_phasor(resp[chnname]) for chnname in resp.channel_names},
            )

        def do[ModeT: Mode](
            wf: HarmonicProjectedWaveform[ModeT, FrequencyPhasor[AnyAxis]],
        ):
            return HarmonicProjectedWaveform[ModeT, FrequencyPhasor[AxisT]](
                {mode: do_response(wf[mode]) for mode in wf.harmonics},
            )

        return do

    return make
