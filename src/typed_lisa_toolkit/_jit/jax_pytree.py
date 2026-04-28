"""JAX PyTree registration for typed_lisa_toolkit types."""

import functools
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import (
        AnyGrid,
        Array,
        Axis,
        AxLike,
        EvolutionarySpectralDensity,
        Linspace,
        SpectralDensity,
    )
    from ..types import representations as reps
    from ..types._mixins import ChannelMapping, Mode, ModeMapping

    AnyReps = reps.Representation[AnyGrid]


def _flatten_linspace(linspace: "Linspace"):
    return (), (type(linspace), linspace.start, linspace.step, linspace.num)


def _unflatten_linspace(
    aux: tuple[type["Linspace"], float, float, int], leaves: tuple[None]
) -> "Linspace":
    (linspace_type, start, step, num) = aux
    _ = leaves
    return linspace_type(start=start, step=step, num=num)


def _flatten_axis(axis: "Axis[AxLike]"):
    return (axis.ax,), (type(axis), axis.xp)


def _unflatten_axis(
    aux: tuple[type["Axis[AxLike]"], ModuleType], leaves: tuple["AxLike", ...]
) -> "Axis[AxLike]":
    (cls, xp) = aux
    (ax,) = leaves
    return cls(ax=ax, xp=xp)


def _flatten_representation(rep: "AnyReps"):
    rep_type = type(rep)
    return (rep.entries, rep.grid), (rep_type,)


def _unflatten_representation(
    aux: "tuple[type[AnyReps]]",
    leaves: "tuple[Array, AnyGrid]",
) -> Any:
    (rep_type,) = aux
    (entries, grid) = leaves
    return rep_type(grid=grid, entries=entries)


def _flatten_channel_mapping(
    obj: "ChannelMapping[AnyReps]",
):
    channel_names = obj.channel_names
    leaves = tuple(obj[channel] for channel in channel_names)
    kwargs: dict[str, Any] = {}
    # TimedFSData needs its associated time axis for reconstruction.
    if hasattr(obj, "times") and obj.domain == "frequency":
        kwargs["times"] = obj.times  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    obj_type = type(obj)
    return leaves, (obj_type, channel_names, obj.name, kwargs)


def _unflatten_channel_mapping(
    aux: tuple[
        type["ChannelMapping[AnyReps]"], tuple[str, ...], str | None, dict[str, Any]
    ],
    leaves: tuple["AnyReps", ...],
):
    cls, channel_names, name, kwargs = aux
    mapping = {channel: rep for channel, rep in zip(channel_names, leaves, strict=True)}
    if name is not None:
        kwargs = {**kwargs, "name": name}
    return cls.from_dict(mapping, **kwargs)


def _flatten_mode_mapping(
    obj: "ModeMapping[Mode, AnyReps]",
):
    mode_keys = tuple(obj.keys())
    values = tuple(obj.values())
    return values, (type(obj), mode_keys)


def _unflatten_mode_mapping(
    aux: tuple[type["ModeMapping[Mode, AnyReps]"], tuple["Mode", ...]],
    leaves: tuple["AnyReps", ...],
):
    cls, mode_keys = aux
    mapping = {mode: rep for mode, rep in zip(mode_keys, leaves, strict=True)}
    return cls(mapping)


def _flatten_spectral_density(sdm: "SpectralDensity"):
    return (sdm._frequencies, sdm._inverse_sdm), (type(sdm), sdm.channel_order)  # pyright: ignore[reportPrivateUsage]


def _unflatten_spectral_density(
    aux: tuple[type[Any], tuple[str, ...]],
    leaves: tuple[Any, ...],
):
    cls, channel_order = aux
    frequencies, inverse_sdm = leaves
    return cls(
        frequencies=frequencies, inverse_sdm=inverse_sdm, channel_order=channel_order
    )


def _flatten_evolutionary_spectral_density(sdm: "EvolutionarySpectralDensity"):
    return (
        sdm._times,  # pyright: ignore[reportPrivateUsage]
        sdm._frequencies,  # pyright: ignore[reportPrivateUsage]
        sdm._inverse_esdm,  # pyright: ignore[reportPrivateUsage]
    ), (type(sdm), sdm.channel_order)


def _unflatten_evolutionary_spectral_density(
    aux: tuple[type[Any], tuple[str, ...]],
    leaves: tuple[Any, ...],
):
    cls, channel_order = aux
    times, frequencies, inverse_esdm = leaves
    return cls(
        times=times,
        frequencies=frequencies,
        inverse_esdm=inverse_esdm,
        channel_order=channel_order,
    )


def _register_all(jtu: ModuleType) -> None:
    from .. import types

    jtu.register_pytree_node(
        types.Linspace,
        _flatten_linspace,
        _unflatten_linspace,
    )

    jtu.register_pytree_node(
        types.Axis,
        _flatten_axis,
        _unflatten_axis,
    )

    jtu.register_pytree_node(
        types.SpectralDensity,
        _flatten_spectral_density,
        _unflatten_spectral_density,
    )

    jtu.register_pytree_node(
        types.EvolutionarySpectralDensity,
        _flatten_evolutionary_spectral_density,
        _unflatten_evolutionary_spectral_density,
    )

    rep_types = (
        types.FrequencySeries,
        types.UniformFrequencySeries,
        types.TimeSeries,
        types.UniformTimeSeries,
        types.Phasor,
        types.STFT,
        types.WDM,
    )
    for rep_type in rep_types:
        jtu.register_pytree_node(
            rep_type,
            _flatten_representation,
            _unflatten_representation,
        )

    chan_map_types = (
        types.TSData,
        types.FSData,
        types.TimedFSData,
        types.STFTData,
        types.WDMData,
        types.ProjectedWaveform,
    )
    for chan_map_type in chan_map_types:
        jtu.register_pytree_node(
            chan_map_type,
            _flatten_channel_mapping,
            _unflatten_channel_mapping,
        )

    mode_map_types = (
        types.HarmonicWaveform,
        types.HomogeneousHarmonicWaveform,
        types.HarmonicProjectedWaveform,
        types.HomogeneousHarmonicProjectedWaveform,
    )
    for mode_map_type in mode_map_types:
        jtu.register_pytree_node(
            mode_map_type,
            _flatten_mode_mapping,
            _unflatten_mode_mapping,
        )


@functools.lru_cache(maxsize=1)
def _register_once() -> bool:
    try:
        from jax import tree_util as jtu
    except ImportError:
        return False

    _register_all(jtu)
    return True


def enable_jax_pytree_registration() -> bool:
    """Register JAX PyTree handlers for TLT types when JAX is installed.

    Returns ``True`` when registration has run in this process and ``False``
    when JAX is not available.
    """
    return _register_once()
