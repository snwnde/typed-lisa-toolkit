from collections.abc import Mapping
from types import ModuleType
from typing import Any, Literal, cast, overload

import array_api_compat as xpc

from ..types import (
    Array,
    Axis,
    EvolutionarySpectralDensity,
    Grid2D,
    Linspace,
    SpectralDensity,
    TimedFSData,
)
from ..types import (
    representations as reps,
)
from ..types._mixins import ChannelMapping

ConvertibleReps = (
    reps.FrequencySeries[Axis]
    | reps.TimeSeries[Axis]
    | reps.WDM[Grid2D[Linspace, Linspace]]
    | reps.STFT[Grid2D[Axis, Axis]]
)


def get_xyz2aet_matrix(xp: ModuleType):
    """Get the matrix that converts from XYZ to AET channels."""
    matrix = xp.array([[-1.0, 0.0, 1.0], [1.0, -2.0, 1.0], [1.0, 1.0, 1.0]])
    scales = xp.array([xp.sqrt(2.0), xp.sqrt(6.0), xp.sqrt(3.0)])
    return matrix / scales[:, None]


def get_aet2xyz_matrix(xp: ModuleType):
    """Get the matrix that converts from AET to XYZ channels."""
    return get_xyz2aet_matrix(xp).T


def _matrix_mult[VT: Array | ConvertibleReps](matrix: Any, *vectors: VT) -> list[VT]:
    """Perform matrix multiplication between a 3x3 matrix and three vectors."""
    # We do manually because not all waveforms have an underlying array that we can
    # use for matrix multiplication.
    return [
        matrix[i, 0] * vectors[0]
        + matrix[i, 1] * vectors[1]
        + matrix[i, 2] * vectors[2]
        for i in range(3)
    ]


def _get_xp(*args: ConvertibleReps | Array):
    try:
        return xpc.get_namespace(*args)
    except TypeError:
        return xpc.get_namespace(*[cast(ConvertibleReps, arg).entries for arg in args])


def _xyz2aet[VT: Array | ConvertibleReps](X: VT, Y: VT, Z: VT) -> tuple[VT, VT, VT]:
    xp = _get_xp(X, Y, Z)
    xyz2aet_matrix = get_xyz2aet_matrix(xp)
    A, E, T = _matrix_mult(xyz2aet_matrix, X, Y, Z)
    return A, E, T


def _aet2xyz[VT: Array | ConvertibleReps](A: VT, E: VT, T: VT) -> tuple[VT, VT, VT]:
    xp = _get_xp(A, E, T)
    aet2xyz_matrix = get_aet2xyz_matrix(xp)
    X, Y, Z = _matrix_mult(aet2xyz_matrix, A, E, T)
    return X, Y, Z


def _get_type_error_msg(original: Mapping[str, ConvertibleReps], /) -> str:
    return (
        "Expected a mapping to :class:`~typed_lisa_toolkit.types.FrequencySeries`,"
        + " :class:`~typed_lisa_toolkit.types.TimeSeries`,"
        + ":class:`~typed_lisa_toolkit.types.WDM`, or :class:`~typed_lisa_toolkit.types.STFT`,"
        + f" got {type(original).__name__}. "
    )


def _get_kwargs(original: Mapping[str, ConvertibleReps], /) -> dict[str, object]:
    if isinstance(original, TimedFSData):
        return {"times": original.times}
    else:
        return {}


def _convert_mapping[MapT: Mapping[str, ConvertibleReps]](
    original: MapT, /, *, direction: Literal["xyz2aet", "aet2xyz"]
) -> MapT:
    """Convert :ref:`data <data_types>` or :ref:`waveforms <waveform_types>` in XYZ channels to AET channels.

    The conversion is performed according to the DDPC Rosetta stone convention.
    """
    if direction == "xyz2aet":
        x, y, z = original["X"], original["Y"], original["Z"]
        a, e, t = _xyz2aet(x, y, z)
        _dict = {"A": a, "E": e, "T": t}
    else:
        a, e, t = original["A"], original["E"], original["T"]
        x, y, z = _aet2xyz(a, e, t)
        _dict = {"X": x, "Y": y, "Z": z}

    if not isinstance(original, ChannelMapping):
        raise TypeError(_get_type_error_msg(original))
    kwargs = _get_kwargs(original)
    return type(original).from_dict(_dict, **kwargs)


def _convert_spectral_density[SDT: SpectralDensity | EvolutionarySpectralDensity](
    original: SDT, /, *, direction: Literal["xyz2aet", "aet2xyz"]
) -> SDT:
    _kernel = original.get_kernel()
    # If original is of type SpectralDensity, the kernel shape is (n_freqs, n_channels, n_channels);
    # if original is of type EvolutionarySpectralDensity, the kernel shape is (n_freqs, n_times, n_channels, n_channels).
    orig_channel_order = original.channel_order
    xp = xpc.get_namespace(_kernel)
    if direction == "xyz2aet":
        convert_matrix = get_xyz2aet_matrix(xp)
        assert orig_channel_order == ("X", "Y", "Z"), (
            f"Expected original channel order to be ('X', 'Y', 'Z'), got {orig_channel_order}."
        )
        new_channel_order = "A", "E", "T"
    else:
        convert_matrix = get_aet2xyz_matrix(xp)
        assert orig_channel_order == ("A", "E", "T"), (
            f"Expected original channel order to be ('A', 'E', 'T'), got {orig_channel_order}."
        )
        new_channel_order = "X", "Y", "Z"
    converted_kernel = xp.einsum(
        "ij,...jk,kl->...il",
        convert_matrix,
        _kernel,
        convert_matrix.T,
    )
    freqs = original._frequencies  # pyright: ignore[reportPrivateUsage]
    if isinstance(original, SpectralDensity):
        return type(original)(
            frequencies=freqs,
            inverse_sdm=converted_kernel,
            channel_order=new_channel_order,
        )
    else:
        times = original._times  # pyright: ignore[reportPrivateUsage]
        return type(original)(
            frequencies=freqs,
            times=times,
            inverse_esdm=converted_kernel,
            channel_order=new_channel_order,
        )


def _convert_array(xyz: Array, /, *, direction: Literal["xyz2aet", "aet2xyz"]) -> Array:
    xp = xpc.get_namespace(xyz)
    if direction == "xyz2aet":
        convert_matrix = get_xyz2aet_matrix(xp)
        assert xyz.shape[-1] == 3, (
            f"Expected last dimension of input array to be 3, got {xyz.shape[-1]}."
        )
    else:
        convert_matrix = get_aet2xyz_matrix(xp)
        assert xyz.shape[-1] == 3, (
            f"Expected last dimension of input array to be 3, got {xyz.shape[-1]}."
        )
    return xp.einsum("ij,...j->...i", convert_matrix, xyz)


_ConvertibleTypes = (
    Mapping[str, ConvertibleReps]
    | Array
    | SpectralDensity
    | EvolutionarySpectralDensity
)


@overload
def xyz2aet[MapT: Mapping[str, ConvertibleReps]](xyz: MapT, /) -> MapT: ...


@overload
def xyz2aet(xyz: Array, /) -> Array: ...


@overload
def xyz2aet(xyz: SpectralDensity, /) -> SpectralDensity: ...


@overload
def xyz2aet(xyz: EvolutionarySpectralDensity, /) -> EvolutionarySpectralDensity: ...


@overload
def xyz2aet(
    *,
    X: Array,
    Y: Array,
    Z: Array,
) -> tuple[Array, Array, Array]: ...


def xyz2aet(
    xyz: _ConvertibleTypes | None = None,
    /,
    *,
    X: Array | None = None,
    Y: Array | None = None,
    Z: Array | None = None,
):
    """Convert :ref:`data <data_types>`, :ref:`waveforms <waveform_types>` or :ref:`spectral density matrices <spectral_density_matrices>` in XYZ channels to AET channels.

    The conversion is performed according to the DDPC Rosetta stone convention.
    """
    if xyz is not None:
        if any(arg is not None for arg in (X, Y, Z)):
            raise ValueError("Cannot specify both xyz and X, Y, Z.")
        if isinstance(xyz, (SpectralDensity, EvolutionarySpectralDensity)):
            return _convert_spectral_density(xyz, direction="xyz2aet")
        if isinstance(xyz, Mapping):
            return _convert_mapping(xyz, direction="xyz2aet")
        return _convert_array(xyz, direction="xyz2aet")

    if X is not None and Y is not None and Z is not None:
        return _xyz2aet(X, Y, Z)
    raise ValueError("Must specify either xyz or all of X, Y, Z.")


@overload
def aet2xyz[MapT: Mapping[str, ConvertibleReps]](aet: MapT, /) -> MapT: ...


@overload
def aet2xyz(aet: Array, /) -> Array: ...


@overload
def aet2xyz(aet: SpectralDensity, /) -> SpectralDensity: ...


@overload
def aet2xyz(aet: EvolutionarySpectralDensity, /) -> EvolutionarySpectralDensity: ...


@overload
def aet2xyz(
    *,
    A: Array,
    E: Array,
    T: Array,
) -> tuple[Array, Array, Array]: ...


def aet2xyz(
    aet: _ConvertibleTypes | None = None,
    /,
    *,
    A: Array | None = None,
    E: Array | None = None,
    T: Array | None = None,
):
    """Convert :ref:`data <data_types>`, :ref:`waveforms <waveform_types>` or :ref:`spectral density matrices <spectral_density_matrices>` in AET channels to XYZ channels.

    The conversion is performed according to the DDPC Rosetta stone convention.
    """
    if aet is not None:
        if any(arg is not None for arg in (A, E, T)):
            raise ValueError("Cannot specify both aet and A, E, T.")
        if isinstance(aet, (SpectralDensity, EvolutionarySpectralDensity)):
            return _convert_spectral_density(aet, direction="aet2xyz")
        if isinstance(aet, Mapping):
            return _convert_mapping(aet, direction="aet2xyz")
        return _convert_array(aet, direction="aet2xyz")

    if A is not None and E is not None and T is not None:
        return _aet2xyz(A, E, T)
    raise ValueError("Must specify either aet or all of A, E, T.")
