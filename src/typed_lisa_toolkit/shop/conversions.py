from collections.abc import Mapping
from types import ModuleType
from typing import Any, Literal, overload

import array_api_compat as xpc

from ..types import (
    Array,
    Axis,
    EvolutionarySpectralDensity,
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
    | reps.WDM
    | reps.STFT[Axis, Axis]
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


def _xyz2aet[VT: ConvertibleReps](X: VT, Y: VT, Z: VT) -> tuple[VT, VT, VT]:
    xp = xpc.get_namespace(X.entries, Y.entries, Z.entries)
    xyz2aet_matrix = get_xyz2aet_matrix(xp)
    A, E, T = _matrix_mult(xyz2aet_matrix, X, Y, Z)
    return A, E, T


def _aet2xyz[VT: ConvertibleReps](A: VT, E: VT, T: VT) -> tuple[VT, VT, VT]:
    xp = xpc.get_namespace(A.entries, E.entries, T.entries)
    aet2xyz_matrix = get_aet2xyz_matrix(xp)
    X, Y, Z = _matrix_mult(aet2xyz_matrix, A, E, T)
    return X, Y, Z


def _get_type_error_msg(original: Mapping[str, ConvertibleReps], /) -> str:
    return (
        "Expected a mapping to :class:`~typed_lisa_toolkit.types.FrequencySeries`,"
        + " :class:`~typed_lisa_toolkit.types.TimeSeries`,"
        + ":class:`~typed_lisa_toolkit.types.WDM`, or :class:`~typed_lisa_toolkit.types.STFT`,"
        + f" got {type(original).__name__}. "
        + "This function is only intended for use with representations, not raw arrays."
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


@overload
def xyz2aet[MapT: Mapping[str, ConvertibleReps]](xyz: MapT, /) -> MapT: ...


@overload
def xyz2aet(xyz: SpectralDensity, /) -> SpectralDensity: ...


@overload
def xyz2aet(xyz: EvolutionarySpectralDensity, /) -> EvolutionarySpectralDensity: ...


def xyz2aet(xyz: Any, /):
    """Convert :ref:`data <data_types>`, :ref:`waveforms <waveform_types>` or :ref:`spectral density matrices <spectral_density_matrices>` in XYZ channels to AET channels.

    The conversion is performed according to the DDPC Rosetta stone convention.
    """
    if isinstance(xyz, (SpectralDensity, EvolutionarySpectralDensity)):
        return _convert_spectral_density(xyz, direction="xyz2aet")
    else:
        return _convert_mapping(xyz, direction="xyz2aet")


@overload
def aet2xyz[MapT: Mapping[str, ConvertibleReps]](aet: MapT, /) -> MapT: ...


@overload
def aet2xyz(aet: SpectralDensity, /) -> SpectralDensity: ...


@overload
def aet2xyz(aet: EvolutionarySpectralDensity, /) -> EvolutionarySpectralDensity: ...


def aet2xyz(aet: Any, /):
    """Convert :ref:`data <data_types>`, :ref:`waveforms <waveform_types>` or :ref:`spectral density matrices <spectral_density_matrices>` in AET channels to XYZ channels.

    The conversion is performed according to the DDPC Rosetta stone convention.
    """
    if isinstance(aet, (SpectralDensity, EvolutionarySpectralDensity)):
        return _convert_spectral_density(aet, direction="aet2xyz")
    else:
        return _convert_mapping(aet, direction="aet2xyz")
