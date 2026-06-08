"""API contract for L2D data and waveform representations.

This module defines the protocol-based contracts for representing gravitational wave
signals in different domains and on different grids.

For conceptual diagrams and shape conventions, see :doc:`overview`.
"""

from collections.abc import Iterator
from typing import Any, Literal, Protocol, Self, overload, override, runtime_checkable

from .array import Array

AnyArray = Array[Any, Any]


class Axis(Protocol):
    """Protocol for axes in representations.

    Note
    ----
    A 1D float array in NumPy/JAX does not follow this protocol because a 0-d array
    is not a python float. Consider wrapping 1D arrays in a custom Axis class.
    """

    @overload
    def __getitem__(self, __index: int, /) -> float: ...

    @overload
    def __getitem__(self, __index: slice, /) -> Self: ...


type Grid1D[AxisT: Axis] = tuple[AxisT]
"""1D grid."""


type Grid2DCartesian[Axis0: Axis, Axis1: Axis] = tuple[Axis0, Axis1]
"""Cartesian 2D grid."""


@runtime_checkable
class Grid2DSparse[Axis0: Axis, Axis1: Axis](Protocol):
    """Protocol for sparse 2D (time-frequency) grids.

    A sparse grid selects a subset of points from the Cartesian product of two axes.
    Implements tuple-like indexing to access axes (compatible with dense Grid2D
    interface) and provides an ``indices`` property mapping sparse points to dense
    grid coordinates.
    """

    @overload
    def __getitem__(self, __index: Literal[0], /) -> Axis0: ...

    @overload
    def __getitem__(self, __index: Literal[1], /) -> Axis1: ...

    def __len__(self) -> Literal[2]:
        """Return the number of dimensions (always 2 for :class:`Grid2DSparse`)."""
        ...

    def __iter__(self) -> Iterator[Axis0 | Axis1]:
        """Iterate over the axes in order (frequency axis first, then time axis)."""
        ...

    @property
    def indices(self) -> AnyArray:
        """Array of shape (n_sparse, 2) containing the indices of the non-zero points in the 2D grid."""  # noqa: E501
        ...


type Grid2D[Axis0: Axis, Axis1: Axis] = (
    Grid2DCartesian[Axis0, Axis1] | Grid2DSparse[Axis0, Axis1]
)
"""2D grid specification for time-frequency representations."""


AnyGrid = Grid1D[Axis] | Grid2D[Axis, Axis]


Domain = Literal["time", "frequency", "time-frequency"]


class Representation[DomainT: Domain, GridT: AnyGrid, KindT: str | None](Protocol):
    """API contract for representation of gravitational wave signals."""

    @property
    def entries(self) -> AnyArray:
        """Multi-dimensional array following the Python Array API standard.

        Shape convention: ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``

        - ``n_batches``: Independent signal realizations
        - ``n_channels``: Detector channels (e.g., 1 for single, 3 for TDI X, Y, Z)
        - ``n_harmonics``: Harmonic modes (1 for single-mode, >1 for multi-mode)
        - ``n_features``: Features per grid point (1 for scalar, >1 for multivariate)

            - Most common case: time-series of scalar or Fourier coefficients (``n_features=1``)
            - Multivariate example: frequency-domain series with both amplitude and phase (``n_features=2``)

        - ``*grid_like``: Remaining dimensions determined by grid type

            - 1D grid: time-domain series or frequency-domain series ``(n_grid,)``
            - 2D dense grid: dense time-frequency representations ``(n_freq, n_time)``
            - 2D sparse grid: sparse time-frequency representations ``(n_sparse,)``
              where `n_sparse` is the number of non-zero points. This can be read
              from `grid.indices` which is of shape ``(n_sparse, 2)``


        .. note::
            Never squeeze dimensions, even if trivial (e.g., ``n_channels=1``).

            The presence of reserved dimensions for channels and harmonics is a design choice
            to support efficient cross-channel and cross-harmonic operations where applicable.
            It should not be taken to imply that all representations must have multiple channels or harmonics
            (many will have just one), nor that it is priledged to populate these dimensions by
            all means. In fact, when signals are homogeneous acros harmonics (common in waveform
            generation), we should use mapping containers keyed by harmonic mode and valued
            by representations with ``n_harmonics=1`` (i.e., ``shape[2] == 1``) rather than
            forcing them into a single array with ``n_harmonics > 1`` (see :class:`HarmonicWaveform`
            and :class:`HarmonicProjectedWaveform`). Even more so, when signals are homogeneous across channels
            (common in detector response and recorded data), mapping containers keyed by channel name and
            valued by representations with ``n_channels=1`` (i.e., ``shape[1] == 1``) also provide more
            semantic clarity (see :class:`Data`, :class:`TransformedData`, and :class:`ProjectedWaveform`).
        """  # noqa: E501
        ...

    @property
    def grid(self) -> GridT:
        """Grid specification defining axis points.

        1D grid for time-domain or frequency-domain series; 2D for time-frequency.
        """
        ...

    @property
    def domain(self) -> DomainT:
        """Physical domain of the representation.

        One of: ``'time'``, ``'frequency'``, or ``'time-frequency'``.
        See :class:`TDRepresentation`, :class:`FDRepresentation`,
        :class:`DenseTFRepresentation`
        and :class:`SparseTFRepresentation`.
        """
        ...

    @property
    def kind(self) -> KindT:
        """Optional semantic kind for domain-specific variants.

        Examples: ``'wavelet'`` for time-frequency representations.
        ``None`` for standard representations (e.g., scalar time/frequency series).
        """
        ...


class TDRepresentation[KindT: str | None, AxisT: Axis](
    Representation[Literal["time"], Grid1D[AxisT], KindT], Protocol
):
    """API contract for time-domain representations of gravitational wave signals."""


class UniformTDRepresentation[KindT: str | None](
    TDRepresentation[KindT, Axis], Protocol
):
    """API contract for uniformly sampled time-domain representations of gravitational wave signals."""  # noqa: E501

    @property
    def dt(self) -> float:
        """Uniform time sampling interval."""
        ...


class FDRepresentation[KindT: str | None, AxisT: Axis](
    Representation[Literal["frequency"], Grid1D[AxisT], KindT], Protocol
):
    """API contract for frequency-domain representations of gravitational wave signals."""  # noqa: E501


class UniformFDRepresentation[KindT: str | None](
    FDRepresentation[KindT, Axis], Protocol
):
    """API contract for uniformly sampled frequency-domain representations of gravitational wave signals."""  # noqa: E501

    @property
    def df(self) -> float:
        """Uniform frequency sampling interval."""
        ...


class TFRepresentation[KindT: str | None, Axis0: Axis, Axis1: Axis](
    Representation[Literal["time-frequency"], Grid2D[Axis0, Axis1], KindT],
    Protocol,
):
    """API contract for time-frequency representations of gravitational wave signals."""


class DenseTFRepresentation[KindT: str | None, Axis0: Axis, Axis1: Axis](
    TFRepresentation[KindT, Axis0, Axis1], Protocol
):
    """API contract for dense time-frequency representations of gravitational wave signals.

    The entries array has shape ``(n_batches, n_channels, n_harmonics, n_features, n_freq, n_time)``
    representing the full Cartesian product of frequency and time axes.
    """  # noqa: E501


class SparseTFRepresentation[KindT: str | None, Axis0: Axis, Axis1: Axis](
    Representation[Literal["time-frequency"], Grid2DSparse[Axis0, Axis1], KindT],
    Protocol,
):
    """API contract for sparse time-frequency representations of gravitational wave signals.

    For sparse representations, entries are flattened along the time-frequency dimensions.
    The actual TF point coordinates are recovered from ``grid.indices``.

    The entries array has shape ``(n_batches, n_channels, n_harmonics, n_features, n_sparse)``
    where ``n_sparse`` is the number of non-zero time-frequency points.

    Example
    -------
    Sparse time-frequency representation (only 5000 active points out of 100×500=50000)::

        # entries shape: (1, 1, 1, 1, 5000)
        # grid.indices: (5000, 2)  with (freq_idx, time_idx) pairs
        # Only 10% of the dense grid is used, enabling memory-efficient storage

    """  # noqa: E501, RUF002


class _ChannelMapping[DomainT: Domain, GridT: AnyGrid, KindT: str | None](Protocol):
    """Base protocol for channel-keyed representation mappings.

    This is an internal base protocol that defines the common API for containers
    that map channel names to :class:`Representation` objects. All channels share
    the same domain, grid, and kind.
    """

    def __getitem__(self, key: str) -> Representation[DomainT, GridT, KindT]:
        """Get a channel representation by name.

        The :class:`Representation` objects returned by this method
        must have ``n_channels=1`` (i.e. ``shape[1] == 1``) and ``n_harmonics=1`` (i.e. ``shape[2] == 1``).

        .. note::

            Though not required, implementations are encouraged
            to return views of the same underlying array rather than storing
            independent arrays per channel.
        """  # noqa: E501
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over channel names."""
        ...

    def __len__(self) -> int:
        """Return the number of channels."""
        ...

    @property
    def domain(self) -> DomainT:
        """Physical domain shared by all channels.

        All :class:`Representation` objects must share the same domain.
        """
        ...

    @property
    def grid(self) -> GridT:
        """Grid specification shared by all channels.

        All :class:`Representation` objects must share the same grid.
        """
        ...

    @property
    def kind(self) -> KindT:
        """Semantic kind shared by all channels.

        All :class:`Representation` objects must share the same kind.
        """
        ...

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Names of all channels and their order."""
        ...

    def get_kernel(self) -> AnyArray:
        """Return an array of the conventional shape ``(n_batches, n_channels, 1, n_features, *grid_like)``
        for downstream processing (e.g., by noise models to compute inner products).

        .. note::
            This method can be trivially implemented if the underlying data entries are already stored in the conventional shape.
            Otherwise, it can be implemented by stacking the representations of individual channels along the channel dimension.
        """  # noqa: D205, E501
        ...


@runtime_checkable
class Data(_ChannelMapping[Literal["time"], Grid1D[Axis], None], Protocol):
    """API contract for data containers of gravitational wave signals.

    :class:`Data` objects represent the output from (pre-processed) L1 Data,
    which is the source of truth
    for the entire L2D pipeline. They should not be modified by L2D analysis.

    Maps channel names to time-domain :class:`TDRepresentation` objects.
    All channels share the same time grid.
    """

    @override
    def __getitem__(self, key: str) -> TDRepresentation[None, Axis]:
        """Get a channel representation by name.

        See :meth:`_ChannelMapping.__getitem__`.
        """
        ...

    @property
    @override
    def domain(self) -> Literal["time"]:
        """Physical domain (always 'time' for `Data`).

        See :attr:`_ChannelMapping.domain`.
        """
        ...

    @property
    @override
    def grid(self) -> Grid1D[Axis]:
        """1D time grid specification shared by all channels.

        See :attr:`_ChannelMapping.grid`.
        """
        ...

    @property
    def times(self) -> Axis:
        """Time axis."""
        ...


class TransformedData[DomainT: Domain, GridT: AnyGrid, KindT: str | None](
    _ChannelMapping[DomainT, GridT, KindT], Protocol
):
    """Protocol for objects transformed from Data objects.

    :class:`TransformedData` objects are outputs of transformation of :class:`Data` objects,
    e.g., by Fourier transform, or time-frequency transform. They preserve channel semantics
    of :class:`Data` objects.

    Maps channel names to :class:`Representation` objects in any domain (time, frequency,
    or time-frequency). All channels share the same domain, grid, kind.

    :class:`TransformedData`, :class:`PlusCrossWaveform`, and :class:`HarmonicProjectedWaveform`
    are the three only entry points for downstream processing of L2D.
    """  # noqa: E501


class HarmonicWaveform[
    HarmonicT: tuple[int, int] | tuple[int, int, int],
    DomainT: Domain,
    GridT: AnyGrid,
    KindT: str | None,
](Protocol):
    """API contract for gravitational wave signals generated by waveform models.

    HarmonicWaveform objects are keyed by harmonic mode,
    supporting heterogeneous grids across modes.
    """

    def __getitem__(self, key: HarmonicT) -> Representation[DomainT, GridT, KindT]:
        """Get a harmonic representation by mode index.

        The :class:`Representation` objects returned by this method
        must have `n_harmonics=1` (i.e. `shape[2] == 1`).
        """
        ...

    def __iter__(self) -> Iterator[HarmonicT]:
        """Iterate over harmonic modes."""
        ...

    def __len__(self) -> int:
        """Return the number of harmonic modes."""
        ...

    @property
    def domain(self) -> DomainT:
        """Physical domain shared by all harmonics.

        All :class:`Representation` objects must share the same domain.
        """
        ...

    @property
    def harmonics(self) -> tuple[HarmonicT, ...]:
        """All harmonic modes and their order."""
        ...


class PlusCrossWaveform[
    DomainT: Domain,
    GridT: AnyGrid,
    KindT: str | None,
](_ChannelMapping[DomainT, GridT, KindT], Protocol):
    """API contract for plus and cross polarization waveforms.

    :class:`PlusCrossWaveform` contains the plus and cross polarization waveforms.
    The method ``get_kernel`` returns an array of shape
    ``(n_batches, 2, 1, 1, *grid_like)`` where the
    channel dimension of size 2 corresponds to the plus (the 0 component) and cross
    (the 1 component) polarizations.
    """

    @property
    def plus(self) -> Representation[DomainT, GridT, KindT]:
        """Plus polarization waveform."""
        ...

    @property
    def cross(self) -> Representation[DomainT, GridT, KindT]:
        """Cross polarization waveform."""
        ...


class ProjectedWaveform[
    DomainT: Domain,
    GridT: AnyGrid,
    KindT: str | None,
](_ChannelMapping[DomainT, GridT, KindT], Protocol):
    """API contract for waveforms projected onto detector channels without harmonic decomposition.

    All channels share the same domain, grid, and kind.
    """  # noqa: E501


class HarmonicProjectedWaveform[
    HarmonicT: tuple[int, int] | tuple[int, int, int],
    DomainT: Domain,
    GridT: AnyGrid,
    KindT: str | None,
](Protocol):
    """API contract for gravitational wave signals projected onto detector channels.

    This is in general obtained by applying LISA response to generated waveforms.

    Due to the heterogeneity of harmonic modes in general,
    :class:`HarmonicProjectedWaveform` does not provide a ``get_kernel`` method, but
    the kernels for individual harmonic modes can be retrieved by
    the ``get_kernel`` method of the corresponding :class:`ProjectedWaveform`
    objects returned by ``__getitem__``.
    """

    def __getitem__(self, key: HarmonicT) -> ProjectedWaveform[DomainT, GridT, KindT]:
        """Get the projected waveform for a harmonic mode (across all detector channels).

        Though not strictly required, implementations are encouraged
        to return views of the same underlying array rather than storing
        independent ProjectedWaveform objects per harmonic (if possible).
        """  # noqa: E501
        ...

    def __iter__(self) -> Iterator[HarmonicT]:
        """Iterate over harmonic modes."""
        ...

    def __len__(self) -> int:
        """Return the number of harmonic modes."""
        ...

    @property
    def domain(self) -> DomainT:
        """Physical domain shared by all harmonics.

        All :class:`ProjectedWaveform` objects must share the same domain.
        """
        ...

    @property
    def harmonics(self) -> tuple[HarmonicT, ...]:
        """All harmonic modes and their order."""
        ...

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Names of all detector channels and their order.

        All :class:`ProjectedWaveform` objects must share the same channel names.
        """
        ...


class HomogeneousHarmonicProjectedWaveform[
    HarmonicT: tuple[int, int] | tuple[int, int, int],
    DomainT: Domain,
    GridT: AnyGrid,
    KindT: str | None,
](HarmonicProjectedWaveform[HarmonicT, DomainT, GridT, KindT], Protocol):
    """API contract for homogeneous gravitational wave signals projected onto detector channels."""  # noqa: E501

    def get_kernel(self) -> AnyArray:
        """Return an array of the conventional shape ``(n_batches, n_channels, n_harmonics, n_features, *grid_like)``
        for downstream processing (e.g., by noise models to compute inner products).
        """  # noqa: D205, E501
        ...
