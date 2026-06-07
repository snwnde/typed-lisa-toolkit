"""Runtime validators for L2D interface contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import TypeIs

if TYPE_CHECKING:
    from .contract import AnyGrid, Axis, Domain, Grid2D, Grid2DSparse, Representation


def _logical_grid_ndim_from_domain(domain: str) -> Literal[1, 2]:
    """Return logical grid dimensionality (number of axes), not entries shape rank.

    For sparse grids, this returns the logical dimension (2 for time-frequency)
    even though the entries shape will be flattened to 1D.
    """
    if domain in ("time", "frequency"):
        return 1
    if domain == "time-frequency":
        return 2
    msg = (
        f"Invalid domain: {domain!r}. "
        "Expected 'time', 'frequency', or 'time-frequency'."
    )
    raise ValueError(msg)


def _axis_length(axis: Any) -> int:
    try:
        length = len(axis)
    except TypeError as error:
        msg = "Each grid axis must have a defined length."
        raise ValueError(msg) from error

    if length < 0:
        msg = "Axis length must be non-negative."
        raise ValueError(msg)
    return length


def _is_2D_grid(grid: AnyGrid) -> TypeIs[Grid2D[Axis, Axis]]:  # noqa: N802
    """Determine if a grid is a 2D grid."""
    return len(grid) == 2  # noqa: PLR2004


def _is_sparse_grid[A0: Axis, A1: Axis](
    grid: Grid2D[A0, A1],
) -> TypeIs[Grid2DSparse[A0, A1]]:
    """Determine if a grid is a sparse 2D grid."""
    return hasattr(grid, "indices")


def validate_representation[DomainT: Domain, GridT: AnyGrid, KindT: str | None](
    representation: Representation[DomainT, GridT, KindT],
) -> Literal[True]:
    """Validate core runtime invariants of a representation object."""
    logical_grid_ndim = _logical_grid_ndim_from_domain(representation.domain)

    shape = representation.entries.shape

    if _is_2D_grid(representation.grid) and _is_sparse_grid(representation.grid):
        # Sparse 2D grids always have shape
        # (n_batch, n_channels, n_harmonics, n_features, n_sparse)
        if len(shape) != 5:  # noqa: PLR2004
            msg = (
                "Invalid entries shape rank for sparse grid. Expected "
                "5 dims (n_batches, n_channels, n_harmonics, n_features, n_sparse), "
                f"got {len(shape)} with shape {shape}."
            )
            raise ValueError(msg)

        # Validate sparse grid indices
        indices = representation.grid.indices
        indices_shape = indices.shape

        if indices_shape != (shape[-1], 2):
            msg = (
                f"Invalid sparse grid indices shape. Expected ({shape[-1]}, 2), got "
                f"{indices_shape}."
            )
            raise ValueError(msg)
    else:
        # Dense or 1D grid validation expects shape
        # (n_batch, n_channels, n_harmonics, n_features, *grid_dims)
        if len(shape) != 4 + logical_grid_ndim:
            msg = (
                "Invalid entries shape rank. Expected "
                f"{4 + logical_grid_ndim} dims for domain {representation.domain!r}, "
                f"got {len(shape)} with shape {shape}."
            )
            raise ValueError(msg)

        if len(representation.grid) != logical_grid_ndim:
            msg = (
                "Invalid grid dimensionality. Expected "
                f"{logical_grid_ndim} axis/axes for domain {representation.domain!r}, "
                f"got {len(representation.grid)}."
            )
            raise ValueError(msg)

        grid_shape = tuple(_axis_length(axis) for axis in representation.grid)
        entries_grid_shape = shape[-logical_grid_ndim:]
        if entries_grid_shape != grid_shape:
            msg = (
                "AnyGrid shape does not match trailing entries dimensions. "
                f"Expected {grid_shape}, got {entries_grid_shape}."
            )
            raise ValueError(msg)

    return True
