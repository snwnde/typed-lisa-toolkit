"""Module for modes."""

import logging
from typing import NamedTuple, Protocol, Self, runtime_checkable, Any

log = logging.getLogger(__name__)

PosInt = int  # Positive integer


@runtime_checkable
class Mode(Protocol):
    """A mode."""

    def __getitem__(self, index: int) -> Any: ...  # noqa: D105
    def __len__(self) -> int: ...  # noqa: D105
    @property
    def degree(self) -> PosInt: ...  # noqa: D102

    @property
    def order(self) -> PosInt: ...  # noqa: D102

    @classmethod
    def cast(cls, mode: tuple[PosInt, ...] | Self) -> Self: ...  # noqa: D102


class Harmonic(NamedTuple):
    """A harmonic mode."""

    l: PosInt  # noqa: E741
    m: PosInt

    @property
    def degree(self) -> PosInt:  # noqa: D102
        return self.l

    @property
    def order(self) -> PosInt:  # noqa: D102
        return self.m

    @classmethod
    def cast(cls, mode: tuple[PosInt, ...] | Self) -> Self:  # noqa: D102
        return cls(*mode)


class QNM(Harmonic):
    """A quasinormal mode."""

    n: PosInt

    def __new__(cls, l: PosInt, m: PosInt, n: PosInt):  # noqa: D102, E741
        obj = super().__new__(cls, l, m)
        obj.n = n
        return obj

    @property
    def overtone(self) -> PosInt:  # noqa: D102
        return self.n


def cast_mode(mode: tuple[PosInt, ...]):
    """Cast a mode to a Mode."""
    if len(mode) == 2:
        return Harmonic.cast(mode)
    elif len(mode) == 3:
        return QNM.cast(mode)
    raise ValueError(f"Invalid mode: {mode}")
