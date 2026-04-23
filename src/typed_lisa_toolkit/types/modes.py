"""Mode types."""

import logging
from typing import NamedTuple, NewType, Self, overload

log = logging.getLogger(__name__)

PosInt = NewType("PosInt", int)  # Positive integer


class Harmonic(NamedTuple):
    """A harmonic mode."""

    l: PosInt  # noqa: E741
    """The degree of the mode."""

    m: int
    """The order of the mode."""

    @property
    def degree(self) -> PosInt:
        """Return the attribute :attr:`l`."""
        return self.l

    @property
    def order(self) -> int:
        """Return the attribute :attr:`m`."""
        return self.m

    @classmethod
    def cast(cls, mode: tuple[int, int]) -> Self:
        """Cast a tuple to :class:`.Harmonic`."""
        return cls(PosInt(mode[0]), mode[1])


class QuasiNormalMode(NamedTuple):
    """A quasinormal mode."""

    l: PosInt  # noqa: E741
    """The degree of the mode."""

    m: int
    """The order of the mode."""

    n: int
    """The overtone of the mode."""

    @property
    def degree(self) -> PosInt:
        """Return the attribute :attr:`l`."""
        return self.l

    @property
    def order(self) -> int:
        """Return the attribute :attr:`m`."""
        return self.m

    @property
    def overtone(self) -> int:
        """Return the attribute :attr:`n`."""
        return self.n

    @classmethod
    def cast(cls, mode: tuple[int, int, int]) -> Self:
        """Cast a tuple to :class:`.QNM`."""
        return cls(PosInt(mode[0]), mode[1], mode[2])


QNM = QuasiNormalMode


@overload
def cast_mode(mode: tuple[int, int]) -> Harmonic: ...


@overload
def cast_mode(mode: tuple[int, int, int]) -> QNM: ...


def cast_mode(mode: tuple[int, ...]):
    """Cast a tuple of positive integers to :class:`.Harmonic` or :class:`.QNM`."""
    if len(mode) == 2:  # noqa: PLR2004
        return Harmonic.cast(mode)
    if len(mode) == 3:  # noqa: PLR2004
        return QNM.cast(mode)
    msg = f"Invalid mode: {mode}. A mode should be a tuple of 2 or 3 positive integers."
    raise ValueError(msg)
