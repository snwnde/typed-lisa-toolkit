"""Mode types."""

import logging
from typing import NamedTuple, Self, overload

log = logging.getLogger(__name__)

PosInt = int  # Positive integer


class Harmonic(NamedTuple):
    """A harmonic mode."""

    l: PosInt  # noqa: E741
    """The degree of the mode."""

    m: PosInt
    """The order of the mode."""

    @property
    def degree(self) -> PosInt:
        """Return the attribute :attr:`l`."""
        return self.l

    @property
    def order(self) -> PosInt:
        """Return the attribute :attr:`m`."""
        return self.m

    @classmethod
    def cast(cls, mode: tuple[PosInt, PosInt]) -> Self:
        """Cast a tuple to :class:`.Harmonic`."""
        return cls(*mode)


class QuasiNormalMode(NamedTuple):
    """A quasinormal mode."""

    l: PosInt  # noqa: E741
    """The degree of the mode."""

    m: PosInt
    """The order of the mode."""

    n: PosInt
    """The overtone of the mode."""

    @property
    def degree(self) -> PosInt:
        """Return the attribute :attr:`l`."""
        return self.l

    @property
    def order(self) -> PosInt:
        """Return the attribute :attr:`m`."""
        return self.m

    @property
    def overtone(self) -> PosInt:
        """Return the attribute :attr:`n`."""
        return self.n

    @classmethod
    def cast(cls, mode: tuple[PosInt, PosInt, PosInt] | Self) -> Self:
        """Cast a tuple to :class:`.QNM`."""
        return cls(*mode)


QNM = QuasiNormalMode


@overload
def cast_mode(mode: tuple[PosInt, PosInt]) -> Harmonic: ...


@overload
def cast_mode(mode: tuple[PosInt, PosInt, PosInt]) -> QNM: ...


def cast_mode(mode: tuple[PosInt, ...]):
    """Cast a tuple of positive integers to :class:`.Harmonic` or :class:`.QNM`."""
    if len(mode) == 2:
        return Harmonic.cast(mode)
    elif len(mode) == 3:
        return QNM.cast(mode)
    raise ValueError(f"Invalid mode: {mode}")
