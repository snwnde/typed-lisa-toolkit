from typing import Literal

from . import mixins

MethodT = Literal["__call__", "reduce", "reduceat", "outer", "accumulate", "at"]

__all__ = ["mixins", "MethodT"]
