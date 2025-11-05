from . import mixins

from typing import Literal

MethodT = Literal["__call__", "reduce", "reduceat", "outer", "accumulate", "at"]

__all__ = ["mixins", "MethodT"]
