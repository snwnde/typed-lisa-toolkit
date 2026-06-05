"""
Utility functions manipulating the core objects.

Not all use cases are handled by these utilities. You are encouraged to
write your own functions for your specific use case. Feel free to share them
with the community by opening an issue and/or a merge request if you think
your function is general enough to be useful for others.

.. currentmodule:: typed_lisa_toolkit.shop

.. autosummary::
   :toctree: _generated
   :template: base.rst

    freq2time
    time2freq
    time2wdm
    wdm2time
    wdm2freq
    freq2wdm
    xyz2aet
    aet2xyz

Below are some general utility functions.

.. currentmodule:: typed_lisa_toolkit.shop

.. autosummary::
   :toctree: _generated
   :template: base.rst

    year2second
    month2second
    day2second
    sday2second
    jday2second
    week2second
    hour2second
    minute2second

"""

from . import conversions, transforms
from .conversions import (
    aet2xyz,
    day2second,
    hour2second,
    jday2second,
    minute2second,
    month2second,
    sday2second,
    week2second,
    xyz2aet,
    year2second,
)
from .transforms import freq2time, freq2wdm, time2freq, time2wdm, wdm2freq, wdm2time

__all__ = [
    "aet2xyz",
    "conversions",
    "day2second",
    "freq2time",
    "freq2wdm",
    "hour2second",
    "jday2second",
    "minute2second",
    "month2second",
    "sday2second",
    "time2freq",
    "time2wdm",
    "transforms",
    "wdm2freq",
    "wdm2time",
    "week2second",
    "xyz2aet",
    "year2second",
]
