"""Top-level API for ``typed_lisa_toolkit``.

.. currentmodule:: typed_lisa_toolkit

Typed LISA Toolkit exposes three complementary API surfaces:

- :mod:`~typed_lisa_toolkit.types` for core typed representations.
- :mod:`~typed_lisa_toolkit.shop` for converters and transformations.
- Top-level helper functions for loading data and constructing objects.

This page focuses on the top-level functions.

Namespaces
----------

Public namespace APIs are documented in:

- :doc:`types`
- :doc:`shop`

Loaders
-------

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   load_data
   load_sangria
   load_mojito
   load_ldc_data

Factory Functions
-----------------

The top-level helpers below are grouped by purpose.


Modes
^^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   cast_mode


Representations
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   frequency_series
   time_series
   phasor
   stft
   wdm

The following functions construct new representations by processing existing representations:

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   densify_phasor


Data
^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   tsdata
   fsdata
   stftdata
   wdmdata
   timed_fsdata
   construct_tsdata
   construct_fsdata
   construct_stftdata
   construct_wdmdata
   construct_timed_fsdata


Waveforms
^^^^^^^^^

Waveform constructors are available in long-form and concise aliases:

.. autosummary::
   :toctree: _generated
   :template: base.rst

   harmonic_waveform
   homogeneous_harmonic_waveform
   projected_waveform
   harmonic_projected_waveform
   homogeneous_harmonic_projected_waveform
   hw
   hhw
   pw
   hpw
   hhpw

The following functions construct new waveforms by processing existing waveforms:

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   sum_harmonics
   densify_phasor_hw
   densify_phasor_pw
   densify_phasor_hpw
   phasor_to_fs_hw
   phasor_to_fs_pw
   phasor_to_fs_hpw
   get_dense_maker

Misc
^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   linspace
   linspace_from_array
   build_grid2d

"""

from . import shop, types, utils, viz
from ._constructors import (
    build_grid2d,
    cast_mode,
    construct_fsdata,
    construct_stftdata,
    construct_timed_fsdata,
    construct_tsdata,
    construct_wdmdata,
    densify_phasor,
    densify_phasor_hpw,
    densify_phasor_hw,
    densify_phasor_pw,
    frequency_series,
    fsdata,
    get_dense_maker,
    harmonic_projected_waveform,
    harmonic_waveform,
    hhpw,
    hhw,
    homogeneous_harmonic_projected_waveform,
    homogeneous_harmonic_waveform,
    hpw,
    hw,
    linspace,
    linspace_from_array,
    phasor,
    phasor_to_fs_hpw,
    phasor_to_fs_hw,
    phasor_to_fs_pw,
    projected_waveform,
    pw,
    stft,
    stftdata,
    sum_harmonics,
    time_series,
    timed_fsdata,
    tsdata,
    wdm,
    wdmdata,
)
from ._loaders import load_data, load_ldc_data, load_mojito, load_sangria

__all__ = [
    "utils",
    "viz",
    "types",
    "shop",
    "frequency_series",
    "harmonic_projected_waveform",
    "harmonic_waveform",
    "homogeneous_harmonic_waveform",
    "hhpw",
    "hhw",
    "homogeneous_harmonic_projected_waveform",
    "hpw",
    "hw",
    "projected_waveform",
    "pw",
    "stft",
    "time_series",
    "wdm",
    "phasor",
    "sum_harmonics",
    "densify_phasor",
    "densify_phasor_hw",
    "densify_phasor_pw",
    "densify_phasor_hpw",
    "phasor_to_fs_hw",
    "phasor_to_fs_pw",
    "phasor_to_fs_hpw",
    "get_dense_maker",
    "cast_mode",
    "load_data",
    "load_sangria",
    "load_ldc_data",
    "load_mojito",
    "tsdata",
    "fsdata",
    "stftdata",
    "wdmdata",
    "timed_fsdata",
    "construct_tsdata",
    "construct_fsdata",
    "construct_timed_fsdata",
    "construct_stftdata",
    "construct_wdmdata",
    "linspace",
    "linspace_from_array",
    "build_grid2d",
]
