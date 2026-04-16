r"""Top-level API of the project.

.. currentmodule:: typed_lisa_toolkit

Typed LISA Toolkit has a top-level module, ``typed_lisa_toolkit``, with functions for
loading data and constructing objects. It also has two submodules:

- :mod:`typed_lisa_toolkit.types` for core typed objects.
- :mod:`typed_lisa_toolkit.shop` for utility functions acting on those objects (conversions, transforms, etc.). Shop for what you need in this module.

This page focuses on the top-level module.

Loaders
-------

Functions that load data from disk or memory in common formats.

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

These are functions that create objects of types in :mod:`~typed_lisa_toolkit.types`,
but with a nicer API than the class initializer methods. Below, they are grouped by
purpose.

Representations
^^^^^^^^^^^^^^^

Representations are the most fundamental types in TLT, and are components
of other more involved types.

They house numerical signal arrays, together with time and/or frequency
grids and other metadata. They not only support basic algebraic operations as
the underlying arrays do, but also some grid-aware operations such as
semantic subsetting and embedding.

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

Data objects house multi-channel LISA data, where all channels
are of the same representation type suitable for representing
recorded data.

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

Similar to data objects, waveform objects group multiple representations together.
Waveform objects are meant to represent modeled signals, and as such, they can carry
theoretical information such as harmonic content, which is not the case for data objects.

In the nomenclature of the project, "raw" waveforms are detector-independent strains,
such as the two polarizations (:math:`h_+` and :math:`h_\times`) or the spherical harmonic
decomposition (:math:`h_{lm}`); "projected" waveforms depend on LISA's response,
and are typically TDI channel signals.

The factory functions are available in long-form and short aliases:

.. autosummary::
   :toctree: _generated
   :template: base.rst

   harmonic_waveform
   homogeneous_harmonic_waveform
   plus_cross_waveform
   projected_waveform
   harmonic_projected_waveform
   homogeneous_harmonic_projected_waveform
   hw
   hhw
   pcw
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

Noise Models and Likelihoods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst
   :nosignatures:

   make_sdm
   noise_model
   whittle




Miscellaneous
^^^^^^^^^^^^^

The following functions do not fit into the above categories,
but are useful helpers for constructing the above objects.

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
    make_sdm,
    noise_model,
    pcw,
    phasor,
    phasor_to_fs_hpw,
    phasor_to_fs_hw,
    phasor_to_fs_pw,
    plus_cross_waveform,
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
    whittle,
)
from ._loaders import load_data, load_ldc_data, load_mojito, load_sangria

__all__ = [
    "build_grid2d",
    "cast_mode",
    "construct_fsdata",
    "construct_stftdata",
    "construct_timed_fsdata",
    "construct_tsdata",
    "construct_wdmdata",
    "densify_phasor",
    "densify_phasor_hpw",
    "densify_phasor_hw",
    "densify_phasor_pw",
    "frequency_series",
    "fsdata",
    "get_dense_maker",
    "harmonic_projected_waveform",
    "harmonic_waveform",
    "hhpw",
    "hhw",
    "homogeneous_harmonic_projected_waveform",
    "homogeneous_harmonic_waveform",
    "hpw",
    "hw",
    "linspace",
    "linspace_from_array",
    "load_data",
    "load_ldc_data",
    "load_mojito",
    "load_sangria",
    "make_sdm",
    "noise_model",
    "pcw",
    "phasor",
    "phasor_to_fs_hpw",
    "phasor_to_fs_hw",
    "phasor_to_fs_pw",
    "plus_cross_waveform",
    "projected_waveform",
    "pw",
    "shop",
    "stft",
    "stftdata",
    "sum_harmonics",
    "time_series",
    "timed_fsdata",
    "tsdata",
    "types",
    "utils",
    "viz",
    "wdm",
    "wdmdata",
    "whittle",
]
