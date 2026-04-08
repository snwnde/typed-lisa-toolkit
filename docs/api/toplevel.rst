Top-level API
=============

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

   load_data
   load_ldc_data

Factory Functions
-----------------

The top-level helpers below are grouped by purpose.


Modes
^^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst

   cast_mode


Representations
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst

   frequency_series
   time_series
   phasor
   stft
   wdm

The following functions construct new representations by processing existing representations:

.. autosummary::
   :toctree: _generated
   :template: base.rst

   densify_phasor


Data
^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst

   tsdata
   fsdata
   stftdata
   wdmdata
   timed_fsdata


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

   linspace