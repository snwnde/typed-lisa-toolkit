Top-level API
=============

.. currentmodule:: typed_lisa_toolkit

This page documents all the top-level objects
of the package.

Loaders
-------
.. autosummary::
   :toctree: _generated
   :template: base.rst

   load_data
   load_ldc_data



Constructors
------------

The following constructors are available for creating instances
of the various types defined in :mod:`types`:


Misc
^^^^

.. autosummary::
   :toctree: _generated
   :template: base.rst

   linspace


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