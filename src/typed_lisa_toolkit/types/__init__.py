r"""
Types provided by TLT.

.. note::
    To create instances of these types, use the factory functions provided at the :mod:`top level <typed_lisa_toolkit>`.
    For example, use :func:`~typed_lisa_toolkit.time_series` to construct a :class:`TimeSeries` object.

.. warning::
    The `__init__` methods of these classes are **not** part of the public API, and may change without deprecation.

.. _representation_types:

Representations
^^^^^^^^^^^^^^^

Representations are the most fundamental types in TLT, and are components
of other more involved types.

They house numerical signal arrays, together with time and/or frequency
grids and other metadata. They not only support basic algebraic operations as
the underlying arrays do, but also some grid-aware operations such as
semantic subsetting and embedding.

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: reps_class.rst
   :nosignatures:

   TimeSeries
   UniformTimeSeries
   FrequencySeries
   UniformFrequencySeries
   Phasor
   ShortTimeFourierTransform
   WilsonDaubechiesMeyer
   STFT
   WDM

.. _data_types:

Data
^^^^

Data objects house multi-channel LISA data, where all channels
are of the same representation type suitable for representing
recorded data.

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: class.rst
   :nosignatures:

    TSData
    FSData
    TimedFSData
    STFTData
    WDMData


.. _mode_types:

Mode Types
^^^^^^^^^^

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: type.rst
   :nosignatures:

    Harmonic
    QuasiNormalMode
    QNM


.. _waveform_types:

Waveforms
^^^^^^^^^

Similar to data objects, waveform objects group multiple representations together.
Waveform objects are meant to represent modeled signals, and as such, they can carry
theoretical information such as harmonic content, which is not the case for data objects.

In the nomenclature of the project, "raw" waveforms are detector-independent strains,
such as the two polarizations (:math:`h_+` and :math:`h_\times`) or the spherical harmonic
decomposition (:math:`h_{lm}`); "projected" waveforms depend on LISA's response,
and are typically TDI channel signals.

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: class.rst
   :nosignatures:

    HarmonicWaveform
    HomogeneousHarmonicWaveform
    PlusCrossWaveform
    ProjectedWaveform
    HarmonicProjectedWaveform
    HomogeneousHarmonicProjectedWaveform

.. _spectral_density_matrices:

Spectral Density Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^

Matrices of Cross Spectral Densities (CSDs) and Power Spectral Densities (PSDs).

These are fundamental objects in the construction of noise models and likelihoods.

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: type.rst
   :nosignatures:

   SpectralDensity
   DiagonalSpectralDensity
   EvolutionarySpectralDensity


Noise Models
^^^^^^^^^^^^

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: type.rst
   :nosignatures:

   FDNoiseModel
   TFNoiseModel

Likelihoods
^^^^^^^^^^^
.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
    :toctree: _generated
    :template: type.rst
    :nosignatures:

    FDWhittleLikelihood

Miscellaneous
^^^^^^^^^^^^^

.. currentmodule:: typed_lisa_toolkit.types.misc

.. autosummary::
   :toctree: _generated
   :template: type.rst
   :nosignatures:

   Array
   ArrayFunc
   Interpolator
   Axis
   Grid1D
   Grid2DCartesian
   Grid2DSparse
   Grid2D
   UniformGrid2D
   AnyGrid
   Domain

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: type.rst
   :nosignatures:

   Linspace
   IntegrationMethod
   IntegrationPolicy

"""

from . import (
    data,
    likelihood,
    misc,
    modes,
    noisemodel,
    representations,
    tapering,
    waveforms,
)
from .data import (
    FSData,
    STFTData,
    TimedFSData,
    TSData,
    WDMData,
)
from .likelihood import FDWhittleLikelihood, Likelihood, WhittleLikelihood
from .misc import (
    AnyGrid,
    Array,
    ArrayFunc,
    Axis,
    Domain,
    Grid1D,
    Grid2D,
    Grid2DCartesian,
    Grid2DSparse,
    Interpolator,
    Linspace,
    UniformGrid2D,
)
from .modes import Harmonic, QuasiNormalMode
from .noisemodel import (
    DiagonalSpectralDensity,
    EvolutionarySpectralDensity,
    FDNoiseModel,
    IntegrationMethod,
    IntegrationPolicy,
    SpectralDensity,
    TFNoiseModel,
)
from .representations import (
    FrequencySeries,
    Phasor,
    ShortTimeFourierTransform,
    TimeSeries,
    UniformFrequencySeries,
    UniformTimeSeries,
    WilsonDaubechiesMeyer,
)
from .waveforms import (
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    HomogeneousHarmonicProjectedWaveform,
    HomogeneousHarmonicWaveform,
    PlusCrossWaveform,
    ProjectedWaveform,
)

QNM = QuasiNormalMode
"""Alias for :class:`.QuasiNormalMode`."""

STFT = ShortTimeFourierTransform
"""Alias for :class:`.ShortTimeFourierTransform`."""

WDM = WilsonDaubechiesMeyer
"""Alias for :class:`.WilsonDaubechiesMeyer`."""

IntegrationMethod = IntegrationMethod  # noqa: PLW0127
"""Name of the numerical integration method."""

__all__ = [
    "QNM",
    "STFT",
    "WDM",
    "AnyGrid",
    "Array",
    "ArrayFunc",
    "Axis",
    "DiagonalSpectralDensity",
    "Domain",
    "EvolutionarySpectralDensity",
    "FDNoiseModel",
    "FDWhittleLikelihood",
    "FSData",
    "FrequencySeries",
    "Grid1D",
    "Grid2D",
    "Grid2DCartesian",
    "Grid2DSparse",
    "Harmonic",
    "HarmonicProjectedWaveform",
    "HarmonicWaveform",
    "HomogeneousHarmonicProjectedWaveform",
    "HomogeneousHarmonicWaveform",
    "IntegrationMethod",
    "IntegrationPolicy",
    "Interpolator",
    "Likelihood",
    "Linspace",
    "Phasor",
    "PlusCrossWaveform",
    "ProjectedWaveform",
    "QuasiNormalMode",
    "STFTData",
    "SpectralDensity",
    "TFNoiseModel",
    "TSData",
    "TimeSeries",
    "TimedFSData",
    "UniformFrequencySeries",
    "UniformGrid2D",
    "UniformTimeSeries",
    "WDMData",
    "WhittleLikelihood",
    "data",
    "likelihood",
    "misc",
    "modes",
    "noisemodel",
    "representations",
    "tapering",
    "waveforms",
]
