"""
Types provided by the package.

.. _representation_types:

Representation Types
^^^^^^^^^^^^^^^^^^^^

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

Data Types
^^^^^^^^^^

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

Waveform Types
^^^^^^^^^^^^^^

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: class.rst
   :nosignatures:

    HarmonicWaveform
    HomogeneousHarmonicWaveform
    ProjectedWaveform
    HarmonicProjectedWaveform
    HomogeneousHarmonicProjectedWaveform

.. _spectral_density_matrices:

Spectral Density Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: type.rst
   :nosignatures:

   SpectralDensity
   DiagonalSpectralDensity
   EvolutionarySpectralDensity


Noise Model Types
^^^^^^^^^^^^^^^^^

.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
   :toctree: _generated
   :template: type.rst
   :nosignatures:

   FDNoiseModelLike
   FDNoiseModel
   TFNoiseModel

Likelihood Types
^^^^^^^^^^^^^^^^
.. currentmodule:: typed_lisa_toolkit.types

.. autosummary::
    :toctree: _generated
    :template: type.rst
    :nosignatures:

    Likelihood
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
from .likelihood import FDWhittleLikelihood, Likelihood
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
    FDNoiseModelLike,
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
    ProjectedWaveform,
)

QNM = QuasiNormalMode
"""Alias for :class:`.QuasiNormalMode`."""

STFT = ShortTimeFourierTransform
"""Alias for :class:`.ShortTimeFourierTransform`."""

WDM = WilsonDaubechiesMeyer
"""Alias for :class:`.WilsonDaubechiesMeyer`."""

IntegrationMethod = IntegrationMethod
"""Name of the numerical integration method."""

__all__ = [
    "data",
    "modes",
    "representations",
    "tapering",
    "waveforms",
    "noisemodel",
    "likelihood",
    "misc",
    "Linspace",
    "Array",
    "ArrayFunc",
    "Interpolator",
    "Axis",
    "Grid1D",
    "Grid2DCartesian",
    "Grid2DSparse",
    "Grid2D",
    "UniformGrid2D",
    "AnyGrid",
    "Domain",
    "FrequencySeries",
    "UniformFrequencySeries",
    "Phasor",
    "TimeSeries",
    "UniformTimeSeries",
    "WDM",
    "STFT",
    "HarmonicProjectedWaveform",
    "HomogeneousHarmonicProjectedWaveform",
    "ProjectedWaveform",
    "HomogeneousHarmonicWaveform",
    "HarmonicWaveform",
    "TSData",
    "FSData",
    "TimedFSData",
    "WDMData",
    "STFTData",
    "Harmonic",
    "QuasiNormalMode",
    "QNM",
    "IntegrationMethod",
    "IntegrationPolicy",
    "SpectralDensity",
    "DiagonalSpectralDensity",
    "EvolutionarySpectralDensity",
    "FDNoiseModelLike",
    "FDNoiseModel",
    "TFNoiseModel",
    "Likelihood",
    "FDWhittleLikelihood",
]
