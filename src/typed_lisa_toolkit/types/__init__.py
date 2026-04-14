"""
Types provided by TLT.

When possible, prefer to construct all objects with top-level functions (such as
:func:`~typed_lisa_toolkit.time_series`) instead of directly instantiating the classes
here (such as :class:`UniformTimeSeries`). The top-level functions tend to have a nicer
API. Ideally, use the classes here only for adding type hints to your code.

Not all types have top-level factory functions available. For example, you still need to
instantiate :class:`SpectralDensity` and :class:`FDNoiseModel` directly.

.. _representation_types:

Representation Types
^^^^^^^^^^^^^^^^^^^^

Representations are the building blocks of all other TLT types. They behave like arrays,
but carry time and frequency grids and other metadata. They are all ultimately different
ways to represent a time series.

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

Data objects carry multiple TDI channels of LISA data, all in the same representation.

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

Similarly to data objects, waveform objects group multiple series together in the same
sort of representation. Unlike data objects, they can carry metadata such as harmonic
content.

Waveforms are h+, hx, while "projected" waveforms are TDI channels.

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

Matrices of Power Spectral Densities (PSDs) and Cross Spectral Densities (CSDs) carry
two TDI channel indices, unlike data and projected waveforms which only carry one.

The utility of these matrices is in constructing noise models and likelihoods.

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
