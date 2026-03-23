"""Module for likelihood manipulation and computation.

.. currentmodule:: typed_lisa_toolkit.consumers.likelihood


.. autoclass:: FDWhittleLikelihood
   :members:
   :member-order: groupwise
   :inherited-members:

"""

import logging
from typing import TYPE_CHECKING, Any, Protocol

from ..containers import data as dm
from ..containers import modes
from ..containers import representations as reps
from ..containers import waveforms as wf
from . import noisemodel as nm

if TYPE_CHECKING:
    Array = reps.Array
    Axis = reps.Axis
    Linspace = reps.Linspace
    Reps = reps.Representation


ChnName = str
# Reps = reps.TimeSeries | reps.FrequencySeries | reps.STFT | reps.Phasor | reps.WDM
Modes = modes.Harmonic | modes.QNM
FDUniformHomogeneous = (
    dm.Data[reps.FrequencySeries["Linspace"]]
    | wf.ProjectedWaveform[reps.FrequencySeries["Linspace"]]
    | wf.HomogeneousHarmonicProjectedWaveform[Modes, reps.FrequencySeries["Linspace"]]
)

log = logging.getLogger(__name__)


class Likelihood[TemplateT: Any](Protocol):
    """Protocol for likelihoods."""

    def get_log_likelihood(
        self,
        template: TemplateT,
    ) -> "Array":
        """Get the log likelihood."""
        ...

    def get_log_likelihood_ratio(
        self,
        template: TemplateT,
    ) -> "Array":
        """Get the log likelihood ratio."""
        ...


class FDWhittleLikelihood(Likelihood[FDUniformHomogeneous]):
    r"""Whittle likelihood for frequency-domain data.

    Assuming the noise is additive, stationary and Gaussian.
    Note :math:`d` the data, :math:`h` the template, the log-likelihood is given by

    .. math::

        \log \mathcal{L} = -\frac{1}{2} \left( d - h \middle| d - h \right).

    We can rewrite this as

    .. math::

        \log \mathcal{L} = -\frac{1}{2} \left( d \middle| d \right) +
        \left( d \middle| h \right) -\frac{1}{2} \left( h \middle| h \right).

    The term :math:`\left( d \middle| d \right)` is computed upon initialization and is constant.
    """

    def __init__(self, data: dm.FSData, noisemodel: nm.FDNoiseModelLike):
        self.data = data
        self.noisemodel = noisemodel  # .reset()
        self.data_square = self.noisemodel.get_scalar_product(data, data)

    @classmethod
    def log_likelihood_ratio(cls, cross_product: "Array", template_square: "Array"):
        """Compute the log likelihood ratio."""
        return cross_product - 0.5 * template_square

    @classmethod
    def log_likelihood(cls, log_likelihood_ratio: "Array", data_square: "Array"):
        """Compute the log likelihood."""
        return log_likelihood_ratio - 0.5 * data_square

    def get_log_likelihood(
        self,
        template: FDUniformHomogeneous,
    ):
        """Get the log likelihood."""
        return self.log_likelihood(
            self.get_log_likelihood_ratio(template),
            self.data_square,
        )

    def get_log_likelihood_ratio(
        self,
        template: FDUniformHomogeneous,
    ):
        """Get the log likelihood ratio."""
        template_waveform, f_interval = self._process(template)
        _cross = self._get_cross_product(template_waveform, f_interval)
        _temp_sq = self._get_template_square(template_waveform)
        return self.log_likelihood_ratio(_cross, _temp_sq)

    def get_cross_product(
        self,
        template: FDUniformHomogeneous,
    ):
        r"""Get the cross product.

        This method computes the term :math:`\left( d \middle| h \right)`.
        """
        template_waveform, f_interval = self._process(template)
        return self._get_cross_product(template_waveform, f_interval)

    def get_template_square(
        self,
        template: FDUniformHomogeneous,
    ):
        r"""Get the template square.

        This method computes the term :math:`\left( h \middle| h \right)`.
        """
        template_waveform, _ = self._process(template)
        return self._get_template_square(template_waveform)

    def _get_cross_product(
        self,
        template: dm.FSData,
        f_interval: tuple[float, float],
    ):
        return self.noisemodel.get_scalar_product(
            self.data.get_subset(interval=f_interval),
            template,
        )

    def _get_template_square(self, template: dm.FSData):
        r"""Get the template square.

        This method computes the term :math:`\left( h \middle| h \right)`.
        """
        return self.noisemodel.get_scalar_product(
            template,
            template,
        )

    def _process(
        self,
        template: FDUniformHomogeneous,
    ):
        """Process the template."""
        xp = template.__xp__()
        if isinstance(template, wf.HomogeneousHarmonicProjectedWaveform):
            __temp = wf.sum_harmonics(template)
            freqs = reps.to_array(__temp.representation.frequencies, xp)
            _temp = dm.FSData(__temp.representation, __temp.channel_names)
        else:
            freqs = reps.to_array(template.representation.frequencies, xp)
            _temp = dm.FSData(template.representation, template.channel_names)
        try:
            f_interval = (
                float(freqs[0]),
                float(freqs[-1]),
            )
        except IndexError:
            f_interval = (0, 0)  # An empty interval
        self.noisemodel.to_subband(f_interval)
        return _temp.get_subset(interval=f_interval), f_interval
