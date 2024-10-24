"""Module for likelihood manipulation and computation."""

import logging
from typing import TypeVar, Protocol

import numpy as np
import numpy.typing as npt

from . import arithdicts
from . import waveforms as wf_
from . import sensitivity as sens_
from . import data as data_

log = logging.getLogger(__name__)

VT = TypeVar(
    "VT",
    npt.NDArray[np.floating],
    arithdicts.ChannelDict[np.floating | npt.NDArray[np.floating]],
)


class Likelihood(Protocol):
    """Protocol for likelihoods."""

    @classmethod
    def log_likelihood_ratio(cls, cross_product: VT, template_square: VT) -> VT:
        """Compute the log likelihood ratio."""
        return cross_product - 0.5 * template_square

    @classmethod
    def log_likelihood(cls, log_likelihood_ratio: VT, data_square: VT) -> VT:
        """Compute the log likelihood."""
        return log_likelihood_ratio - 0.5 * data_square

    def get_log_likelihood(self, template) -> arithdicts.ChannelDict[np.floating]:
        """Get the log likelihood."""

    def get_log_likelihood_ratio(self, template) -> arithdicts.ChannelDict[np.floating]:
        """Get the log likelihood ratio."""


class WhittleLikelihood(Likelihood):
    """Whittle likelihood.

    Using this likelihood requires assuming the noise is stationary and Gaussian.
    """

    def __init__(self, data: data_.FSData, sensitivity: sens_.FDSensitivity):
        self.data = data
        self.sensitivity = sensitivity.get_cache(
            sensitivity.get_noise_psd(sens_._collect_frequencies(data))
        )
        self.data_square = self.sensitivity.get_scalar_product(data, data)

    def get_log_likelihood(
        self, template: wf_.FDTemplate
    ) -> arithdicts.ChannelDict[np.floating]:
        """Get the log likelihood."""
        return self.log_likelihood(self.get_log_likelihood_ratio(template), self.data_square)

    def get_log_likelihood_ratio(self, template: wf_.FDTemplate):
        """Get the log likelihood ratio."""
        return self.log_likelihood_ratio(
            self.get_cross_product(template), self.get_template_square(template)
        )

    def get_cross_product(self, template: wf_.FDTemplate):
        """Get the cross product."""
        template_waveform, f_interval = self._process(template)
        return self.sensitivity.get_scalar_product(
            self.data.get_subset(interval=f_interval),
            template_waveform.get_subset(interval=f_interval),
        )

    def get_template_square(self, template: wf_.FDTemplate):
        """Get the template square."""
        template_waveform, f_interval = self._process(template)
        template_waveform_ = template_waveform.get_subset(interval=f_interval)
        return self.sensitivity.get_scalar_product(
            template_waveform_,
            template_waveform_,
        )

    def _process(self, template: wf_.FDTemplate):
        """Process the template."""
        internal_frequencies = template.get_frequencies()
        f_min = min(f[0] for f in internal_frequencies.values())
        f_max = max(f[-1] for f in internal_frequencies.values())
        f_interval = f_min, f_max
        data_frequencies = self.data.frequencies
        template_waveform = wf_.to_fsdata(template.get_waveform(data_frequencies))
        return template_waveform, f_interval
