"""Module for likelihood manipulation and computation.

.. currentmodule:: typed_lisa_toolkit.containers.likelihood

Types
-----

.. autoprotocol:: Likelihood

Entities
--------

.. autoclass:: WhittleLikelihood
   :members:
   :member-order: groupwise
   :exclude-members: log_likelihood, log_likelihood_ratio
   :inherited-members:

"""

import logging
from typing import TypeVar, Protocol

import numpy as np
import numpy.typing as npt

from . import arithdicts
from . import waveforms as wf
from . import sensitivity as sens
from . import data as data_

log = logging.getLogger(__name__)

VT = TypeVar(
    "VT",
    npt.NDArray[np.floating],
    arithdicts.ChannelDict[np.floating | npt.NDArray[np.floating]],
)


class Likelihood(Protocol):
    """Protocol for likelihoods."""

    def get_log_likelihood(self, template) -> arithdicts.ChannelDict[np.floating]:
        """Get the log likelihood."""

    def get_log_likelihood_ratio(self, template) -> arithdicts.ChannelDict[np.floating]:
        """Get the log likelihood ratio."""


class WhittleLikelihood(Likelihood):
    r"""Whittle likelihood.

    Assuming the noise is additive, stationary and Gaussian.
    Note :math:`d` the data, :math:`h` the template, the log-likelihood is given by

    .. math::

        \log \mathcal{L} = -\frac{1}{2} \left( d - h \middle| d - h \right).

    We can rewrite this as

    .. math::

        \log \mathcal{L} = -\frac{1}{2} \left( d \middle| d \right) + \left( d \middle| h \right) -\frac{1}{2} \left( h \middle| h \right).

    The term :math:`\left( d \middle| d \right)` is computed upon initialization and is constant.
    """

    def __init__(self, data: data_.FSData, sensitivity: sens.FDSensitivity):
        self.data = data
        self.sensitivity = sensitivity.get_cache(
            sensitivity.get_noise_psd(sens._collect_frequencies(data))
        )
        self.data_square = self.sensitivity.get_scalar_product(data, data)

    @classmethod
    def log_likelihood_ratio(cls, cross_product: VT, template_square: VT) -> VT:
        """Compute the log likelihood ratio."""
        return cross_product - 0.5 * template_square

    @classmethod
    def log_likelihood(cls, log_likelihood_ratio: VT, data_square: VT) -> VT:
        """Compute the log likelihood."""
        return log_likelihood_ratio - 0.5 * data_square

    def get_log_likelihood(
        self, template: wf.FormattedWaveform
    ) -> arithdicts.ChannelDict[np.floating]:
        """Get the log likelihood."""
        return self.log_likelihood(
            self.get_log_likelihood_ratio(template), self.data_square
        )

    def get_log_likelihood_ratio(self, template: wf.FormattedWaveform):
        """Get the log likelihood ratio."""
        return self.log_likelihood_ratio(
            self.get_cross_product(template), self.get_template_square(template)
        )

    def get_cross_product(self, template: wf.FormattedWaveform):
        r"""Get the cross product.

        This method computes the term :math:`\left( d \middle| h \right)`.
        """
        template_waveform, f_interval = self._process(template)
        return self.sensitivity.get_scalar_product(
            self.data.get_subset(interval=f_interval),
            template_waveform.get_subset(interval=f_interval),
        )

    def get_template_square(self, template: wf.FormattedWaveform):
        r"""Get the template square.

        This method computes the term :math:`\left( h \middle| h \right)`.
        """
        template_waveform, f_interval = self._process(template)
        template_waveform_ = template_waveform.get_subset(interval=f_interval)
        return self.sensitivity.get_scalar_product(
            template_waveform_,
            template_waveform_,
        )

    def _process(self, template: wf.FormattedWaveform):
        """Process the template."""
        template_waveform = wf.to_fsdata(template)
        f_interval = template_waveform.frequencies[0], template_waveform.frequencies[-1]
        return template_waveform, f_interval
