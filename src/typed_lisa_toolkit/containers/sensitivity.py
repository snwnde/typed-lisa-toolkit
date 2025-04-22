"""(Deprecated) Module for sensitivity.

Sensitivity carries the knowledge of noise properties in the data.
It determines the geometry of the data space, of which the template
space is a subspace.

.. currentmodule:: typed_lisa_toolkit.containers.sensitivity

Types
-----

.. autoprotocol:: FDNoiseModel

Entities
--------

.. autoclass:: FDSensitivity
   :members:
   :member-order: groupwise
   :inherited-members:

"""

import warnings
import logging
from typing import Protocol, Unpack

import numpy as np
import numpy.typing as npt
import scipy.integrate  # type: ignore[import]

from . import arithdicts
from . import representations as reps
from . import data
from . import noisemodel

log = logging.getLogger(__name__)

ChnName = str


class FDNoiseModel(Protocol):
    """Protocol for frequency domain noise models."""

    def psd(
        self, frequencies: npt.NDArray[np.floating], option: ChnName
    ) -> npt.NDArray[np.floating]:
        """Return the power spectral density (PSD)."""


class Sensitivity:
    """Base sensitivity class."""


class FDSensitivity(noisemodel.FDNoiseModel):
    """(Deprecated) Frequency domain noise model.

    Use :class:`.noisemodel.FDNoiseModel` instead.
    """

    def __init__(
        self,
        integrator=scipy.integrate.trapezoid,
        cumulative_integrator=scipy.integrate.cumulative_trapezoid,
    ):
        warnings.warn(
            "FDSensitivity is deprecated and will be removed in a future release. "
            "Use noisemodel.FDNoiseModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(integrator, cumulative_integrator)

    @classmethod
    def make(  # noqa: D102
        cls,
        noise_model: FDNoiseModel | None = None,
        noise_cache: data.FSData | None = None,
        **kwargs: Unpack[noisemodel.IntegratorConfig],
    ):
        args = (noise_model, noise_cache)
        error = ValueError(
            "Exactly one of fd_noise and noise_cache should be provided."
        )
        if sum(arg is not None for arg in args) != 1:
            raise error
        if noise_model is not None:
            return _NoiseModelSensitivity(noise_model, **kwargs)
        if noise_cache is not None:
            return _CacheSensitivity(noise_cache, **kwargs)
        raise error


class _NoiseModelSensitivity(FDSensitivity):
    def __init__(self, noise_model: FDNoiseModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_model = noise_model

    def create_new(self, integrator, cumulative_integrator):
        return type(self)(self.noise_model, integrator, cumulative_integrator)

    def get_noise_psd(
        self, frequencies: arithdicts.ChannelDict[npt.NDArray[data.NPFloatingT]]
    ):
        def make_psd(chnname: ChnName):
            freq = frequencies[chnname]
            entries = self.noise_model.psd(freq, option=chnname)
            return reps.FrequencySeries(freq, entries)

        _dict = {chnname: make_psd(chnname) for chnname in frequencies.keys()}
        return data.FSData(_dict)


class _CacheSensitivity(FDSensitivity):
    def __init__(self, noise_cache: data.FSData, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_cache = noise_cache

    def create_new(self, integrator, cumulative_integrator):
        return type(self)(self.noise_cache, integrator, cumulative_integrator)

    def get_noise_psd(
        self, frequencies: arithdicts.ChannelDict[npt.NDArray[data.NPFloatingT]]
    ):
        freq = next(iter(frequencies.values()))
        try:
            interval = freq[0], freq[-1]
        except IndexError:
            interval = (0, 0)  # An empty interval.
        noise_psd = self.noise_cache.get_subset(interval=interval)
        if log.isEnabledFor(logging.DEBUG):
            try:
                assert np.array_equal(
                    noise_psd.frequencies, freq
                ), f"The frequencies {noise_psd.frequencies} and {freq} are not exactly equal."
            except AssertionError as e:
                log.debug(e)
                try:
                    assert np.allclose(
                        noise_psd.frequencies, freq
                    ), f"The frequencies {noise_psd.frequencies} and {freq} are not close."
                except AssertionError as exc:
                    log.debug(exc)
        return noise_psd
