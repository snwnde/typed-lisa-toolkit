"""Likelihood types."""

import abc
import logging
from typing import TYPE_CHECKING, Any, Protocol, cast

from . import data as dm
from . import noisemodel as nm
from . import representations as reps
from . import waveforms as wf
from .misc import AnyGrid, Array, Linspace

if TYPE_CHECKING:
    AnyReps = reps.Representation[AnyGrid]
    AnyData = dm.Data[AnyReps]


ChnName = str
Mode = tuple[int, int] | tuple[int, int, int]
FDUniformHomogeneous = (
    dm.Data[reps.FrequencySeries[Linspace]]
    | wf.ProjectedWaveform[reps.FrequencySeries[Linspace]]
    | wf.HomogeneousHarmonicProjectedWaveform[Mode, reps.FrequencySeries[Linspace]]
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


class _DataProperty:
    def __get__[DataT: "AnyData"](
        self,
        instance: "WhittleLikelihood[nm.NoiseModelLike[DataT, Any]]",
        owner: Any,
    ) -> DataT: ...


class WhittleLikelihood[
    NoiseModelT: nm.NoiseModelLike[Any, Any],
](abc.ABC):
    """Abstract Whittle likelihood."""

    def __init__(self, data: "AnyData", noisemodel: nm.NoiseModelLike[Any, Any]):
        self._data: AnyData = data
        self._noisemodel: nm.NoiseModelLike[Any, Any] = noisemodel
        self.data_square: Array = self.noisemodel.get_scalar_product(data, data)

    @property
    def data(self):  # pyright: ignore[reportRedeclaration]
        """Get the data."""
        return self._data

    data: _DataProperty  # For correct type hinting

    @property
    def noisemodel(self) -> NoiseModelT:
        """Get the noise model."""
        return cast("NoiseModelT", self._noisemodel)

    @classmethod
    def log_likelihood_ratio(cls, cross_product: "Array", template_square: "Array"):
        """Compute the log likelihood ratio."""
        return cross_product - 0.5 * template_square

    @classmethod
    def log_likelihood(cls, log_likelihood_ratio: "Array", data_square: "Array"):
        """Compute the log likelihood."""
        return log_likelihood_ratio - 0.5 * data_square


class FDWhittleLikelihood(WhittleLikelihood[nm.FDNoiseModel]):
    r"""Whittle likelihood for frequency-domain data.

    Assuming the noise is additive, stationary and Gaussian.
    Note :math:`d` the data, :math:`h` the template, the log-likelihood is given by

    .. math::

        \log \mathcal{L} = -\frac{1}{2} \left( d - h \middle| d - h \right).

    We can rewrite this as

    .. math::

        \log \mathcal{L} = -\frac{1}{2} \left( d \middle| d \right) +
        \left( d \middle| h \right) -\frac{1}{2} \left( h \middle| h \right).

    The term :math:`\left( d \middle| d \right)` is computed upon
    initialization and is constant.

    Attention
    ---------
    This class is considered experimental. If you are interested in using it,
    please reach out to the
    developers to discuss your use case and how we can best support it.
    """

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
            freqs = __temp.grid[0].asarray(xp)
            _temp = dm.fsdata(__temp)
        else:
            freqs = template.grid[0].asarray(xp)
            _temp = dm.fsdata(template)
        try:
            f_interval = (
                float(freqs[0]),
                float(freqs[-1]),
            )
        except IndexError:
            f_interval = (0, 0)  # An empty interval
        self.noisemodel.to_subband(f_interval)
        return _temp.get_subset(interval=f_interval), f_interval


def whittle(
    data: dm.FSData,
    noisemodel: nm.FDNoiseModel,
) -> FDWhittleLikelihood:
    """Construct a WhittleLikelihood."""
    return FDWhittleLikelihood(data, noisemodel)
