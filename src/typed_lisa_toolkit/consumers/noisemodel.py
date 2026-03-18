"""Module for the noise model.

A noise model carries the knowledge of noise properties in the dm.
It defines the inner product and determines how to whiten the dm.

.. currentmodule:: typed_lisa_toolkit.consumers.noisemodel

Protocols
---------

.. autoclass:: IntegrationPolicy
   :members:
   :member-order: groupwise

Spectral Density Matrices
-------------------------

.. autoclass:: SpectralDensity
   :members:
   :member-order: groupwise
   :inherited-members:

.. autoclass:: DiagonalSpectralDensity
    :members:
    :member-order: groupwise
    :inherited-members:

.. autoclass:: EvolutionarySpectralDensity
    :members:
    :member-order: groupwise
    :inherited-members:

Noise Models
------------

.. autoclass:: FDNoiseModel
    :members:
    :member-order: groupwise
    :inherited-members:

.. autoclass:: TFNoiseModel
    :members:
    :member-order: groupwise
    :inherited-members:

"""

import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, Sequence, Union

from .. import utils
from ..containers import data as dm
from ..containers import representations as reps
from ..containers import waveforms


def _import_quadax() -> ModuleType:
    try:
        import quadax

        return quadax
    except ImportError:
        raise ImportError(
            "Default JAX-backed integration requires quadax. Install it with: pip install quadax"
        ) from None


def _import_scipy_integrate() -> ModuleType:
    try:
        import scipy.integrate  # type: ignore[import]

        return scipy.integrate
    except ImportError:
        raise ImportError(
            "Default Numpy-backed integration requires scipy. Install it with: pip install scipy"
        ) from None


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    Array = reps.Array
    Axis = reps.Axis
    Linspace = reps.Linspace


ChnName = str
FDEntry = dm.FSData | waveforms.ProjectedWaveform[reps.FrequencySeries["Axis"]]
TFEntry = dm.WDMData | waveforms.ProjectedWaveform[reps.WDM]
IntegrationMethod = Literal["trapezoid", "simpson"]


def _first_frequencies(__x: FDEntry):
    return next(iter(__x.values())).frequencies  # type: ignore[attr-defined] # mypy complains mistakenly


def _first_entries(__x: FDEntry):
    return next(iter(__x.values())).entries  # type: ignore[attr-defined] # mypy complains mistakenly


class IntegrationPolicy(Protocol):
    """Protocol for quadrature policies used by noise models."""

    def integrate(
        self,
        __y: "Array",
        *,
        x: Union["Array", None] = None,
        **kwargs: Any,
    ) -> Any:
        """Integrate the given dm."""
        ...

    def cumulative(
        self,
        __y: "Array",
        *,
        x: Union["Array", None] = None,
        **kwargs: Any,
    ) -> Any:
        """Integrate the given data cumulatively."""
        ...


class _NumpyIntegrationPolicy(IntegrationPolicy):
    """Numpy-backed integration policy with selectable quadrature method."""

    def __init__(
        self,
        method: IntegrationMethod = "trapezoid",
    ) -> None:
        self.method = method

    def integrate(
        self, __y: "Array", *, x: Union["Array", None] = None, **kwargs: Any
    ) -> Any:
        scipy_integrate = _import_scipy_integrate()
        return getattr(scipy_integrate, self.method)(__y, x=x, **kwargs)

    def cumulative(
        self, __y: "Array", *, x: Union["Array", None] = None, **kwargs: Any
    ) -> Any:
        scipy_integrate = _import_scipy_integrate()
        return getattr(scipy_integrate, "cumulative_" + self.method)(__y, x=x, **kwargs)


class _JaxIntegrationPolicy(IntegrationPolicy):
    """Jax-backed integration policy with selectable quadrature method."""

    def __init__(
        self,
        method: IntegrationMethod = "trapezoid",
    ) -> None:
        self.method = method

    def integrate(
        self, __y: "Array", *, x: Union["Array", None] = None, **kwargs: Any
    ) -> Any:
        quadax = _import_quadax()
        return getattr(quadax, self.method)(__y, x=x, **kwargs)

    def cumulative(
        self, __y: "Array", *, x: Union["Array", None] = None, **kwargs: Any
    ) -> Any:
        quadax = _import_quadax()
        return getattr(quadax, "cumulative_" + self.method)(__y, x=x, **kwargs)


def _make_integration_policy(
    xp: ModuleType,
    method: IntegrationMethod = "trapezoid",
) -> IntegrationPolicy:
    """Build an integration policy from a method name."""
    module_name = xp.__name__
    if module_name == "numpy":
        return _NumpyIntegrationPolicy(method=method)
    if module_name == "jax.numpy":
        return _JaxIntegrationPolicy(method=method)
    raise NotImplementedError(
        f"Unsupported array module {xp}. Cannot create integration policy."
    )


class _StationaryFDNoise(Protocol):
    """Protocol for frequency domain stationary noise PSD models."""

    def psd(self, frequencies: "Array", option: ChnName) -> "Array":
        """Return the power spectral density (PSD) values on the given frequency grid for the specified channel."""
        ...


class SpectralDensity:
    """Represent the spectral density matrix (SDM) of a frequency domain stationary noise model.

    .. note::

        The SDM is a real-valued, symmetric, positive-definite matrix, hence it is invertible.
        It is the inverse SDM that is used in the inner product and whitening operations,
        and we store the inverse SDM in this representation.
    """

    def __init__(
        self,
        frequencies: "Array",
        inverse_sdm: "Array",
        channel_order: Sequence[ChnName],
    ):
        # kernel shape: (n_freqs, n_channels, n_channels)
        self._frequencies = frequencies
        self._inverse_sdm = inverse_sdm
        self.channel_order = channel_order

    def to_subband(self, f_interval: tuple[float, float]) -> Self:
        """Return a new SpectralDensity instance with the frequency grid restricted to the given subband."""
        f_min, f_max = f_interval
        _slice = utils.get_subset_slice(self._frequencies, f_min, f_max)
        return type(self)(
            self._frequencies[_slice],
            self._inverse_sdm[_slice],
            self.channel_order,
        )

    def get_kernel(self, backend: str | None = None) -> "Array":
        """Return the inverse of the spectral density matrix.

        The inverse SDM is returned as an array of shape
        ``(n_freqs, n_channels, n_channels)``.

        .. note:: We denote the inverse SDM as :math:`S_n^{-1}`.
        """
        if backend is not None:
            raise NotImplementedError("Backend conversion is not implemented yet.")
        return self._inverse_sdm

    def get_whitening_matrix(
        self, kind: Literal["cholesky"] | None = "cholesky"
    ) -> "Array":
        r"""Return whitening matrix :math:`W` with shape ``(n_freqs, n_channels, n_channels)``.

        The whitening matrix represents a linear transformation that
        yields unit variance white noise when applied to noise that
        follows this model. This is useful in detecting deviations from the model.

        .. note:: :math:`W` satisfies :math:`S_n^{-1} = W^\top W`.
        """
        if kind != "cholesky":
            raise NotImplementedError(f"Unsupported whitening matrix kind {kind}.")
        xp = self._inverse_sdm.__array_namespace__()
        return xp.linalg.cholesky(self._inverse_sdm, upper=True)


class DiagonalSpectralDensity(SpectralDensity):
    """Represent a SDM for a frequency domain stationary noise model with no inter-channel correlations."""

    @property
    def is_diagonal(self) -> Literal[True]:
        """Return True."""
        return True

    @classmethod
    def from_fd_noise(
        cls,
        fd_noise: _StationaryFDNoise,
        frequencies: "Array",
        channel_names: tuple[ChnName, ...],
    ):
        """Create a SpectralDensity instance from a frequency domain noise model and a frequency grid."""
        _dict = {
            chnname: 1 / fd_noise.psd(frequencies, option=chnname)
            for chnname in channel_names
        }
        xp = next(iter(_dict.values())).__array_namespace__()
        diag = xp.stack(
            [xp.squeeze(_dict[c]) for c in channel_names], axis=-1
        )  # (n_freqs, n_channels)
        kernel = diag[:, :, None] * xp.eye(
            len(channel_names), dtype=diag.dtype
        )  # (n_freqs, n_channels, n_channels)
        return cls(frequencies, kernel, channel_names)

    def get_whitening_matrix(self, kind: Literal["cholesky"] | None = None) -> "Array":
        r"""Return whitening matrix :math:`W` with shape ``(n_freqs, n_channels, n_channels)``.

        The whitening matrix represents a linear transformation that
        yields unit variance white noise when applied to noise that
        follows this model. This is useful in detecting deviations from the model.

        .. note:: :math:`W` satisfies :math:`S_n^{-1} = W^\top W`.
        """
        if kind is not None:
            return super().get_whitening_matrix(kind=kind)
        xp = self._inverse_sdm.__array_namespace__()
        # S_n^{-1} is diagonal => W = sqrt(S_n^{-1})
        diag = xp.diagonal(self._inverse_sdm, axis1=-2, axis2=-1)
        return xp.sqrt(diag)[:, :, None] * xp.eye(
            len(self.channel_order), dtype=diag.dtype
        )


class FDNoiseModelLike(Protocol):
    """Protocol for frequency domain noise models."""

    def get_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        """Return the scalar product."""
        ...

    def whiten(self, _data: FDEntry) -> FDEntry:
        """Whiten the data according to the noise model."""
        ...

    def to_subband(self, f_interval: tuple[float, float]) -> Self:
        """Return a new noise model with the frequency grid restricted to the given subband."""
        ...


class FDNoiseModel[DensityT: SpectralDensity]:
    """Frequency domain noise model.

    Assuming the noise is stationary, the noise model is given by the
    noise power spectral density (PSD) in the frequency domain. This
    class might not be suitable for non-stationary noise.

    """

    def __init__(
        self,
        psd: DensityT,
        integration_method: IntegrationMethod = "trapezoid",
    ):
        self._psd = psd  # Keep the original PSD object for potential future use (e.g., subband restriction)
        self.psd = psd
        xp = psd.get_kernel().__array_namespace__()
        self._ip = _make_integration_policy(xp, integration_method)

    def reset(self) -> Self:
        """Reset the noise model to its original state."""
        self.psd = self._psd
        return self

    def to_subband(self, f_interval: tuple[float, float]) -> Self:
        """Restrict the noise model to a subband."""
        self.psd = self._psd.to_subband(f_interval)
        return self

    def _get_whitened_entries(self, _data: FDEntry) -> "Array":
        """Return the whitened kernel entries of the given dm."""
        kernel = _data.get_kernel()  # (n_batches, n_ch, 1, 1, n_freqs)
        xp = kernel.__array_namespace__()
        W = self.psd.get_whitening_matrix()  # (n_freqs, n_ch, n_ch)
        whitened_e = xp.einsum("fij,...fj->...fi", W, xp.moveaxis(kernel, 1, -1))
        return xp.moveaxis(whitened_e, -1, 1)

    def get_integrand(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the frequency-domain inner-product integrand.

        Computes :math:`4\, d^*(f)\, S_n^{-1}(f)\, h(f)` at each frequency bin,
        returning an array of shape ``(n_batches, n_channels, 1, 1, n_freqs)``.
        """
        _left = left.get_kernel()  # shape (n_batches, n_channels, 1, 1, n_freqs)
        _right = right.get_kernel()  # same shape as _left
        xp = _left.__array_namespace__()
        try:
            if getattr(self.psd, "is_diagonal"):
                # If the spectral density matrix is diagonal, we can simply divide by the diagonal elements.
                diag = xp.diagonal(
                    self.psd.get_kernel(), axis1=-1, axis2=-2
                )  # shape (n_freqs, n_channels)
                return (4 * _left.conj() * _right) * diag.T[None, :, None, None, :]
        except AttributeError:
            pass
        return 4 * xp.einsum(
            "...fi,fij,...fj->...f",
            xp.moveaxis(_left.conj(), 1, -1),
            self.psd.get_kernel(),
            xp.moveaxis(_right, 1, -1),
        )

    def get_complex_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the complex scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \langle d, h \rangle = 4 \int_{f_\text{min}}^{f_\text{max}} \frac{d^*(f) h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        frequencies = _first_frequencies(left)
        xp = _first_entries(left).__array_namespace__()
        return self._ip.integrate(
            self.get_integrand(left, right), x=reps.to_array(frequencies, xp=xp)
        )

    def get_cumulative_complex_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the cumulative complex scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns an array of the following function on the input frequency grid
        :math:`[f_\text{min}, f_\text{max}]`:

        .. math::

            F \mapsto 4\int_{f_\text{min}}^{F} \frac{d^*(f) h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        frequencies = _first_frequencies(left)
        xp = _first_entries(left).__array_namespace__()
        return self._ip.cumulative(
            self.get_integrand(left, right),
            x=reps.to_array(frequencies, xp=xp),
            initial=0,
        )

    def get_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \left( d \middle| h \right) = 4 \Re \int_{f_\text{min}}^{f_\text{max}} \frac{d^*(f) h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        return self.get_complex_scalar_product(left, right).real

    inner = get_scalar_product
    """Alias for :meth:`get_scalar_product`."""

    def get_cumulative_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the cumulative scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns an array of the following function on the input frequency grid
        :math:`[f_\text{min}, f_\text{max}]`:

        .. math::

            F \mapsto 4\Re \int_{f_\text{min}}^{F} \frac{d^*(f) h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        return self.get_cumulative_complex_scalar_product(left, right).real

    def get_cross_correlation(
        self,
        left: dm.TimedFSData,
        right: FDEntry,
    ):
        r"""Return the cross correlation.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        we define the cross-correlation as

        .. math::

            (d \ast h)(\tau) := \left( \hat{d}(t) \middle| \hat{h}(t + \tau) \right),

        where the hat denotes the real Fourier transform. This methods returns a generalization of the above
        cross-correlation which is **complex-valued**

        .. math::

            (d \star h)(\tau) := \langle \hat{d}(t), \hat{h}(t + \tau) \rangle.

        In the implementation, the generalized cross-correlation is computed using

        .. math::

            (d \star h)(\tau) \propto \mathcal{F}^{-1}\left(4\frac{d^*(f) h(f)}{S_n(f)}\right),

        where :math:`\mathcal{F}^{-1}` is the two-sided inverse Fourier transform. Note
        that the input arrays :math:`d(f)` and :math:`h(f)` are one-sided, and the negative frequencies are
        populated by **zero** before the inverse Fourier transform.
        """
        xp = _first_entries(left).__array_namespace__()
        two_sided_freq = xp.fft.fftshift(
            xp.fft.fftfreq(len(left.times), left.times.step)
        )
        _first = next(iter(left.values()))
        frequencies, df = reps.to_array(_first.frequencies, xp), _first.df
        two_sided_integrand_entries = utils.extend_to(two_sided_freq)(
            frequencies, self.get_integrand(left, right)
        )
        cross_correlation = xp.fft.ifft(
            xp.fft.ifftshift(two_sided_integrand_entries) * df,
            len(left.times),
            norm="forward",
            axis=-1,
        )
        return dm.TSData(
            reps.UniformTimeSeries((left.times,), cross_correlation), left.channel_names
        )

    def whiten(self, _data: FDEntry):
        r"""Return whitened data with the same container type as input.

        Applies the whitening matrix, so whitened noise has unit covariance.
        """
        d_k = _data.get_kernel()  # (n_batches, n_ch, 1, 1, n_freqs)
        xp = d_k.__array_namespace__()
        W = self.psd.get_whitening_matrix()  # (n_freqs, n_ch, n_ch)
        d_e = xp.moveaxis(d_k[:, :, 0, 0, :], 1, -1)  # (n_batches, n_freqs, n_ch)
        whitened_e = xp.einsum("fij,...fj->...fi", W, d_e)  # (n_batches, n_freqs, n_ch)
        whitened_k = xp.moveaxis(whitened_e, -1, 1)[:, :, None, None, :]
        # whitened_repr = _data.representation.create_like(whitened_k)
        return _data.create_like(whitened_k, _data.channel_names)

    def get_overlap(self, left: FDEntry, right: FDEntry) -> "Array":
        r"""Return the overlap.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \frac{\langle d, h \rangle}{\sqrt{\langle d, d \rangle \langle h, h \rangle}}.
        """
        xp = _first_entries(left).__array_namespace__()
        return self.get_scalar_product(left, right) / xp.sqrt(
            self.get_scalar_product(left, left) * self.get_scalar_product(right, right)
        )


class EvolutionarySpectralDensity:
    """Evolutionary spectral density matrix (ESDM) for a time-frequency noise model.

    Stores the inverse ESDM as an array of shape ``(n_freqs, n_times, n_channels, n_channels)``.
    """

    def __init__(
        self,
        frequencies: "Array",
        times: "Array",
        inverse_esdm: "Array",
        channel_order: Sequence[ChnName],
    ):
        _ = self.is_valid_sdm(
            inverse_esdm,
            raise_exception=True,
            channel_order=channel_order,
        )
        # kernel shape: (n_freqs, n_times, n_channels, n_channels)
        self._frequencies = frequencies
        self._times = times
        self._inverse_esdm = inverse_esdm
        self.channel_order = channel_order

    def get_kernel(self, backend: str | None = None) -> "Array":
        """Return the inverse of the evolutionary spectral density matrix.

        The returned array has shape ``(n_freqs, n_times, n_channels, n_channels)``.
        """
        if backend is not None:
            raise NotImplementedError("Backend conversion is not implemented yet.")
        return self._inverse_esdm

    @staticmethod
    def is_valid_sdm(
        _evsdm_or_invevsdm: "Array",
        /,
        *,
        channel_order: Sequence[ChnName],
        raise_exception: bool = False,
    ):
        """Check validity of the (inverse) evolutionary spectral density matrix."""
        _d = _evsdm_or_invevsdm
        # Check if channel_order is valid.
        if len(channel_order) != len(set(channel_order)):
            if raise_exception:
                raise ValueError(
                    f"Received channel_order {channel_order} with duplicate channels."
                )
            else:
                return False
        # Check if the shape is correct.
        if len(_d.shape) != 4 or _d.shape[-2:] != (
            len(channel_order),
            len(channel_order),
        ):
            if raise_exception:
                raise ValueError(
                    "Expected (inverse) evolutionary spectral density matrix",
                    "to have shape (n_freq, n_time, n_channels, n_channels), ",
                    f"but got shape {_d.shape} instead.",
                )
            else:
                return False
        return True

    def get_whitening_matrix(self, kind: Literal["cholesky"] = "cholesky") -> "Array":
        r"""Return whitening matrix :math:`W` with shape ``(n_freqs, n_times, n_channels, n_channels)``.

        .. note::

            Currently only supports Cholesky decomposition: :math:`W`
            satisfies :math:`S_n^{-1} = W^\top W`.

        The whitening matrix represents a linear transformation that
        yields unit variance white noise when applied to noise that
        follows this model. This is useful in detecting deviations from the model.

        Parameters
        ----------
        kind: the kind of whitening matrix. Defaults to "cholesky".
        """
        if kind != "cholesky":
            raise NotImplementedError(f"Unsupported whitening matrix kind {kind}.")
        xp = self._inverse_esdm.__array_namespace__()
        return xp.linalg.cholesky(self._inverse_esdm, upper=True)


class TFNoiseModel:
    """Time-frequency Gaussian noise model.

    This model is a Gaussian noise model suitable for non-stationary noise.
    The covariance matrix is diagonal in the chosen time-frequency representation.
    The model allows correlations between TDI channels. In other words, we have a 3x3
    symmetric matrix at each location in the time-frequency plane.
    """

    def __init__(
        self,
        esd: EvolutionarySpectralDensity,
    ):
        self.esd = esd

    def _get_whitened_entries(self, _data: TFEntry) -> "Array":
        """Return the whitened kernel entries of the given dm."""
        kernel = _data.get_kernel()  # (n_batches, n_ch, 1, 1, n_freqs, n_times)
        xp = kernel.__array_namespace__()
        W = self.esd.get_whitening_matrix()  # (n_freqs, n_times, n_ch, n_ch)
        whitened_e = xp.einsum("ftij,...ftj->...fti", W, xp.moveaxis(kernel, 1, -1))
        return xp.moveaxis(whitened_e, -1, 1)

    def get_scalar_product(
        self,
        left: TFEntry,
        right: TFEntry,
    ) -> "Array":
        """Return the scalar product."""
        _left = left.get_kernel()  # shape (n_batches, n_channels, 1, 1, n_freq, n_time)
        _right = right.get_kernel()  # same shape as _left
        xp = _left.__array_namespace__()
        return (
            xp.einsum(
                "...fti,ftij,...ftj->...ft",
                xp.moveaxis(_left.conj(), 1, -1),
                self.esd.get_kernel(),
                xp.moveaxis(_right, 1, -1),
            )
            .sum()
            .real
        )

    inner = get_scalar_product
    """Alias for :meth:`get_scalar_product`."""

    def whiten(self, _data: TFEntry) -> TFEntry:
        """Whiten the data according to the noise model."""
        whitened_array = self._get_whitened_entries(_data)
        whitened_repr = _data.representation.create_like(whitened_array)
        return _data.create_new(whitened_repr, _data.channel_names)
