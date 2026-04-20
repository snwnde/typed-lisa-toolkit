"""Noise model types."""

import logging
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import (
    Any,
    Literal,
    Protocol,
    Self,
    Union,
    overload,
)

import array_api_compat as xpc

from .. import utils
from . import _mixins, waveforms
from . import data as dm
from . import representations as reps
from .misc import Array, Axis, Domain, Grid2D, Linspace


def _import_quadax() -> ModuleType:
    try:
        import quadax
    except ImportError:
        msg = (
            "Default JAX-backed integration requires quadax. "
            "Install it with: pip install quadax"
        )
        raise ImportError(msg) from None
    else:
        return quadax


def _import_scipy_integrate() -> ModuleType:
    try:
        import scipy.integrate
    except ImportError:
        msg = (
            "Default Numpy-backed integration requires scipy. "
            "Install it with: pip install scipy"
        )
        raise ImportError(msg) from None
    else:
        return scipy.integrate


log = logging.getLogger(__name__)


ChnName = str
FDEntry = dm.FSData | waveforms.ProjectedWaveform[reps.FrequencySeries[Axis]]
TFEntry = (
    dm.WDMData[Grid2D[Linspace, Linspace]]
    | waveforms.ProjectedWaveform[reps.WDM[Grid2D[Linspace, Linspace]]]
)
IntegrationMethod = Literal["trapezoid", "simpson"]


def _first_frequencies(__x: FDEntry, /):
    return next(iter(__x.values())).frequencies


def _first_entries(__x: FDEntry, /):
    return next(iter(__x.values())).entries


class IntegrationPolicy(Protocol):
    """Protocol for quadrature policies used by noise models."""

    def integrate(
        self,
        __y: "Array",  # noqa: PYI063
        *,
        x: Union["Array", None] = None,
        **kwargs: Any,
    ) -> Any:
        """Integrate the given dm."""
        ...

    def cumulative(
        self,
        __y: "Array",  # noqa: PYI063
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
        self.method: IntegrationMethod = method

    def integrate(
        self,
        __y: "Array",  # noqa: PYI063
        *,
        x: Union["Array", None] = None,
        **kwargs: Any,
    ) -> Any:
        scipy_integrate = _import_scipy_integrate()
        return getattr(scipy_integrate, self.method)(__y, x=x, **kwargs)

    def cumulative(
        self,
        __y: "Array",  # noqa: PYI063
        *,
        x: Union["Array", None] = None,
        **kwargs: Any,
    ) -> Any:
        scipy_integrate = _import_scipy_integrate()
        return getattr(scipy_integrate, "cumulative_" + self.method)(__y, x=x, **kwargs)


class _JaxIntegrationPolicy(IntegrationPolicy):
    """Jax-backed integration policy with selectable quadrature method."""

    def __init__(
        self,
        method: IntegrationMethod = "trapezoid",
    ) -> None:
        self.method: IntegrationMethod = method

    def integrate(
        self,
        __y: "Array",  # noqa: PYI063
        *,
        x: Union["Array", None] = None,
        **kwargs: Any,
    ) -> Any:
        quadax = _import_quadax()
        return getattr(quadax, self.method)(__y, x=x, **kwargs)

    def cumulative(
        self,
        __y: "Array",  # noqa: PYI063
        *,
        x: Union["Array", None] = None,
        **kwargs: Any,
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
    msg = f"Unsupported array module {module_name}. Cannot create integration policy."
    raise NotImplementedError(msg)


class _StationaryFDNoise(Protocol):
    """Protocol for frequency domain stationary noise PSD models."""

    def psd(self, frequencies: "Array", option: ChnName) -> "Array":
        """Return the power spectral density (PSD) values on the given frequency grid for the specified channel."""  # noqa: E501
        ...


class SpectralDensity:
    """Represent the spectral density matrix (SDM) of a frequency domain stationary noise model.

    .. note::

        The SDM is a real-valued, symmetric, positive-definite matrix,
        hence it is invertible.
        It is the inverse SDM that is used in the inner product
        and whitening operations,
        and we store the inverse SDM in this representation.

    Note
    ----
    To construct a :class:`.SpectralDensity`,
    use the :func:`~typed_lisa_toolkit.make_sdm` factory function.
    """  # noqa: E501

    def __init__(
        self,
        frequencies: "Array",
        inverse_sdm: "Array",
        channel_order: Sequence[ChnName],
    ):
        # kernel shape: (n_freqs, n_channels, n_channels)
        self._frequencies: Array = frequencies
        self._inverse_sdm: Array = inverse_sdm
        self.channel_order: tuple[ChnName, ...] = tuple(channel_order)

    def to_subband(self, f_interval: tuple[float, float]) -> Self:
        """Return a new SpectralDensity instance with the frequency grid restricted to the given subband."""  # noqa: E501
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
            msg = f"Backend conversion is not implemented yet. Got backend={backend}."
            raise NotImplementedError(msg)
        return self._inverse_sdm

    def get_whitening_matrix(
        self,
        kind: Literal["cholesky"] | None = "cholesky",
    ) -> "Array":
        r"""Return whitening matrix :math:`W` with shape ``(n_freqs, n_channels, n_channels)``.

        The whitening matrix represents a linear transformation that
        yields unit variance white noise when applied to noise that
        follows this model. This is useful in detecting deviations from the model.

        .. note:: :math:`W` satisfies :math:`S_n^{-1} = W^\top W`.
        """  # noqa: E501
        if kind != "cholesky":
            msg = (
                f"Unsupported whitening matrix kind {kind}. "
                "Only 'cholesky' is supported currently."
            )
            raise NotImplementedError(msg)
        xp = self._inverse_sdm.__array_namespace__()
        return xp.linalg.cholesky(self._inverse_sdm, upper=True)


class DiagonalSpectralDensity(SpectralDensity):
    """Represent a SDM for a frequency domain stationary noise model with no inter-channel correlations.

    See Also
    --------
    :class:`.SpectralDensity`

    Note
    ----
    To construct a :class:`.DiagonalSpectralDensity`,
    use the :func:`~typed_lisa_toolkit.make_sdm` factory function.
    """  # noqa: E501

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
        """Create a SpectralDensity instance from a frequency domain noise model and a frequency grid."""  # noqa: E501
        _dict = {
            chnname: 1 / fd_noise.psd(frequencies, option=chnname)
            for chnname in channel_names
        }
        xp = next(iter(_dict.values())).__array_namespace__()
        diag = xp.stack(
            [xp.squeeze(_dict[c]) for c in channel_names],
            axis=-1,
        )  # (n_freqs, n_channels)
        kernel = diag[:, :, None] * xp.eye(
            len(channel_names),
            dtype=diag.dtype,
        )  # (n_freqs, n_channels, n_channels)
        return cls(frequencies, kernel, channel_names)

    def get_whitening_matrix(self, kind: Literal["cholesky"] | None = None) -> "Array":
        r"""Return whitening matrix :math:`W` with shape ``(n_freqs, n_channels, n_channels)``.

        The whitening matrix represents a linear transformation that
        yields unit variance white noise when applied to noise that
        follows this model. This is useful in detecting deviations from the model.

        .. note:: :math:`W` satisfies :math:`S_n^{-1} = W^\top W`.
        """  # noqa: E501
        if kind is not None:
            return super().get_whitening_matrix(kind=kind)
        xp = self._inverse_sdm.__array_namespace__()
        # S_n^{-1} is diagonal => W = sqrt(S_n^{-1})
        diag = xp.diagonal(self._inverse_sdm, axis1=-2, axis2=-1)
        return xp.sqrt(diag)[:, :, None] * xp.eye(
            len(self.channel_order),
            dtype=diag.dtype,
        )


class _EntryInDomain[DomainT: Domain](Protocol):
    @property
    def domain(self) -> DomainT:
        """Return the domain of the entry."""
        ...

    def get_kernel(self) -> "Array":
        """Return the kernel array of the entry."""
        ...


class NoiseModelLike[EntryT1: _EntryInDomain[Domain], EntryT2: _EntryInDomain[Domain]](
    Protocol,
):
    """Protocol for noise models."""

    def get_scalar_product(
        self,
        left: EntryT1 | EntryT2,
        right: EntryT1 | EntryT2,
    ) -> "Array":
        """Return the scalar product."""
        ...


class FDNoiseModel(
    NoiseModelLike[dm.FSData, waveforms.ProjectedWaveform[reps.FrequencySeries[Axis]]],
):
    """Frequency domain noise model.

    Assuming the noise is stationary, the noise model is given by the
    noise power spectral density (PSD) in the frequency domain. This
    class might not be suitable for non-stationary noise.

    Note
    ----
    To construct a :class:`.FDNoiseModel`,
    use the :func:`~typed_lisa_toolkit.noise_model` factory function.


    Attention
    ---------
    This class is considered experimental. If you are interested in using it,
    please reach out to the developers to discuss your use case and
    how we can best support it.
    """

    def __init__(
        self,
        sdm: SpectralDensity,
        integration_method: IntegrationMethod = "trapezoid",
    ):
        self._sdm_orig_: SpectralDensity = sdm
        # Keep the original PSD object for potential future use
        # (e.g., subband restriction)
        self.sdm: SpectralDensity = sdm
        xp = sdm.get_kernel().__array_namespace__()
        self._ip: IntegrationPolicy = _make_integration_policy(xp, integration_method)

    def reset(self) -> Self:
        """Reset the noise model to its original state, typically after subband restriction."""  # noqa: E501
        self.sdm = self._sdm_orig_
        return self

    def to_subband(self, f_interval: tuple[float, float]) -> Self:
        """Restrict the noise model to a subband."""
        self.sdm = self._sdm_orig_.to_subband(f_interval)
        return self

    def _get_whitened_entries(self, _data: FDEntry) -> "Array":
        """Return the whitened kernel entries of the given dm."""
        kernel = _data.get_kernel()  # (n_batches, n_ch, 1, 1, n_freqs)
        xp = kernel.__array_namespace__()
        W = self.sdm.get_whitening_matrix()  # (n_freqs, n_ch, n_ch)  # noqa: N806
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
            if self.sdm.is_diagonal:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                # If the spectral density matrix is diagonal,
                # we can simply divide by the diagonal elements.
                diag = xp.diagonal(
                    self.sdm.get_kernel(),
                    axis1=-1,
                    axis2=-2,
                )  # shape (n_freqs, n_channels)
                return (4 * _left.conj() * _right) * diag.T[None, :, None, None, :]
        except AttributeError:
            pass
        return 4 * xp.einsum(
            "...fi,fij,...fj->...f",
            xp.moveaxis(_left.conj(), 1, -1),
            self.sdm.get_kernel(),
            xp.moveaxis(_right, 1, -1),
        )

    def get_complex_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the complex scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`,
        and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \langle d, h \rangle = 4 \int_{f_\text{min}}^{f_\text{max}}
            \frac{d^*(f) h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        frequencies = _first_frequencies(left)
        xp = _first_entries(left).__array_namespace__()
        return self._ip.integrate(
            self.get_integrand(left, right),
            x=_mixins.to_array(frequencies, xp=xp),
        )

    def get_cumulative_complex_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the cumulative complex scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`,
        and the noise PSD is :math:`S_n(f)`,
        this method returns an array of the following function
        on the input frequency grid
        :math:`[f_\text{min}, f_\text{max}]`:

        .. math::

            F \mapsto 4\int_{f_\text{min}}^{F} \frac{d^*(f)
            h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        frequencies = _first_frequencies(left)
        xp = _first_entries(left).__array_namespace__()
        return self._ip.cumulative(
            self.get_integrand(left, right),
            x=_mixins.to_array(frequencies, xp=xp),
            initial=0,
        )

    def get_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`,
        and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \left( d \middle| h \right) = 4 \Re \int_{f_\text{min}}^{f_\text{max}}
            \frac{d^*(f) h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        return self.get_complex_scalar_product(left, right).real

    inner: Callable[..., "Array"] = get_scalar_product
    """Alias for :meth:`get_scalar_product`."""

    def get_cumulative_scalar_product(
        self,
        left: FDEntry,
        right: FDEntry,
    ) -> "Array":
        r"""Return the cumulative scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`,
        and the noise PSD is :math:`S_n(f)`,
        this method returns an array of the following function
        on the input frequency grid
        :math:`[f_\text{min}, f_\text{max}]`:

        .. math::

            F \mapsto 4\Re \int_{f_\text{min}}^{F} \frac{d^*(f)
            h(f)}{S_n(f)} \, \mathrm{d} f.
        """
        return self.get_cumulative_complex_scalar_product(left, right).real

    def get_cross_correlation(
        self,
        left: dm.TimedFSData,
        right: FDEntry,
    ):
        r"""Return the cross correlation.

        Assuming `left` is :math:`d`, `right` is :math:`h`,
        and the noise PSD is :math:`S_n(f)`,
        we define the cross-correlation as

        .. math::

            (d \ast h)(\tau) := \left( \hat{d}(t) \middle| \hat{h}(t + \tau) \right),

        where the hat denotes the real Fourier transform.
        This methods returns a generalization of the above
        cross-correlation which is **complex-valued**

        .. math::

            (d \star h)(\tau) := \langle \hat{d}(t), \hat{h}(t + \tau) \rangle.

        In the implementation, the generalized cross-correlation is computed using

        .. math::

            (d \star h)(\tau) \propto
            \mathcal{F}^{-1}\left(4\frac{d^*(f) h(f)}{S_n(f)}\right),

        where :math:`\mathcal{F}^{-1}` is the two-sided inverse Fourier transform. Note
        that the input arrays :math:`d(f)` and :math:`h(f)` are one-sided,
        and the negative frequencies are
        populated by **zero** before the inverse Fourier transform.
        """
        xp = _first_entries(left).__array_namespace__()
        two_sided_freq = xp.fft.fftshift(
            xp.fft.fftfreq(len(left.times), left.times.step),
        )
        _first = next(iter(left.values()))
        frequencies, df = _mixins.to_array(_first.frequencies, xp), _first.df
        two_sided_integrand_entries = utils.extend_to(two_sided_freq)(
            frequencies,
            self.get_integrand(left, right),
        )
        cross_correlation = xp.fft.ifft(
            xp.fft.ifftshift(two_sided_integrand_entries) * df,
            len(left.times),
            norm="forward",
            axis=-1,
        )
        return dm.tsdata(
            times=left.times,
            entries=cross_correlation,
            channels=left.channel_names,
        )

    def whiten(self, _data: FDEntry):
        r"""Return whitened data with the same container type as input.

        Applies the whitening matrix, so whitened noise has unit covariance.
        """
        d_k = _data.get_kernel()  # (n_batches, n_ch, 1, 1, n_freqs)
        xp = d_k.__array_namespace__()
        W = self.sdm.get_whitening_matrix()  # (n_freqs, n_ch, n_ch)  # noqa: N806
        d_e = xp.moveaxis(d_k[:, :, 0, 0, :], 1, -1)  # (n_batches, n_freqs, n_ch)
        whitened_e = xp.einsum("fij,...fj->...fi", W, d_e)  # (n_batches, n_freqs, n_ch)
        whitened_k = xp.moveaxis(whitened_e, -1, 1)[:, :, None, None, :]
        return _data.create_like(whitened_k)

    def get_overlap(self, left: FDEntry, right: FDEntry) -> "Array":
        r"""Return the overlap.

        Assuming `left` is :math:`d`, `right` is :math:`h`,
        and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \frac{\langle d, h \rangle}
            {\sqrt{\langle d, d \rangle \langle h, h \rangle}}.
        """
        xp = _first_entries(left).__array_namespace__()
        return self.get_scalar_product(left, right) / xp.sqrt(
            self.get_scalar_product(left, left) * self.get_scalar_product(right, right),
        )


class EvolutionarySpectralDensity:
    """Evolutionary spectral density matrix (ESDM) for a time-frequency noise model.

    Stores the inverse ESDM as an array of shape
    ``(n_freqs, n_times, n_channels, n_channels)``.
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
        self._frequencies: Array = frequencies
        self._times: Array = times
        self._inverse_esdm: Array = inverse_esdm
        self.channel_order: tuple[ChnName, ...] = tuple(channel_order)

    def get_kernel(self, backend: str | None = None) -> "Array":
        """Return the inverse of the evolutionary spectral density matrix.

        The returned array has shape ``(n_freqs, n_times, n_channels, n_channels)``.
        """
        if backend is not None:
            msg = f"Backend conversion is not implemented yet. Got backend={backend}."
            raise NotImplementedError(msg)
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
                msg = (
                    f"Invalid channel_order {channel_order}. "
                    "Channel names must be unique."
                )
                raise ValueError(msg)
            return False
        # Check if the shape is correct.
        shape_size = 4
        if len(_d.shape) != shape_size or _d.shape[-2:] != (
            len(channel_order),
            len(channel_order),
        ):
            if raise_exception:
                msg = (
                    "Expected (inverse) evolutionary spectral density matrix."
                    "To have shape (n_freq, n_time, n_channels, n_channels), "
                    f"but got shape {_d.shape} instead."
                )
                raise ValueError(msg)
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
        """  # noqa: E501
        if kind != "cholesky":
            msg = (
                f"Unsupported whitening matrix kind {kind}. "
                "Only 'cholesky' is supported currently."
            )
            raise NotImplementedError(msg)  # pyright: ignore[reportUnreachable]
        xp = self._inverse_esdm.__array_namespace__()
        return xp.linalg.cholesky(self._inverse_esdm, upper=True)


class TFNoiseModel:
    """Time-frequency Gaussian noise model.

    This model is a Gaussian noise model suitable for non-stationary noise.
    The covariance matrix is diagonal in the chosen time-frequency representation.
    The model allows correlations between TDI channels. In other words, we have a 3x3
    symmetric matrix at each location in the time-frequency plane.

    Note
    ----
    To construct a :class:`.TFNoiseModel`, use the
    :func:`~typed_lisa_toolkit.noise_model` factory function.

    Attention
    ---------
    This class is considered experimental. If you are interested in using it,
    please reach out to the developers to discuss your use case and
    how we can best support it.
    """

    def __init__(
        self,
        esd: EvolutionarySpectralDensity,
    ):
        self.esd: EvolutionarySpectralDensity = esd

    def _get_whitened_entries(self, _data: TFEntry) -> "Array":
        """Return the whitened kernel entries of the given dm."""
        kernel = _data.get_kernel()  # (n_batches, n_ch, 1, 1, n_freqs, n_times)
        xp = kernel.__array_namespace__()
        W = (  # noqa: N806
            self.esd.get_whitening_matrix()
        )  # (n_freqs, n_times, n_ch, n_ch)
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

    inner: Callable[..., "Array"] = get_scalar_product
    """Alias for :meth:`get_scalar_product`."""

    def whiten(self, _data: TFEntry) -> TFEntry:
        """Whiten the data according to the noise model."""
        whitened_array = self._get_whitened_entries(_data)
        return _data.create_like(whitened_array)


def _validate_shape(entries: "Array", expected_shape: tuple[int, ...]) -> None:
    if entries.shape != expected_shape:
        msg = (
            "Invalid shape for `inverse_sdm`. "
            f"Expected {expected_shape}, got {entries.shape}."
        )
        raise ValueError(msg)


@overload
def make_sdm(
    inverse_sdm: "Array",
    /,
    *,
    frequencies: "Array",
    channel_names: Sequence[ChnName],
    times: None = None,
    is_diagonal: Literal[False] = False,
) -> SpectralDensity: ...
@overload
def make_sdm(
    inverse_sdm: "Array",
    /,
    *,
    frequencies: "Array",
    channel_names: Sequence[ChnName],
    is_diagonal: Literal[True],
    times: None = None,
) -> DiagonalSpectralDensity: ...
@overload
def make_sdm(
    inverse_sdm: "Array",
    /,
    *,
    frequencies: "Array",
    times: "Array",
    channel_names: Sequence[ChnName],
) -> EvolutionarySpectralDensity: ...
def make_sdm(
    inverse_sdm: Array,
    /,
    *,
    frequencies: Array,
    channel_names: Sequence[ChnName],
    times: Array | None = None,
    is_diagonal: bool = False,
):
    """Make a :class:`~types.SpectralDensity`, a :class:`~types.DiagonalSpectralDensity` or an :class:`~types.EvolutionarySpectralDensity`.

    Parameters
    ----------
    inverse_sdm: :class:`~types.misc.Array`
        The inverse spectral density matrix (SDM) or inverse evolutionary spectral density matrix (ESDM).
        If `is_diagonal` is False, it must have shape (n_freqs, n_channels, n_channels) for SDM or
        (n_freqs, n_times, n_channels, n_channels) for ESDM.
        If `is_diagonal` is True, it must have shape (n_freqs, n_channels) and represent the diagonal elements of the inverse SDM
        (currently only supported for SDM, not ESDM).

    frequencies: :class:`~types.misc.Array`
        An array of shape (n_freqs,) representing the frequency grid.

    channel_names: Sequence[str]
        A sequence of channel names corresponding to the channels in the SDM/ESDM.

    times: :class:`~types.misc.Array`, optional
        An array of shape (n_times,) representing the time grid. If None, a :class:`~types.SpectralDensity`
        will be constructed. If provided, an :class:`~types.EvolutionarySpectralDensity` will be constructed.

    is_diagonal: bool
        Whether the SDM is diagonal. Only relevant if `times` is None. If True, a :class:`~types.DiagonalSpectralDensity`
        will be constructed. Defaults to False.
    """  # noqa: E501
    if times is None:
        if not is_diagonal:
            _validate_shape(
                inverse_sdm,
                expected_shape=(
                    len(frequencies),
                    len(channel_names),
                    len(channel_names),
                ),
            )
            return SpectralDensity(frequencies, inverse_sdm, channel_names)
        xp = xpc.get_namespace(inverse_sdm)
        _inverse_sdm = inverse_sdm[:, :, None] * xp.eye(
            len(channel_names),
            dtype=inverse_sdm.dtype,
        )
        return DiagonalSpectralDensity(frequencies, _inverse_sdm, channel_names)
    _validate_shape(
        inverse_sdm,
        expected_shape=(
            len(frequencies),
            len(times),
            len(channel_names),
            len(channel_names),
        ),
    )
    return EvolutionarySpectralDensity(
        frequencies,
        times,
        inverse_sdm,
        channel_names,
    )


@overload
def noise_model(
    sdm: SpectralDensity,
    integration_method: IntegrationMethod = "trapezoid",
) -> FDNoiseModel: ...
@overload
def noise_model(
    sdm: EvolutionarySpectralDensity,
) -> TFNoiseModel: ...


def noise_model(
    sdm: SpectralDensity | EvolutionarySpectralDensity,
    integration_method: IntegrationMethod = "trapezoid",
):
    """Construct a :class:`~types.FDNoiseModel` or :class:`~types.TFNoiseModel` from a :class:`~types.SpectralDensity` or a :class:`~types.EvolutionarySpectralDensity`.

    Parameters
    ----------
    sdm: :class:`~types.SpectralDensity` or :class:`~types.EvolutionarySpectralDensity`
        The (evolutionary) spectral density matrix defining the noise model.

    integration_method: :class:`~types.IntegrationMethod`
        The quadrature method to use for integration in the frequency domain.
        Only relevant if `sdm` is a :class:`~types.SpectralDensity`.
        Defaults to "trapezoid".
    """  # noqa: E501
    if isinstance(sdm, SpectralDensity):
        return FDNoiseModel(sdm=sdm, integration_method=integration_method)
    return TFNoiseModel(esd=sdm)
