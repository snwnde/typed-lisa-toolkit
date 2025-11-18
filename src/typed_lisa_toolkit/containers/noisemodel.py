"""Module for the noise model.

A noise model carries the knowledge of noise properties in the data.
It determines the geometry of the data space, of which the template
space is a subspace.

Currently available models are :class:`.FDNoiseModel` and
:class:`.WDMWhittleNoise`.

.. currentmodule:: typed_lisa_toolkit.containers.noisemodel

Types
-----

.. autoclass:: FSDataT
.. autoclass:: NPNumT
.. autoprotocol:: StationaryFDNoise
.. autoprotocol:: Integrator
.. autoprotocol:: CumIntegrator
.. autoclass:: IntegratorConfig

Entities
--------

.. autoclass:: FDNoiseModel
   :members:
   :member-order: groupwise
   :inherited-members:

.. autoclass:: WDMWhittleNoise
   :members:
   :member-order: groupwise
   :undoc-members:
"""

import abc
import logging
from typing import (
    Protocol,
    TypeVar,
    TypedDict,
    Unpack,
    Self,
    NotRequired,
    Literal,
    KeysView,
    overload,
)

import numpy as np
import numpy.typing as npt
import scipy.integrate  # type: ignore[import]

from . import arithdicts
from . import representations as reps
from . import data
from .. import utils

log = logging.getLogger(__name__)

ChnName = str
FSDataT = TypeVar("FSDataT", bound=data.FSData)
NPNumT = TypeVar("NPNumT", bound=np.number)


class Integrator(Protocol):
    """Protocol for integrators."""

    def __call__(
        self,
        __y: npt.NDArray[NPNumT],
        **kwargs,
    ) -> NPNumT:  # pyright: ignore[reportReturnType]
        """Integrate the given data."""


class CumIntegrator(Protocol):
    """Protocol for cumulative integrators."""

    def __call__(
        self,
        __y: npt.NDArray[NPNumT],
        **kwargs,
    ) -> npt.NDArray[NPNumT]:  # pyright: ignore[reportReturnType]
        """Integrate the given data cumulatively."""


class IntegratorConfig(TypedDict):
    """Dictionary for integrator and cumulative integrator configuration."""

    integrator: NotRequired[Integrator]
    cumulative_integrator: NotRequired[CumIntegrator]


def _collect_frequencies(data: data.FSData):
    return arithdicts.ChannelDict(
        {chnname: series.frequencies for chnname, series in data.items()}
    )


class StationaryFDNoise(Protocol):
    """Protocol for frequency domain stationary noise PSD models."""

    def psd(
        self, frequencies: npt.NDArray[np.floating], option: ChnName
    ) -> npt.NDArray[np.floating]:  # pyright: ignore[reportReturnType]
        """Return the power spectral density (PSD)."""


class FDNoiseModel:
    """Frequency domain noise model.

    Assuming the noise is stationary, the noise model is given by the
    noise power spectral density (PSD) in the frequency domain. This
    class might not be suitable for non-stationary noise.

    """

    _DEFAULT_INTEGRATOR = scipy.integrate.trapezoid
    _DEFAULT_CUM_INTEGRATOR = scipy.integrate.cumulative_trapezoid

    def __init__(
        self,
        integrator: Integrator = _DEFAULT_INTEGRATOR,  # pyright: ignore[reportArgumentType]
        cumulative_integrator: CumIntegrator = _DEFAULT_CUM_INTEGRATOR,  # pyright: ignore[reportArgumentType]
    ) -> None:
        self.integrator = integrator
        self.cumulative_integrator = cumulative_integrator

    @abc.abstractmethod
    def create_new(
        self,
        integrator: Integrator = _DEFAULT_INTEGRATOR,  # pyright: ignore[reportArgumentType]
        cumulative_integrator: CumIntegrator = _DEFAULT_CUM_INTEGRATOR,  # pyright: ignore[reportArgumentType]
    ) -> Self:
        """Return a new instance of the class with the given integrators."""

    @classmethod
    def make(
        cls,
        fd_noise: StationaryFDNoise | None = None,
        noise_cache: data.FSData | None = None,
        **kwargs: Unpack[IntegratorConfig],
    ):
        """Return an :class:`.FDNoiseModel` instance.

        To create an instance, either provide a noise PSD model or a noise
        PSD cache. If both are provided, an error is raised.

        The user may also provide a custom integrator :class:`.Integrator`
        and/or a custom cumulative integrator :class:`.CumIntegrator` to be
        used in the scalar product and the cumulative scalar product
        computations. The default integrator is :func:`scipy.integrate.trapezoid`
        and the default cumulative integrator is :func:`scipy.integrate.cumulative_trapezoid`.
        """
        args = (fd_noise, noise_cache)
        error = ValueError(
            "Exactly one of fd_noise and noise_cache should be provided."
        )
        if sum(arg is not None for arg in args) != 1:
            raise error
        if fd_noise is not None:
            return _StationaryNoiseModel(fd_noise, **kwargs)
        if noise_cache is not None:
            return _CacheNoiseModel(noise_cache, **kwargs)
        raise error

    @abc.abstractmethod
    def get_noise_psd(
        self, frequencies: arithdicts.ChannelDict[reps.Linspace | npt.NDArray]
    ) -> data.FSData:
        """Return the noise PSD.

        This method returns the noise PSD generated by the noise PSD model
        or stored in the cache, on the given dictionary of frequencies.
        """

    def get_integrand(
        self,
        left: data.FSData,
        right: data.FSData,
    ) -> data.FSData:
        r"""Return the integrand of the scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            4\frac{d(f) h^*(f)}{S_n(f)}.
        """
        noise_psd = self.get_noise_psd(_collect_frequencies(left))
        return 4 * left * (1 / noise_psd) * right.conj()

    def get_complex_scalar_product(
        self,
        left: data.FSData,
        right: data.FSData,
    ):
        r"""Return the complex scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \langle d, h \rangle = 4 \int_{f_\text{min}}^{f_\text{max}} \frac{d(f) h^*(f)}{S_n(f)} \, \mathrm{d} f.
        """
        integrand = self.get_integrand(left, right)
        _dict = {
            chnname: self.integrator(series.entries, x=series.frequencies)
            for chnname, series in integrand.items()
        }
        return arithdicts.ChannelDict(_dict)

    def get_cumulative_complex_scalar_product(
        self,
        left: data.FSData,
        right: data.FSData,
    ):
        r"""Return the cumulative complex scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns an array of the following function on the input frequency grid
        :math:`[f_\text{min}, f_\text{max}]`:

        .. math::

            F \mapsto 4\int_{f_\text{min}}^{F} \frac{d(f) h^*(f)}{S_n(f)} \, \mathrm{d} f.
        """
        integrand = self.get_integrand(left, right)

        def make_product(chname: ChnName):
            series = integrand[chname]

            return self.cumulative_integrator(
                series.entries, x=series.frequencies, initial=0
            )

        _dict = {chnname: make_product(chnname) for chnname in integrand.keys()}
        return arithdicts.ChannelDict(_dict)

    def get_scalar_product(
        self,
        left: data.FSData,
        right: data.FSData,
    ):
        r"""Return the scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \left( d \middle| h \right) = 4 \Re \int_{f_\text{min}}^{f_\text{max}} \frac{d(f) h^*(f)}{S_n(f)} \, \mathrm{d} f.
        """
        return self.get_complex_scalar_product(left, right).real

    def get_cumulative_scalar_product(
        self,
        left: data.FSData,
        right: data.FSData,
    ):
        r"""Return the cumulative scalar product.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns an array of the following function on the input frequency grid
        :math:`[f_\text{min}, f_\text{max}]`:

        .. math::

            F \mapsto 4\Re \int_{f_\text{min}}^{F} \frac{d(f) h^*(f)}{S_n(f)} \, \mathrm{d} f.
        """
        return self.get_cumulative_complex_scalar_product(left, right).pass_through(
            np.real
        )

    def get_cross_correlation(
        self,
        left: data.TimedFSData,
        right: data.FSData,
    ) -> arithdicts.ChannelDict[reps.TimeSeries]:
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

            (d \star h)(\tau) \propto \mathcal{F}^{-1}\left(4\frac{d(f) h^*(f)}{S_n(f)}\right),

        where :math:`\mathcal{F}^{-1}` is the two-sided inverse Fourier transform. Note
        that the input arrays :math:`d(f)` and :math:`h(f)` are one-sided, and the negative frequencies are
        populated by **zero** before the inverse Fourier transform.
        """
        left_times = reps.Linspace.make(left.times)
        two_sided_freq = np.fft.fftshift(
            np.fft.fftfreq(len(left_times), left_times.step)
        )
        integrand = self.get_integrand(left, right)
        integ_freqs = np.array(integrand.frequencies)
        inband_slice = utils.get_subset_slice(
            two_sided_freq, integ_freqs[0], integ_freqs[-1]
        )
        # Note that we assumed all the channels have the same frequency grid.

        def make_cross_correlation(chnname: ChnName):
            series = integrand[chnname]
            integrand_two_sided_freq = np.zeros_like(
                two_sided_freq, dtype=series.entries.dtype
            )
            integrand_two_sided_freq[inband_slice] = series.entries
            # Pay attention to the sign convention (ifft or fft) and the normalization.
            cross_correlation = np.fft.ifft(
                np.fft.ifftshift(integrand_two_sided_freq) * integrand.df,
                len(left.times),
                norm="forward",
            )
            return reps.TimeSeries(left.times, cross_correlation)

        _dict = {
            chnname: make_cross_correlation(chnname) for chnname in integrand.keys()
        }
        return arithdicts.ChannelDict(_dict)

    def get_whitened(self, data: FSDataT) -> FSDataT:
        r"""Return the whitened data.

        Assuming the noise PSD is :math:`S_n(f)`, this method returns

        .. math::

            \frac{d(f)}{\sqrt{S_n(f)}}.
        """
        noise_psd = self.get_noise_psd(_collect_frequencies(data))
        return data / noise_psd.sqrt()

    def get_overlap(self, left: data.FSData, right: data.FSData):
        r"""Return the overlap.

        Assuming `left` is :math:`d`, `right` is :math:`h`, and the noise PSD is :math:`S_n(f)`,
        this method returns

        .. math::

            \frac{\langle d, h \rangle}{\sqrt{\langle d, d \rangle \langle h, h \rangle}}.
        """
        left_norm_square = self.get_scalar_product(left, left)
        right_norm_square = self.get_scalar_product(right, right)
        overlap = self.get_scalar_product(left, right) / (
            left_norm_square * right_norm_square
        ).pass_through(np.sqrt)
        return overlap

    def get_cache(self, data: data.FSData):
        """Return the cache.

        This method returns an instance of :class:`.FDNoiseModel` that
        caches the noise PSD on the grid of the given data. The cache
        can then be used in the computation on the same grid or a
        subset of it, without calling the noise PSD model again.
        """
        noise_psd = self.get_noise_psd(_collect_frequencies(data))
        return _CacheNoiseModel(
            noise_cache=noise_psd,
            integrator=self.integrator,
            cumulative_integrator=self.cumulative_integrator,
        )


class _StationaryNoiseModel(FDNoiseModel):
    def __init__(
        self, fd_noise: StationaryFDNoise, **kwargs: Unpack[IntegratorConfig]
    ) -> None:
        super().__init__(**kwargs)
        self.fd_noise = fd_noise

    def create_new(
        self,
        integrator: Integrator = FDNoiseModel._DEFAULT_INTEGRATOR,  # pyright: ignore[reportArgumentType]
        cumulative_integrator: CumIntegrator = FDNoiseModel._DEFAULT_CUM_INTEGRATOR,  # pyright: ignore[reportArgumentType]
    ):
        return type(self)(
            self.fd_noise,
            integrator=integrator,
            cumulative_integrator=cumulative_integrator,
        )

    def get_noise_psd(
        self, frequencies: arithdicts.ChannelDict[reps.Linspace | npt.NDArray]
    ):
        def make_psd(chnname: ChnName):
            freq = np.array(frequencies[chnname])
            entries = self.fd_noise.psd(freq, option=chnname)
            return reps.FrequencySeries(freq, entries)

        _dict = {chnname: make_psd(chnname) for chnname in frequencies.keys()}
        return data.FSData(_dict)


class _CacheNoiseModel(FDNoiseModel):
    def __init__(
        self, noise_cache: data.FSData, **kwargs: Unpack[IntegratorConfig]
    ) -> None:
        super().__init__(**kwargs)
        self.noise_cache = noise_cache

    def create_new(
        self,
        integrator: Integrator = FDNoiseModel._DEFAULT_INTEGRATOR,  # pyright: ignore[reportArgumentType]
        cumulative_integrator: CumIntegrator = FDNoiseModel._DEFAULT_CUM_INTEGRATOR,  # pyright: ignore[reportArgumentType]
    ):
        return type(self)(
            self.noise_cache,
            integrator=integrator,
            cumulative_integrator=cumulative_integrator,
        )

    def get_noise_psd(
        self, frequencies: arithdicts.ChannelDict[reps.Linspace | npt.NDArray]
    ):
        freq = np.array(next(iter(frequencies.values())))
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


################################################################
# In the following dynamic Whittle noise model, there is no
# integrator: all inner products are computed from finite vectors.
# This is because we do not use any of the continuum limits. In
# the case of WDM, these correspond (loosely) to the time domain
# and the frequency domain, so there is no point in doing so: we
# would lose the ability to express non-stationarities with a
# diagonal covariance matrix.
################################################################


# TODO abstract this into a more general TF Whittle Noise?
# I tried making it generic on data.WDMData but that is difficult since
# I need to instantiate a bunch of WDMData here and a TypeVar is not
# a class, i.e. cannot be instantiated. Generic classes in python
# only look like C++ templates, they cannot actually do what templates do.
# Maybe the way to avoid code duplication would be to make a module
# for the static methods in here.
class WDMWhittleNoise:
    """WDM dynamic Whittle noise model.

    This model uses :class:`.representations.WDM` wavelets to represent
    TDI data. It is a Whittle model in the sense that the likelihood
    is Gaussian, and its covariance matrix is diagonal in the chosen WDM
    representation. However, the model allows correlations between TDI
    channels. In other words, we have a 3x3 symmetric matrix at each
    location in the time-frequency plane.

    :param invevsdm: The inverse evolutionary spectral density matrix.
        The keys must be strings of length 2, where each character stands
        for a TDI channel (for example, "XX", "XY"). A maximum of 3 TDI
        channels are allowed. The off-diagonal elements are equal by
        symmetry and must be specified exactly once (i.e., exaclty one of
        the keys "XY", "YX" can exist). If ``channel_order`` is not given,
        the off-diagonal keys are used to derive an ordering of the channels,
        making them all above-diagonal elements. In that case, a circular
        permutation is not allowed (e.g. ``["XY", "YZ", "ZX"]`` is not allowed).
    :type invevsdm: data.WDMData

    :param channel_order: string of length 1-3 specifying an ordering to the TDI channels.
        If this is given, the keys of ``invevsdm`` are changed to be upper-triangular.
        The passed ``invevsdm`` is not changed in-place. Defaults to None
    :type channel_order: str | None, optional

    :param invert_sdm: if True, invert the passed spectral density matrix before storing it.
        This is a convenience flag to be used when you have the evolutionary spectral
        density rather than its inverse. For performance, it could useful to perform the inversion
        yourself before passing it here (see :meth:`invert_sdm`). Defaults to False
    :type invert_sdm: bool, optional

    :raises ValueError: if the data keys are not of length 2, if any are
        missing or redundant, or if ``channel_order`` is not given and
        no ordering can be derived from the TDI channels.
    """

    def __init__(
        self,
        invevsdm: data.WDMData,
        *,
        channel_order: str | None = None,
        invert_sdm: bool = False,
    ):
        _ = self.is_valid_sdm(
            invevsdm,
            raise_exception=True,
            channel_order=channel_order,
        )

        self.single_channels: list[str]
        """Single channel names. These are single characters."""

        if channel_order:
            self.single_channels = list(channel_order)
            invevsdm = self._order_evsdm_channels(invevsdm, channel_order)
        else:
            self.single_channels = self._get_single_channels(invevsdm)

        self._set_single_channels = set(self.single_channels)
        self._set_channel_pairs = set(list(invevsdm.keys()))

        self.invevsdm = invevsdm
        """
        The inverse evolutionary spectral density matrix.

        The keys of the WDMData are length-2 strings representing a
        pair of TDI channels. For the off-diagonal terms, only one of
        them is stored, but either one can be dynamically obtained
        through :meth:`.invevsdm_at_channels`.
        """

        if invert_sdm:
            self.invevsdm = self.invert_sdm(invevsdm)

        # TODO allow deep-copy of invevsdm

    @staticmethod
    def is_valid_sdm(
        _evsdm_or_invevsdm: data.WDMData,
        /,
        *,
        raise_exception=False,
        channel_order: str | None = None,
    ) -> bool:
        """Check validity of dictionary keys in an (inverse) evolutionary spectral density matrix.

        The keys must be strings of length 2, where each character stands
        for a TDI channel (for example, "XX", "XY"). A maximum of 3 TDI
        channels are allowed. The off-diagonal elements are equal by
        symmetry and must be specified exactly once (i.e., exaclty one of
        the keys "XY", "YX" can exist). If ``channel_order`` is not given,
        the off-diagonal keys are used to derive an ordering of the channels,
        making them all above-diagonal elements. In that case, a circular
        permutation is not allowed (e.g. ``["XY", "YZ", "ZX"]`` is not allowed).

        :param _evsdm_or_invevsdm: the data to check.
        :type _evsdm_or_invevsdm: data.WDMData

        :param raise_exception: whether to raise exceptions on invalid input, defaults to False
        :type raise_exception: bool, optional

        :param channel_order_given: whether a channel ordering is explicitly given, which changes
            how off-diagonal elements are to be treated as in :meth:`__init__`. Defaults to False
        :type channel_order_given: bool, optional

        :return: True if the input is valid, False if ``raise_exception`` is False and the input
            is invalid.
        :rtype: bool

        :raises ValueError: if the input is invalid and ``raise_exception`` is true.
        """
        _d = _evsdm_or_invevsdm
        keys = list(_d.keys())

        # check that keys are length-2
        valid_keys = [len(key) == 2 for key in keys]
        if not all(valid_keys):
            invalid = [k for k, valid in zip(keys, valid_keys) if not valid]
            if raise_exception:
                raise ValueError(f"The following keys are not length-2: {invalid}")
            else:
                return False

        # find unique channel names
        # NOTE cannot use self._get_single_channels() as that assumes a valid sdm
        single_channels = []
        for c in "".join(list(_evsdm_or_invevsdm.keys())):
            if c not in single_channels:
                single_channels.append(c)

        # disallow empty model
        if len(single_channels) == 0:
            if raise_exception:
                raise ValueError("received empty evsdm or invevsdm")
            else:
                return False

        # disallow more than 3 TDI channels
        if len(single_channels) > 3:
            if raise_exception:
                raise ValueError(
                    f"received evsdm or invevsdm with more than 3 TDI channels, {single_channels=}"
                )
            else:
                return False

        # check that number of channels makes sense
        n_channels = len(single_channels)
        n_offdiag = n_channels * (n_channels - 1) / 2
        n_pairs = n_channels + n_offdiag
        if len(keys) != n_pairs:
            if raise_exception:
                raise ValueError(
                    f"Received {n_channels} single channels {single_channels}, "
                    f"expected {n_pairs} spectral densities in evsdm, got only "
                    f"{len(keys)} densities {keys} instead."
                )
            else:
                return False

        # check that the implicit ordering is possible
        if not channel_order and n_channels == 3:
            offdiag_keys = [k for k in keys if k[0] != k[1]]
            k0, k1, k2 = offdiag_keys
            is_circular = len({k0[0], k1[0], k2[0]}) == 3
            if is_circular:
                if raise_exception:
                    raise ValueError(
                        f"cannot derive an ordering of TDI channels from off-diagonal "
                        f"keys {(k0, k1, k2)}. Either give an explicit channel order "
                        f"or invert one of the keys."
                    )
                else:
                    return False

        if channel_order:
            if len(channel_order) != len(single_channels):
                if raise_exception:
                    raise ValueError(
                        f"different number of channels in {channel_order=} and input {single_channels}"
                    )
                else:
                    return False

        return True

    @staticmethod
    def order_evsdm_channels(
        _evsdm_or_invevsdm: data.WDMData, /, channel_order: str
    ) -> data.WDMData:
        """Order (inverse) evolutionary spectral density matrix keys as upper triangular.

        :param _evsdm_or_invevsdm: the data whose keys are to be ordered. The argument is not
            altered in-place. Only the keys, not the values, are to be changed.
        :type _evsdm_or_invevsdm: data.WDMData

        :param channel_order: a string containing the TDI channels in order.
        :type channel_order: str

        :return: the ordered spectral density matrix.
        :rtype: data.WDMData

        :raises ValueError: if the arguments are invalid or incompatible as specified in
            :meth:`.__init__`.
        """
        _ = WDMWhittleNoise.is_valid_sdm(
            _evsdm_or_invevsdm, raise_exception=True, channel_order=channel_order
        )
        return WDMWhittleNoise._order_evsdm_channels(_evsdm_or_invevsdm, channel_order)

    @staticmethod
    def _order_evsdm_channels(_d: data.WDMData, /, channel_order: str) -> data.WDMData:
        """Order (inv)evsdm as upper triangular without checking input validity."""
        new_keys = {
            k: WDMWhittleNoise._reorder_channel_pair(k, channel_order)
            for k in _d.keys()
        }
        return data.WDMData({newk: _d[oldk] for oldk, newk in new_keys.items()})

    @staticmethod
    def _reorder_channel_pair(pair: str, channel_order: str) -> str:
        """Reorder a single channel pair, assuming validity of inputs."""
        assert len(pair) == 2
        assert set(pair).issubset(set(channel_order))
        i0, i1 = channel_order.find(pair[0]), channel_order.find(pair[1])
        if i0 < i1:
            return pair
        else:
            return pair[::-1]

    @staticmethod
    def _get_single_channels(_evsdm_or_invevsdm: data.WDMData) -> list[str]:
        """Get single_channels list from sdm assuming that it's valid."""
        single_channels: list[str] = []

        for c in "".join(list(_evsdm_or_invevsdm.keys())):
            if c not in single_channels:
                single_channels.append(c)
        if len(single_channels) <= 2:
            return single_channels

        assert len(single_channels) == 3
        # implicit order given by off-diagonal keys
        offdiag_keys = [k for k in _evsdm_or_invevsdm.keys() if k[0] != k[1]]
        k0, k1, k2 = offdiag_keys
        # start inserting two channels from k0. Then find where to place the
        # remaining channel from k1 if possible, from k2 if not
        single_channels = list(k0)
        if k1[0] == k0[1]:
            single_channels.insert(2, k1[1])
            return single_channels
        if k1[1] == k0[0]:
            single_channels.insert(0, k1[0])
            return single_channels
        if k2[0] == k0[1]:
            single_channels.insert(2, k2[1])
            return single_channels
        if k2[1] == k0[0]:
            single_channels.insert(0, k2[0])
            return single_channels

        assert False, "unreachable for valid input"

    @staticmethod
    def in_given_order(channels: str, channel_pairs: KeysView | set | list) -> str:
        """Return two-channel string in the order stored in ``channel_pairs``.

        :param channels: length-2 string describing two TDI channels
        :type channels: str

        :param channel_pairs: the channel pairs in your (inverse) evolutionary spectral density matrix.
        :type channel_pairs: KeysView | set | list

        :return: a pair in ``channel_pairs``.
        :rtype: str

        :raises ValueError: if ``channels`` is not of length 2 or the corresponding pair is ``channel_pairs``.
        """
        if len(channels) != 2:
            raise ValueError(f"channels '{channels}' should have been length-2 string")
        if channels in channel_pairs:
            return channels
        elif channels[::-1] in channel_pairs:
            return channels[::-1]
        else:
            raise ValueError(f"invalid TDI channels {channels}, not in {channel_pairs}")

    def in_stored_order(self, channels: str) -> str:
        """Return two-channel string in the order stored in :attr:`.invevsdm`.

        :param channels: length-2 string describing two TDI channels.
        :type channels: str

        :return: a valid key to :attr:`.invevsdm`.
        :rtype: str

        :raises ValueError: if the specified TDI channel pair is not in :attr:`.invevsdm`,
            or if the passed string has the wrong length.
        """
        return self.in_given_order(channels, self._set_channel_pairs)

    @staticmethod
    def at_channels(_evsdm_or_invevsdm: data.WDMData, /, channel_pair: str) -> reps.WDM:
        """Return the spectral density matrix element for the specified channel pair.

        The channels do not have to be in the stored order, i.e. "XY" will automatically
        be changed to "YX" if that is the stored key.

        :param _evsdm_or_invevsdm: the (inverse) evolutionary spectral density matrix.
        :type _evsdm_or_invevsdm: data.WDMData

        :param channel_pair: length-2 string specifying two TDI channels.
        :type channel_pair: str

        :return: the matrix element.
        :rtype: reps.WDM
        """
        return _evsdm_or_invevsdm[
            WDMWhittleNoise.in_given_order(channel_pair, _evsdm_or_invevsdm.keys())
        ]

    def invevsdm_at_channels(self, channel_pair: str) -> reps.WDM:
        """Return the :attr:`.invevsdm` matrix element for the specified channels.

        The channels do not have to be in the stored order, i.e. "XY" will automatically
        be changed to "YX" if that is the stored key.

        :param channel_pair: length-2 string specifying two TDI channels.
        :type channel_pair: str

        :return: the :attr:`.invevsdm` matrix element.
        :rtype: representations.WDM
        """
        return self.invevsdm[self.in_stored_order(channel_pair)]

    def inner(self, left: data.WDMData, right: data.WDMData, /) -> float:
        """Compute the inner product of data in WDM form.

        :param left: data on the left of the product.
        :type left: data.WDMData

        :param right: data on the right of the product.
        :type right: data.WDMData

        :return: the inner product according to :attr:`.invevsdm`.
        :rtype: float

        :raises ValueError: if the TDI channels don't match.
        """
        # NOTE it might be better to normalize the invevsdm differently later.
        # for now its elements directly correspond to inverse covariance matrix elements.
        received_channels = set(left.keys()).union(set(right.keys()))
        if not received_channels.issubset(self._set_single_channels):
            raise ValueError(
                f"Inner product left anr right have keys {received_channels}, not contained in {self._set_single_channels}"
            )

        sum = 0.0
        for c1 in left.keys():
            for c2 in right.keys():
                pair = c1 + c2  # string concat
                sum += np.sum(
                    (left[c1] * right[c2] * self.invevsdm_at_channels(pair)).entries
                )
        return float(sum)

    @staticmethod
    def invert_sdm(_evsdm_or_invevsdm: data.WDMData, /) -> data.WDMData:
        """Invert a spectral density matrix.

        :param _evsdm_or_invevsdm: the matrix to invert, must be at most 3x3. Should be
            a valid spectral density matrix as specified by :meth:`is_valid_sdm`.
        :type _evsdm_or_invevsdm: data.WDMData
        :return: the matrix inverse. Only the keys present in the input will be
            present here, so only one of the equal off-diagonal elements (e.g. one
            of 'XY' and 'YX') will be returned.
        :rtype: data.WDMData
        """
        _ = WDMWhittleNoise.is_valid_sdm(_evsdm_or_invevsdm, raise_exception=True)
        return WDMWhittleNoise._invert_sdm(_evsdm_or_invevsdm)

    @staticmethod
    def _invert_sdm(_evsdm_or_invevsdm: data.WDMData, /) -> data.WDMData:
        """Invert an SDM without checking for input validity."""
        _d = _evsdm_or_invevsdm
        _set_keys = set(_d.keys())
        single_channels = WDMWhittleNoise._get_single_channels(_d)

        if len(single_channels) == 1:
            # inverse of 1-element matrix
            key = list(_d.keys())[0]
            invpsd = _d[key]
            psd_entries = 1 / invpsd.entries
            psd = invpsd.create_like(psd_entries)
            return data.WDMData(data={key: psd})

        elif len(single_channels) == 2:
            # inverse of 2x2 symmetric matrix
            #     [ a  b ]                 1    [  c  -b ]
            # A = [      ]  =>  inv(A) = ------ [        ]
            #     [ b  c ]               ac-b^2 [ -b   a ]
            c1, c2 = single_channels
            a = WDMWhittleNoise.at_channels(_d, c1 + c1)
            b = WDMWhittleNoise.at_channels(_d, c1 + c2)
            c = WDMWhittleNoise.at_channels(_d, c2 + c2)
            det = a * c - b * b
            evsdm_11 = c / det
            evsdm_12 = -b / det
            evsdm_22 = a / det
            return data.WDMData(
                data={
                    c1 + c1: evsdm_11,
                    WDMWhittleNoise.in_given_order(c1 + c2, _set_keys): evsdm_12,
                    c2 + c2: evsdm_22,
                }
            )

        elif len(single_channels) == 3:
            # inverse of 3x3 symmetric matrix
            #     [ a  b  c ]                 1    [ +Ma -Mb +Mc ]
            # A = [ b  d  e ]  =>  inv(A) = ------ [ -Mb +Md -Me ]
            #     [ c  e  f ]               det(A) [ +Mc -Me +Mf ]
            # where Mx are the minors
            # and det(A) = adf + 2bec - dcc - aee - fbb
            c1, c2, c3 = single_channels
            a = WDMWhittleNoise.at_channels(_d, c1 + c1)
            b = WDMWhittleNoise.at_channels(_d, c1 + c2)
            c = WDMWhittleNoise.at_channels(_d, c1 + c3)
            d = WDMWhittleNoise.at_channels(_d, c2 + c2)
            e = WDMWhittleNoise.at_channels(_d, c2 + c3)
            f = WDMWhittleNoise.at_channels(_d, c3 + c3)
            Ma = d * f - e * e
            Mb = b * f - e * c
            Mc = b * e - d * c
            Md = a * f - c * c
            Me = a * e - b * c
            Mf = a * d - b * b
            det = a * d * f + 2 * b * e * c - d * c * c - a * e * e - f * b * b
            return data.WDMData(
                data={
                    WDMWhittleNoise.in_given_order(c1 + c1, _set_keys): Ma / det,
                    WDMWhittleNoise.in_given_order(c1 + c2, _set_keys): -Mb / det,
                    WDMWhittleNoise.in_given_order(c1 + c3, _set_keys): Mc / det,
                    WDMWhittleNoise.in_given_order(c2 + c2, _set_keys): Md / det,
                    WDMWhittleNoise.in_given_order(c2 + c3, _set_keys): -Me / det,
                    WDMWhittleNoise.in_given_order(c3 + c3, _set_keys): Mf / det,
                }
            )

        else:
            assert False, "unreachable"

    def evsdm(self) -> data.WDMData:
        """Compute the evolutionary spectral density matrix by inverting :attr:`.invevsdm`.

        :return: The evolutionary spectral density matrix. Only the keys present in
            :attr:`.invevsdm` will be present here, so only one of the equal off-diagonal
            elements (e.g. one of 'XY' and 'YX') will be returned.
        :rtype: data.WDMData
        """
        return self._invert_sdm(self.invevsdm)

    def get_whitening_matrix(
        self, kind: Literal["cholesky"] = "cholesky"
    ) -> data.WDMData:
        """Get matrix W satisfying W^T W = invevsdm.

        The whitening matrix represents a linear transformation that yields
        unit variance white noise when applied to noise that follows this
        model. This is useful in detecting deviations from the model.

        Currently the only available whitening matrix is the one computed from
        Cholesky decomposition: invevsdm = L L^T, W = L^T. This W is an upper
        triangular matrix. It therefore does not treat TDI channels symmetrically.

        :param kind: the kind of whitening matrix, for now can only be "cholesky",
            returning an upper triangular matrix. Defaults to "cholesky"
        :type kind: str, optional
        :return: whitening matrix.
        :rtype: :class:`data.WDMData`

        :raises ValueError: if ``kind`` is not one of the available whitening methods.
        """
        if kind != "cholesky":
            raise ValueError(f"Unrecognized whitening method: '{kind}'")
        # TODO add some whitening method that treats XYZ symmetrically.

        if len(self.single_channels) == 1:
            # cholesky 1x1 case: just a square root
            key = list(self.invevsdm.keys())[0]
            invpsd = self.invevsdm[key]
            sqrt_entries = np.sqrt(invpsd.entries)
            whitening_mat = invpsd.create_like(sqrt_entries)
            return data.WDMData(data={key: whitening_mat})

        elif len(self.single_channels) == 2:
            # cholesky 2x2 symmetric real case
            #     [ a  b ]              [ μ  0 ]
            # A = [      ] = L L^T, L = [      ], W = L^T
            #     [ b  c ]              [ λ  σ ]
            c1, c2 = self.single_channels
            a = self.invevsdm_at_channels(c1 + c1)
            b = self.invevsdm_at_channels(c1 + c2)
            c = self.invevsdm_at_channels(c2 + c2)
            mu = np.sqrt(a)
            lam = b / mu
            sigma = np.sqrt(c - lam * lam)
            # return W = transpose of L
            return data.WDMData(
                data={
                    self.in_stored_order(c1 + c1): mu,
                    self.in_stored_order(c1 + c2): lam,  # above diagonal
                    self.in_stored_order(c1 + c2)[::-1]: 0 * lam,  # below diagonal
                    self.in_stored_order(c2 + c2): sigma,
                }
            )

        elif len(self.single_channels) == 3:
            # cholesky 3x3 symmetric real case
            #     [ a  b  c ]              [ μ  0  0 ]
            # A = [ b  d  e ] = L L^T, L = [ λ  σ  0 ], W = L^T
            #     [ c  e  f ]              [ ρ  κ  β ]
            c1, c2, c3 = self.single_channels
            a = self.invevsdm_at_channels(c1 + c1)
            b = self.invevsdm_at_channels(c1 + c2)
            c = self.invevsdm_at_channels(c1 + c3)
            d = self.invevsdm_at_channels(c2 + c2)
            e = self.invevsdm_at_channels(c2 + c3)
            f = self.invevsdm_at_channels(c3 + c3)
            mu = np.sqrt(a)
            lam = b / mu
            rho = c / mu
            sigma = np.sqrt(d - lam * lam)
            kappa = (e - lam * rho) / sigma
            beta = np.sqrt(f - rho * rho - kappa * kappa)
            # return W = transpose of L
            return data.WDMData(
                data={
                    self.in_stored_order(c1 + c1): mu,  # diagonal
                    self.in_stored_order(c2 + c2): sigma,
                    self.in_stored_order(c3 + c3): beta,
                    self.in_stored_order(c1 + c2): lam,  # above diagonal
                    self.in_stored_order(c1 + c3): rho,
                    self.in_stored_order(c2 + c3): kappa,
                    self.in_stored_order(c2 + c1)[::-1]: 0 * lam,  # below diagonal
                    self.in_stored_order(c3 + c1)[::-1]: 0 * rho,
                    self.in_stored_order(c3 + c2)[::-1]: 0 * kappa,
                }
            )

        else:
            assert False, "unreachable"

    def whiten(
        self, _data: data.WDMData, /, kind: Literal["cholesky"] = "cholesky"
    ) -> data.WDMData:
        """Whiten data according to the noise model.

        :param _data: data to whiten.
        :type _data: data.WDMData

        :param kind: kind of whitening to apply (see :meth:`.get_whitening_matrix`), defaults to "cholesky"
        :type kind: str, optional

        :return: whitened data
        :rtype: data.WDMData

        :raises ValueError: if the input data TDI channels do not match the noise model ones.
        """
        wmat = self.get_whitening_matrix(kind=kind)

        # make sure the single channels match
        _single_channels_wmat = set(list("".join(wmat.keys())))
        _single_channels_data = set(list("".join(_data.keys())))
        if _single_channels_data != _single_channels_wmat:
            raise ValueError(
                f"data channels {_single_channels_data} do not match noise model "
                f"channels {_single_channels_wmat}"
            )

        # multiply whitening matrix by data
        chan_list = list(_single_channels_wmat)
        whitened_data = data.zeros_like(_data)
        for c in chan_list:
            for c_dummy in chan_list:
                whitened_data[c] += wmat[c + c_dummy] * _data[c_dummy]

        return whitened_data
