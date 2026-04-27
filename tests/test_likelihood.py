from types import ModuleType

import numpy as np
import numpy.testing as npt

from typed_lisa_toolkit import (
    fsdata,
    make_sdm,
    noise_model,
    sum_harmonics,
    whittle,
)
from typed_lisa_toolkit.types import (
    FDWhittleLikelihood,
    FSData,
    Harmonic,
    HomogeneousHarmonicProjectedWaveform,
    SpectralDensity,
    UniformFrequencySeries,
)


def test_fd_whittle_classmethod_formulas():
    assert FDWhittleLikelihood.log_likelihood_ratio(5.0, 2.0) == 4.0  # pyright: ignore[reportArgumentType]
    assert FDWhittleLikelihood.log_likelihood(4.0, 2.0) == 3.0  # pyright: ignore[reportArgumentType]


def test_cross_product_and_template_square_match_noise_model(
    fsdata: FSData, sdm: SpectralDensity
):
    model = noise_model(sdm)
    likelihood = whittle(fsdata, model)

    cross = np.asarray(likelihood.get_cross_product(fsdata))
    template_square = np.asarray(likelihood.get_template_square(fsdata))

    npt.assert_allclose(
        cross,
        np.asarray(model.reset().get_scalar_product(fsdata, fsdata)),
    )
    npt.assert_allclose(
        template_square,
        np.asarray(model.reset().get_scalar_product(fsdata, fsdata)),
    )


def test_log_likelihood_matches_closed_form(fsdata: FSData, sdm: SpectralDensity):
    model = noise_model(sdm)
    likelihood = whittle(fsdata, model)

    got = np.asarray(likelihood.get_log_likelihood(fsdata))
    expected = (
        np.asarray(model.reset().get_scalar_product(fsdata, fsdata))
        - 0.5 * np.asarray(model.reset().get_scalar_product(fsdata, fsdata))
        - 0.5 * np.asarray(model.reset().get_scalar_product(fsdata, fsdata))
    )

    npt.assert_allclose(got, expected)


def test_harmonic_projected_template_is_summed_before_evaluation(
    homogeneous_harmonic_projected_waveform: HomogeneousHarmonicProjectedWaveform[
        Harmonic, UniformFrequencySeries
    ],
    sdm: SpectralDensity,
):
    data = fsdata(sum_harmonics(homogeneous_harmonic_projected_waveform))
    model = noise_model(sdm)
    likelihood = whittle(data, model)

    got = np.asarray(
        likelihood.get_cross_product(homogeneous_harmonic_projected_waveform)
    )
    expected = np.asarray(
        model.reset().get_scalar_product(
            data,
            fsdata(sum_harmonics(homogeneous_harmonic_projected_waveform)),
        ),
    )

    npt.assert_allclose(got, expected)


def test_template_is_restricted_to_its_frequency_band(xp: ModuleType, fsdata: FSData):
    freqs = fsdata.frequencies
    template = fsdata.get_subset(interval=(1.0, 3.0))
    kernel = xp.broadcast_to(xp.eye(3), (len(freqs), 3, 3)).copy()
    model = noise_model(
        make_sdm(kernel, frequencies=freqs, channel_names=("X", "Y", "Z")),
    )
    likelihood = whittle(fsdata, model)

    got = xp.asarray(likelihood.get_cross_product(template))
    expected = xp.asarray(
        noise_model(
            make_sdm(
                kernel[0:4],
                frequencies=xp.asarray(template.frequencies),
                channel_names=("X", "Y", "Z"),
            ),
        ).get_scalar_product(fsdata.get_subset(interval=(1.0, 3.0)), template),
    )

    npt.assert_allclose(got, expected)


def test_whittle_factory_returns_fd_whittle_likelihood(
    fsdata: FSData, sdm: SpectralDensity
):
    model = noise_model(sdm)

    likelihood = whittle(fsdata, model)

    assert isinstance(likelihood, FDWhittleLikelihood)
