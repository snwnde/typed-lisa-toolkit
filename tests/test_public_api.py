"""Tests for top-level public API re-exports."""

import unittest

import typed_lisa_toolkit as tlt
from typed_lisa_toolkit import (
    cast_mode,
    construct_fsdata,
    construct_stftdata,
    construct_timed_fsdata,
    construct_tsdata,
    construct_wdmdata,
    densify_phasor,
    densify_phasor_hpw,
    densify_phasor_hw,
    densify_phasor_pw,
    frequency_series,
    fsdata,
    get_dense_maker,
    harmonic_projected_waveform,
    harmonic_waveform,
    hhpw,
    hhw,
    homogeneous_harmonic_projected_waveform,
    homogeneous_harmonic_waveform,
    hpw,
    hw,
    linspace,
    load_data,
    load_ldc_data,
    phasor,
    phasor_to_fs_hpw,
    phasor_to_fs_hw,
    phasor_to_fs_pw,
    projected_waveform,
    pw,
    stft,
    stftdata,
    sum_harmonics,
    time_series,
    timed_fsdata,
    tsdata,
    wdm,
    wdmdata,
)
from typed_lisa_toolkit.types import FrequencySeries, TSData


class TestPublicApi(unittest.TestCase):
    def test_module_reexports_are_available(self):
        assert tlt.types is not None
        assert tlt.shop is not None
        assert tlt.viz is not None
        assert tlt.utils is not None

    def test_common_symbols_are_reexported(self):
        assert tlt.frequency_series is frequency_series
        assert tlt.time_series is time_series
        assert tlt.phasor is phasor
        assert tlt.stft is stft
        assert tlt.wdm is wdm
        assert tlt.sum_harmonics is sum_harmonics
        assert tlt.load_data is load_data
        assert tlt.load_ldc_data is load_ldc_data
        assert tlt.tsdata is tsdata
        assert tlt.fsdata is fsdata
        assert tlt.stftdata is stftdata
        assert tlt.wdmdata is wdmdata
        assert tlt.timed_fsdata is timed_fsdata
        assert tlt.linspace is linspace
        assert tlt.cast_mode is cast_mode
        assert tlt.densify_phasor is densify_phasor
        assert tlt.densify_phasor_hw is densify_phasor_hw
        assert tlt.densify_phasor_pw is densify_phasor_pw
        assert tlt.densify_phasor_hpw is densify_phasor_hpw
        assert tlt.phasor_to_fs_hw is phasor_to_fs_hw
        assert tlt.phasor_to_fs_pw is phasor_to_fs_pw
        assert tlt.phasor_to_fs_hpw is phasor_to_fs_hpw
        assert tlt.get_dense_maker is get_dense_maker
        assert tlt.harmonic_waveform is harmonic_waveform
        assert tlt.homogeneous_harmonic_waveform is homogeneous_harmonic_waveform
        assert tlt.projected_waveform is projected_waveform
        assert tlt.harmonic_projected_waveform is harmonic_projected_waveform
        assert (
            tlt.homogeneous_harmonic_projected_waveform
            is homogeneous_harmonic_projected_waveform
        )
        assert tlt.hw is hw
        assert tlt.hhw is hhw
        assert tlt.pw is pw
        assert tlt.hpw is hpw
        assert tlt.hhpw is hhpw
        assert tlt.construct_fsdata is construct_fsdata
        assert tlt.construct_stftdata is construct_stftdata
        assert tlt.construct_timed_fsdata is construct_timed_fsdata
        assert tlt.construct_tsdata is construct_tsdata
        assert tlt.construct_wdmdata is construct_wdmdata

    def test_types_namespace_exports_are_available(self):
        assert tlt.types.TSData is TSData
        assert tlt.types.FrequencySeries is FrequencySeries

    def test_top_level_public_names_are_listed(self):
        for name in (
            "types",
            "shop",
            "viz",
            "utils",
            "frequency_series",
            "time_series",
            "phasor",
            "stft",
            "wdm",
            "sum_harmonics",
            "load_data",
            "load_ldc_data",
            "hw",
            "hhw",
            "pw",
            "hpw",
            "hhpw",
            "harmonic_waveform",
            "homogeneous_harmonic_waveform",
            "projected_waveform",
            "harmonic_projected_waveform",
            "homogeneous_harmonic_projected_waveform",
            "linspace",
            "cast_mode",
            "densify_phasor",
            "densify_phasor_hw",
            "densify_phasor_pw",
            "densify_phasor_hpw",
            "phasor_to_fs_hw",
            "phasor_to_fs_pw",
            "phasor_to_fs_hpw",
            "get_dense_maker",
            "tsdata",
            "fsdata",
            "stftdata",
            "wdmdata",
            "timed_fsdata",
            "construct_tsdata",
            "construct_fsdata",
            "construct_timed_fsdata",
            "construct_stftdata",
            "construct_wdmdata",
            "build_grid2d",
            "make_sdm",
            "noise_model",
            "whittle",
        ):
            assert name in tlt.__all__


if __name__ == "__main__":
    unittest.main()
