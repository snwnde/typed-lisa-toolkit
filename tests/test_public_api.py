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
        self.assertIsNotNone(tlt.types)
        self.assertIsNotNone(tlt.shop)
        self.assertIsNotNone(tlt.viz)
        self.assertIsNotNone(tlt.utils)

    def test_common_symbols_are_reexported(self):
        self.assertIs(tlt.frequency_series, frequency_series)
        self.assertIs(tlt.time_series, time_series)
        self.assertIs(tlt.phasor, phasor)
        self.assertIs(tlt.stft, stft)
        self.assertIs(tlt.wdm, wdm)
        self.assertIs(tlt.sum_harmonics, sum_harmonics)
        self.assertIs(tlt.load_data, load_data)
        self.assertIs(tlt.load_ldc_data, load_ldc_data)
        self.assertIs(tlt.tsdata, tsdata)
        self.assertIs(tlt.fsdata, fsdata)
        self.assertIs(tlt.stftdata, stftdata)
        self.assertIs(tlt.wdmdata, wdmdata)
        self.assertIs(tlt.timed_fsdata, timed_fsdata)
        self.assertIs(tlt.linspace, linspace)
        self.assertIs(tlt.cast_mode, cast_mode)
        self.assertIs(tlt.densify_phasor, densify_phasor)
        self.assertIs(tlt.densify_phasor_hw, densify_phasor_hw)
        self.assertIs(tlt.densify_phasor_pw, densify_phasor_pw)
        self.assertIs(tlt.densify_phasor_hpw, densify_phasor_hpw)
        self.assertIs(tlt.phasor_to_fs_hw, phasor_to_fs_hw)
        self.assertIs(tlt.phasor_to_fs_pw, phasor_to_fs_pw)
        self.assertIs(tlt.phasor_to_fs_hpw, phasor_to_fs_hpw)
        self.assertIs(tlt.get_dense_maker, get_dense_maker)
        self.assertIs(tlt.harmonic_waveform, harmonic_waveform)
        self.assertIs(tlt.homogeneous_harmonic_waveform, homogeneous_harmonic_waveform)
        self.assertIs(tlt.projected_waveform, projected_waveform)
        self.assertIs(tlt.harmonic_projected_waveform, harmonic_projected_waveform)
        self.assertIs(
            tlt.homogeneous_harmonic_projected_waveform,
            homogeneous_harmonic_projected_waveform,
        )
        self.assertIs(tlt.hw, hw)
        self.assertIs(tlt.hhw, hhw)
        self.assertIs(tlt.pw, pw)
        self.assertIs(tlt.hpw, hpw)
        self.assertIs(tlt.hhpw, hhpw)
        self.assertIs(tlt.construct_fsdata, construct_fsdata)
        self.assertIs(tlt.construct_stftdata, construct_stftdata)
        self.assertIs(tlt.construct_timed_fsdata, construct_timed_fsdata)
        self.assertIs(tlt.construct_tsdata, construct_tsdata)
        self.assertIs(tlt.construct_wdmdata, construct_wdmdata)

    def test_types_namespace_exports_are_available(self):
        self.assertIs(tlt.types.TSData, TSData)
        self.assertIs(tlt.types.FrequencySeries, FrequencySeries)

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
        ):
            self.assertIn(name, tlt.__all__)


if __name__ == "__main__":
    unittest.main()
