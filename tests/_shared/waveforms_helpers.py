"""Shared helper builders for waveform backend tests."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

from typing import Any
from unittest.mock import MagicMock

from typed_lisa_toolkit.containers import modes
from typed_lisa_toolkit.containers import representations as reps
from typed_lisa_toolkit.containers.waveforms import (
    HarmonicProjectedWaveform,
    HarmonicWaveform,
    ProjectedWaveform,
)

# ---------------------------------------------------------------------------
# Fake container stubs for dense-maker tests (no real arrays needed)
# ---------------------------------------------------------------------------


class FakeResponse(dict[str, Any]):
    @property
    def channel_names(self):
        return tuple(self.keys())


class FakeHarmonicWaveform(dict[modes.Harmonic, FakeResponse]):
    @property
    def harmonics(self):
        return tuple(self.keys())


def make_mock_phasor(*, f_min: float, f_max: float):
    """Return (phasor, interpolated, embedded) mock triple with a preset frequency range."""
    phasor = MagicMock()
    phasor.f_min = f_min
    phasor.f_max = f_max

    interpolated = MagicMock()
    embedded = MagicMock()

    phasor.get_interpolated.return_value = interpolated
    interpolated.get_embedded.return_value = embedded
    return phasor, interpolated, embedded


def build_fake_harmonic_projected_waveform():
    """Build a two-mode, two-channel fake waveform where every phasor has a distinct
    frequency range so windowing assertions are unambiguous."""
    mode_22 = modes.Harmonic(2, 2)
    mode_33 = modes.Harmonic(3, 3)

    p22x, i22x, e22x = make_mock_phasor(f_min=1.0, f_max=3.0)
    p22y, i22y, e22y = make_mock_phasor(f_min=1.5, f_max=2.5)
    p33x, i33x, e33x = make_mock_phasor(f_min=0.5, f_max=2.0)
    p33y, i33y, e33y = make_mock_phasor(f_min=2.0, f_max=4.0)

    wf = FakeHarmonicWaveform(
        {
            mode_22: FakeResponse({"X": p22x, "Y": p22y}),
            mode_33: FakeResponse({"X": p33x, "Y": p33y}),
        }
    )

    handles = {
        mode_22: {"X": (p22x, i22x, e22x), "Y": (p22y, i22y, e22y)},
        mode_33: {"X": (p33x, i33x, e33x), "Y": (p33y, i33y, e33y)},
    }
    return wf, handles


def _make_fs(xp, frequencies, values):
    entries = xp.asarray(values, dtype=xp.complex128)[None, None, None, None, :]
    return reps.FrequencySeries((frequencies,), entries)


def build_harmonic_waveform_frequency_series(xp):
    frequencies = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    mode_22 = modes.Harmonic(2, 2)
    mode_33 = modes.Harmonic(3, 3)

    wf_22 = _make_fs(xp, frequencies, [1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j])
    wf_33 = _make_fs(xp, frequencies, [-0.5 + 1.0j, 0.25 + 0.0j, 1.5 - 0.25j])

    wf = HarmonicWaveform({mode_22: wf_22, mode_33: wf_33})

    return {
        "frequencies": frequencies,
        "modes": (mode_22, mode_33),
        "mode_22": mode_22,
        "mode_33": mode_33,
        "wf": wf,
        "wf_22": wf_22,
        "wf_33": wf_33,
    }


def build_harmonic_projected_frequency_waveform(xp):
    frequencies = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    mode_22 = modes.Harmonic(2, 2)
    mode_33 = modes.Harmonic(3, 3)

    resp_22 = ProjectedWaveform.from_dict(
        {
            "X": _make_fs(xp, frequencies, [1.0 + 0.0j, 2.0 - 1.0j, 3.0 + 0.5j]),
            "Y": _make_fs(xp, frequencies, [0.5 + 0.25j, -1.0 + 0.0j, 0.25 - 0.25j]),
        }
    )
    resp_33 = ProjectedWaveform.from_dict(
        {
            "X": _make_fs(xp, frequencies, [0.2 + 0.0j, -0.5 + 1.0j, 0.1 - 0.2j]),
            "Y": _make_fs(xp, frequencies, [1.0 + 0.0j, 1.5 + 0.0j, 2.0 + 0.0j]),
        }
    )

    wf = HarmonicProjectedWaveform({mode_22: resp_22, mode_33: resp_33})

    return {
        "frequencies": frequencies,
        "mode_22": mode_22,
        "mode_33": mode_33,
        "wf": wf,
        "resp_22": resp_22,
        "resp_33": resp_33,
    }


def build_nonhomogeneous_harmonic_projected_frequency_waveform(xp):
    frequencies_a = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
    frequencies_b = xp.asarray([1.5, 2.5, 3.5], dtype=xp.float64)

    mode_22 = modes.Harmonic(2, 2)
    mode_33 = modes.Harmonic(3, 3)

    resp_22 = ProjectedWaveform.from_dict(
        {
            "X": _make_fs(xp, frequencies_a, [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]),
            "Y": _make_fs(xp, frequencies_a, [0.5 + 0.0j, 1.0 + 0.0j, 1.5 + 0.0j]),
        }
    )
    resp_33 = ProjectedWaveform.from_dict(
        {
            "X": _make_fs(xp, frequencies_b, [0.2 + 0.0j, 0.4 + 0.0j, 0.6 + 0.0j]),
            "Y": _make_fs(xp, frequencies_b, [1.0 + 0.0j, 1.2 + 0.0j, 1.4 + 0.0j]),
        }
    )

    wf = HarmonicProjectedWaveform({mode_22: resp_22, mode_33: resp_33})

    return {
        "wf": wf,
        "mode_22": mode_22,
        "mode_33": mode_33,
        "resp_22": resp_22,
        "resp_33": resp_33,
    }
