"""Shared test mixins for data-container tests."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import numpy as np
from typing import TYPE_CHECKING
from typed_lisa_toolkit.containers import data
from typed_lisa_toolkit.containers.representations import TimeSeries

if TYPE_CHECKING:
    from typed_lisa_toolkit.containers.representations import Axis


class DataAbstractBranchesMixin:
    """Mix-in testing abstract/NotImplementedError branches in data-container mixins.

    Uses plain numpy arrays throughout — the tested code paths are
    backend-agnostic and only check NotImplementedError semantics.
    """

    def test_data_base_get_plotter_notimplemented(self):
        class Dummy(data.Data[TimeSeries["Axis"]]):
            _reps_type = TimeSeries["Axis"]

        times = np.linspace(0.0, 1.0, 4)
        representation = TimeSeries["Axis"]((times,), np.ones((1, 1, 1, 1, 4)))
        dummy = Dummy(representation, ("X",))

        with self.assertRaises(NotImplementedError):  # type: ignore[attr-defined]
            dummy._get_plotter()
