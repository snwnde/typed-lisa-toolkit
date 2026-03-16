"""Shared test mixins for data-container tests."""
# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportCallIssue=false

import numpy as np

from typed_lisa_toolkit.containers import data
from typed_lisa_toolkit.containers.representations import TimeSeries


class DataAbstractBranchesMixin:
    """Mix-in testing abstract/NotImplementedError branches in data-container mixins.

    Uses plain numpy arrays throughout — the tested code paths are
    backend-agnostic and only check NotImplementedError semantics.
    """

    def test_get_subset_mixin_notimplemented_methods(self):
        class Dummy(data._GetSubsetMixin[TimeSeries]):
            def __getitem__(self, key):
                raise KeyError(key)

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        dummy = Dummy()
        with self.assertRaises(NotImplementedError):  # type: ignore[attr-defined]
            dummy.create_new(None, ())  # type: ignore[arg-type]
        with self.assertRaises(NotImplementedError):  # type: ignore[attr-defined]
            dummy._get_plotter()

    def test_embeddable_mixin_notimplemented_create_new(self):
        class Dummy(data._EmbeddableMixin[TimeSeries]):
            def __getitem__(self, key):
                raise KeyError(key)

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

        dummy = Dummy()
        with self.assertRaises(NotImplementedError):  # type: ignore[attr-defined]
            dummy.create_new(None, ())  # type: ignore[arg-type]

    def test_data_base_get_plotter_notimplemented(self):
        class Dummy(data.Data[TimeSeries]):
            _reps_type = TimeSeries

        times = np.linspace(0.0, 1.0, 4)
        representation = TimeSeries((times,), np.ones((1, 1, 1, 1, 4)))
        dummy = Dummy(representation, ("X",))

        with self.assertRaises(NotImplementedError):  # type: ignore[attr-defined]
            dummy._get_plotter()
