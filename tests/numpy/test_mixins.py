# pyright: reportPrivateUsage=false

import operator
import unittest

import numpy as np

from typed_lisa_toolkit.types._mixins import NDArrayMixin


class _Probe(NDArrayMixin):
    """Minimal concrete mixin implementation to test delegation behavior."""

    def __init__(self):
        self.calls: list[tuple[str, object, bool, bool]] = []

    def __xp__(self, api_version: str | None = None):
        del api_version
        return np

    def _binary_op(
        self,
        other: object,
        op: object,
        /,
        *,
        reflected: bool = False,
        inplace: bool = False,
        **kwargs,
    ):
        del kwargs
        self.calls.append(("binary", op, reflected, inplace))
        return self if inplace else _Probe()

    def _unary_op(self, op: object, /, *args, **kwargs):
        del args, kwargs
        self.calls.append(("unary", op, False, False))
        return _Probe()


class TestNDArrayMixinDelegates(unittest.TestCase):
    def test_xp_property(self):
        probe = _Probe()
        self.assertIs(probe.xp, np)

    def test_binary_operator_delegation_flags(self):
        probe = _Probe()

        # comparisons
        _ = probe < 1
        _ = probe <= 1
        _ = probe == 1
        _ = probe != 1
        _ = probe > 1
        _ = probe >= 1

        # arithmetic and bitwise shifts/logical variants
        _ = probe + 1
        _ = probe.__radd__(1)
        _ = probe.__iadd__(1)
        _ = probe - 1
        _ = probe.__rsub__(1)
        _ = probe.__isub__(1)
        _ = probe * 1
        _ = probe.__rmul__(1)
        _ = probe.__imul__(1)
        _ = probe @ 1
        _ = probe.__rmatmul__(1)
        _ = probe.__imatmul__(1)
        _ = probe / 1
        _ = probe.__rtruediv__(1)
        _ = probe.__itruediv__(1)
        _ = probe // 1
        _ = probe.__rfloordiv__(1)
        _ = probe.__ifloordiv__(1)
        _ = probe % 1
        _ = probe.__rmod__(1)
        _ = probe.__imod__(1)
        _ = probe**1
        _ = probe.__rpow__(1)
        _ = probe.__ipow__(1)
        _ = probe << 1
        _ = probe.__rlshift__(1)
        _ = probe.__ilshift__(1)
        _ = probe >> 1
        _ = probe.__rrshift__(1)
        _ = probe.__irshift__(1)
        _ = probe & 1
        _ = probe.__rand__(1)
        _ = probe.__iand__(1)
        _ = probe ^ 1
        _ = probe.__rxor__(1)
        _ = probe.__ixor__(1)

        calls = probe.calls
        self.assertGreaterEqual(len(calls), 42)

        reflected_ops = [entry for entry in calls if entry[2]]
        inplace_ops = [entry for entry in calls if entry[3]]

        self.assertTrue(any(op == operator.add for _, op, _, _ in reflected_ops))
        self.assertTrue(any(op == operator.sub for _, op, _, _ in reflected_ops))
        self.assertTrue(any(op == operator.mul for _, op, _, _ in reflected_ops))
        self.assertTrue(any(op == operator.iadd for _, op, _, _ in inplace_ops))
        self.assertTrue(any(op == operator.isub for _, op, _, _ in inplace_ops))
        self.assertTrue(any(op == operator.imul for _, op, _, _ in inplace_ops))

    def test_unary_and_namespace_method_delegation(self):
        probe = _Probe()

        _ = -probe
        _ = +probe
        _ = abs(probe)
        _ = ~probe
        _ = probe.square()
        _ = probe.exp()
        _ = probe.sqrt()
        _ = probe.conj
        _ = probe.real
        _ = probe.imag
        _ = probe.abs()
        _ = probe.angle()
        _ = probe.unwrap()

        unary_ops = [entry[1] for entry in probe.calls if entry[0] == "unary"]
        self.assertIn(operator.neg, unary_ops)
        self.assertIn(operator.pos, unary_ops)
        self.assertIn(operator.abs, unary_ops)
        self.assertIn(operator.invert, unary_ops)
        self.assertIn(np.square, unary_ops)
        self.assertIn(np.exp, unary_ops)
        self.assertIn(np.sqrt, unary_ops)
        self.assertIn(np.conj, unary_ops)
        self.assertIn(np.real, unary_ops)
        self.assertIn(np.imag, unary_ops)
        self.assertIn(np.abs, unary_ops)
        self.assertIn(np.angle, unary_ops)
        self.assertIn(np.unwrap, unary_ops)
