import unittest
import numpy as np
from typed_lisa_toolkit.containers.arithdicts import (
    ModeDict,
    ChannelDict,
)
from typed_lisa_toolkit.containers import modes as _modes


class TestModeDict(unittest.TestCase):
    def setUp(self):
        self.modes = [_modes.Harmonic(2, 2), _modes.Harmonic(2, 3)]
        self.values = [np.sin(np.linspace(0, 1, 10)), np.cos(np.linspace(0, 1, 10))]
        self.data = {mode: value for mode, value in zip(self.modes, self.values)}
        self.mode_dict = ModeDict(self.data)

    def test_pick(self):
        result = next(iter(self.mode_dict.pick((2, 2)).values()))
        expected = np.sin(np.linspace(0, 1, 10))
        self.assertTrue(np.array_equal(result, expected))

    def test_drop(self):
        result = self.mode_dict.drop((2, 2))
        self.assertNotIn((2, 2), result.keys())

    def test_add(self):
        result = self.mode_dict + self.mode_dict
        expected = {k: v + v for k, v in self.data.items()}
        for k in result.keys():
            self.assertTrue(np.array_equal(result[k], expected[k]))

    def test_sub(self):
        result = self.mode_dict - self.mode_dict
        expected = {k: v - v for k, v in self.data.items()}
        for k in result.keys():
            self.assertTrue(np.array_equal(result[k], expected[k]))

    def test_mul(self):
        coeff = 2
        array = np.ones(10)
        with self.assertRaises(TypeError):
            array * self.mode_dict
        result = coeff * self.mode_dict * array
        expected = {k: v * coeff for k, v in self.data.items()}
        for k in result.keys():
            self.assertTrue(np.array_equal(result[k], expected[k]))

    def test_truediv(self):
        result = self.mode_dict / 2
        expected = {k: v / 2 for k, v in self.data.items()}
        for k in result.keys():
            self.assertTrue(np.array_equal(result[k], expected[k]))

    def test_neg(self):
        result = -self.mode_dict
        expected = {k: -v for k, v in self.data.items()}
        for k in result.keys():
            self.assertTrue(np.array_equal(result[k], expected[k]))

    def test_pass_through(self):
        result = self.mode_dict.pass_through(np.square)
        expected = {k: np.square(v) for k, v in self.data.items()}
        for k in result.keys():
            self.assertTrue(np.array_equal(result[k], expected[k]))

    def test_sum(self):
        result = self.mode_dict.sum()
        expected = sum(self.values, np.zeros_like(self.values[0]))
        self.assertTrue(np.array_equal(result, expected))

class TestChannelDict(unittest.TestCase):
    def setUp(self):
        self.channels = ["channel1", "channel2"]
        modes = [_modes.Harmonic(2, 2), _modes.Harmonic(2, 3)]
        values = [np.sin(np.linspace(0, 1, 10)), np.cos(np.linspace(0, 1, 10))]
        self.mode_dict = ModeDict({mode: value for mode, value in zip(modes, values)})
        self.values = [self.mode_dict, self.mode_dict * 2]
        self.data = {
            channel: value for channel, value in zip(self.channels, self.values)
        }
        self.channel_dict = ChannelDict(self.data)

    def test_add(self):
        result = self.channel_dict + self.channel_dict
        expected = {k: v + v for k, v in self.data.items()}
        for k in result.keys():
            for mode in result[k].keys():
                self.assertTrue(np.array_equal(result[k][mode], expected[k][mode]))

    def test_mul(self):
        coeff = 2
        array = np.ones(10)
        with self.assertRaises(TypeError):
            array * self.channel_dict
        result = coeff * self.channel_dict * array
        expected = {k: v * coeff for k, v in self.data.items()}
        for k in result.keys():
            for mode in result[k].keys():
                self.assertTrue(np.array_equal(result[k][mode], expected[k][mode]))

    def test_truediv(self):
        result = self.channel_dict / 2
        expected = {k: v / 2 for k, v in self.data.items()}
        for k in result.keys():
            for mode in result[k].keys():
                self.assertTrue(np.array_equal(result[k][mode], expected[k][mode]))

    def test_passthrough(self):
        result = self.channel_dict.pass_through(np.square)
        expected = {k: v.pass_through(np.square) for k, v in self.data.items()}
        for k in result.keys():
            for mode in result[k].keys():
                self.assertTrue(np.array_equal(result[k][mode], expected[k][mode]))


if __name__ == "__main__":
    unittest.main()
