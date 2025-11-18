import unittest
from unittest.mock import MagicMock, create_autospec
import numpy as np
from numpy.typing import NDArray
import scipy.integrate  # type: ignore[import]
from typed_lisa_toolkit.containers.data import FSData, TSData, WDMData
from typed_lisa_toolkit.containers.representations import (
    FrequencySeries,
    TimeSeries,
    WDM,
    Linspace,
)

from typed_lisa_toolkit.containers.noisemodel import (
    StationaryFDNoise,
    FDNoiseModel,
    _StationaryNoiseModel,
    _CacheNoiseModel,
    _collect_frequencies,
    WDMWhittleNoise,
)


class MixinTestUtils(unittest.TestCase):
    "common test utilities"

    def assertAllClose(self, x, y: NDArray | Linspace, /, **kwargs):
        "This won't work if the second argument is a _Series!"
        # that's because allclose will call asanyarray on y, and _Series is
        # not a subclass of ndarray, meaning it will get converted to an
        # array of dtype object, containing your series as the only element.
        # this will subsequently fail isfinite().
        self.assertTrue(np.allclose(x, y, **kwargs), msg=f"first = {x}, second = {y}")

    def assertArrayEq(self, x, y, /):
        self.assertTrue(np.array_equal(x, y), msg=f"first = {x}, second = {y}")


class TestFDSensitivity(unittest.TestCase):
    def setUp(self):
        self.noise_model = create_autospec(StationaryFDNoise, instance=True)
        self.noise_cache = MagicMock(spec=FSData)
        self.frequencies = np.array([0.1, 0.2, 0.3])
        self.entries = np.array([1.0, 2.0, 3.0])
        self.data = FSData(
            {"channel1": FrequencySeries(self.frequencies, self.entries)}
        )

    def test_noise_model_sensitivity_creation(self):
        sensitivity = FDNoiseModel.make(fd_noise=self.noise_model)
        self.assertIsInstance(sensitivity, _StationaryNoiseModel)

    def test_cache_sensitivity_creation(self):
        sensitivity = FDNoiseModel.make(noise_cache=self.noise_cache)
        self.assertIsInstance(sensitivity, _CacheNoiseModel)

    def test_invalid_sensitivity_creation(self):
        with self.assertRaises(ValueError):
            FDNoiseModel.make()

    def test_get_noise_psd_noise_model(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        noise_psd = sensitivity.get_noise_psd(_collect_frequencies(self.data))
        self.assertTrue(np.array_equal(noise_psd["channel1"].entries, self.entries))

    def test_get_noise_psd_cache(self):
        self.noise_cache.get_subset.return_value = self.data
        sensitivity = _CacheNoiseModel(self.noise_cache)
        noise_psd = sensitivity.get_noise_psd(_collect_frequencies(self.data))
        self.assertEqual(noise_psd, self.data)

    def test_get_integrand(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        integrand = sensitivity.get_integrand(self.data, self.data)
        expected_integrand = (
            4.0 * (1.0 / self.entries) * self.entries * self.entries.conj()
        )
        self.assertTrue(
            np.array_equal(integrand["channel1"].entries, expected_integrand)
        )

    def test_get_complex_scalar_product(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        product = sensitivity.get_complex_scalar_product(self.data, self.data)
        expected_product = scipy.integrate.trapezoid(
            4.0 * (1.0 / self.entries) * self.entries * self.entries.conj(),
            x=self.frequencies,
        )
        self.assertTrue(np.array_equal(product["channel1"], expected_product))

    def test_get_scalar_product(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        product = sensitivity.get_scalar_product(self.data, self.data)
        expected_product = np.real(
            scipy.integrate.trapezoid(
                4.0 * (1.0 / self.entries) * self.entries * self.entries.conj(),
                x=self.frequencies,
            )
        )
        self.assertTrue(np.array_equal(product["channel1"], expected_product))

    def test_get_cross_correlation(self):
        times = np.array([0.0, 1.0, 2.0])
        tsdata = TSData({"channel1": TimeSeries(times, np.sin(times))})
        timed_data = tsdata.to_fsdata()
        another_data = FSData(
            {
                "channel1": FrequencySeries(
                    timed_data.frequencies, np.cos(timed_data.frequencies)
                )
            }
        )
        self.noise_model.psd.return_value = np.ones_like(timed_data.frequencies)
        sensitivity = _StationaryNoiseModel(self.noise_model)
        cross_correlation = sensitivity.get_cross_correlation(timed_data, another_data)
        self.assertIsInstance(next(iter(cross_correlation.values())), TimeSeries)

    def test_get_whitened(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        whitened = sensitivity.get_whitened(self.data)
        expected_whitened = self.entries / np.sqrt(self.entries)
        self.assertTrue(np.array_equal(whitened["channel1"].entries, expected_whitened))

    def test_get_overlap(self):
        self.noise_model.psd.return_value = self.entries
        sensitivity = _StationaryNoiseModel(self.noise_model)
        overlap = sensitivity.get_overlap(self.data, self.data)
        self.assertTrue(np.array_equal(overlap["channel1"], 1.0))


class TestWDMWhittleNoise(MixinTestUtils, unittest.TestCase):

    @staticmethod
    def mat2_from_tf(e2x2, m, n):
        # make a 2x2 matrix from an sdm assuming the channels are "XY"
        a = e2x2["XX"].entries[m, n]
        b = e2x2["XY"].entries[m, n]
        c = e2x2["YY"].entries[m, n]
        return np.array([[a, b], [b, c]])

    @staticmethod
    def mat3_from_tf(e3x3, m, n):
        # make a 3x3 matrix from an sdm assuming the channels are "XYZ"
        a = e3x3["XX"].entries[m, n]
        b = e3x3["XY"].entries[m, n]
        c = e3x3["XZ"].entries[m, n]
        d = e3x3["YY"].entries[m, n]
        e = e3x3["YZ"].entries[m, n]
        f = e3x3["ZZ"].entries[m, n]
        return np.array([[a, b, c], [b, d, e], [c, e, f]])

    def assertDataEqual(self, left: WDMData, right: WDMData):
        self.assertEqual(set(left.keys()), set(right.keys()))
        for k in left.keys():
            self.assertArrayEq(left[k].times, right[k].times)
            self.assertArrayEq(left[k].frequencies, right[k].frequencies)
            self.assertArrayEq(left[k].entries, right[k].entries)

    def assertDataAllClose(self, left: WDMData, right: WDMData):
        self.assertEqual(set(left.keys()), set(right.keys()))
        for k in left.keys():
            self.assertAllClose(left[k].times, right[k].times)
            self.assertAllClose(left[k].frequencies, right[k].frequencies)
            self.assertAllClose(left[k].entries, right[k].entries)

    def setUp(self):
        self.Nf, self.Nt, self.dT = 2, 2, 0.123843
        self.dF = 1 / (2 * self.dT)
        self.tgrid = self.dT * np.arange(self.Nt)
        self.fgrid = self.dF * np.arange(self.Nf)

        # correlation matrices for xx, xy, ...
        self.wa = WDM(self.tgrid, self.fgrid, np.array([[1.00, 1.10], [1.00, 1.10]]))
        self.wb = WDM(self.tgrid, self.fgrid, np.array([[0.50, 0.52], [0.50, 0.52]]))
        self.wc = WDM(self.tgrid, self.fgrid, np.array([[0.55, 0.52], [0.55, 0.52]]))
        self.wd = WDM(self.tgrid, self.fgrid, np.array([[1.30, 1.52], [1.30, 1.52]]))
        self.we = WDM(self.tgrid, self.fgrid, np.array([[0.31, 0.31], [0.31, 0.31]]))
        self.wf = WDM(self.tgrid, self.fgrid, np.array([[0.87, 0.92], [0.87, 0.92]]))
        self.w_ones = WDM(self.tgrid, self.fgrid, np.ones((self.Nf, self.Nt)))
        self.w_zeros = WDM(self.tgrid, self.fgrid, np.zeros((self.Nf, self.Nt)))

        # put the w* together in WDMData
        self.c1 = WDMData({"XX": self.wa})
        self.c2 = WDMData({"XX": self.wa, "XY": self.wb, "YY": self.wd})
        self.c3 = WDMData(
            {
                "XX": self.wa,
                "XY": self.wb,
                "XZ": self.wc,
                "YY": self.wd,
                "YZ": self.we,
                "ZZ": self.wf,
            }
        )
        self.c_id = WDMData({"XX": self.w_ones, "XY": self.w_zeros, "YY": self.w_ones})

        # wrap them in noise models
        self.n1 = WDMWhittleNoise(self.c1)
        self.n2 = WDMWhittleNoise(self.c2)
        self.n3 = WDMWhittleNoise(self.c3)
        self.n_id = WDMWhittleNoise(self.c_id)

        # some data for testing
        self.ww0 = WDM(self.tgrid, self.fgrid, np.zeros((self.Nf, self.Nt)))
        self.ww1 = WDM(
            self.tgrid, self.fgrid, np.random.normal(size=(self.Nf, self.Nt))
        )
        self.ww2 = WDM(
            self.tgrid, self.fgrid, np.random.normal(size=(self.Nf, self.Nt))
        )
        self.ww3 = WDM(
            self.tgrid, self.fgrid, np.random.normal(size=(self.Nf, self.Nt))
        )
        self.ww4 = WDM(
            self.tgrid, self.fgrid, np.random.normal(size=(self.Nf, self.Nt))
        )
        self.ww5 = WDM(
            self.tgrid, self.fgrid, np.random.normal(size=(self.Nf, self.Nt))
        )
        self.ww6 = WDM(
            self.tgrid, self.fgrid, np.random.normal(size=(self.Nf, self.Nt))
        )
        self.d1_0 = WDMData({"X": self.ww0})  # just zero
        self.d1_1 = WDMData({"X": self.ww1})
        self.d1_2 = WDMData({"X": self.ww2})
        self.d2_0 = WDMData({"X": self.ww1, "Y": self.ww0})  # only has an X component
        self.d2_1 = WDMData({"X": self.ww1, "Y": self.ww2})
        self.d2_2 = WDMData({"X": self.ww3, "Y": self.ww3})
        self.d3_0 = WDMData(
            {"X": self.ww1, "Y": self.ww0, "Z": self.ww0}
        )  # only has an X component
        self.d3_1 = WDMData({"X": self.ww1, "Y": self.ww2, "Z": self.ww3})
        self.d3_2 = WDMData({"X": self.ww4, "Y": self.ww5, "Z": self.ww6})

    def test_init(self):
        ########## test the happy path first ##########
        # properties to test:
        # - valid invevsdm is stored as-is if no optional params passed
        # - inferred order in single-channels is correct
        # - passing a channel order doesn't crash and actually reorders sdm keys
        # - passing a channel order makes the circular sdm keys valid
        # - the stored invevsdm is always upper-triangular in the obj.singlechannels order
        # - passed invevsdm values (grids and entries) are not altered at all
        # - invert_sdm works

        # 1x1
        d = WDMData({"00": self.wa})
        n = WDMWhittleNoise(d)
        self.assertDataEqual(d, n.invevsdm)
        self.assertEqual(n.single_channels, ["0"])
        n2 = WDMWhittleNoise(d, channel_order="0")
        self.assertDataEqual(d, n2.invevsdm)
        self.assertEqual(n.single_channels, ["0"])
        id = WDMData({"00": 1 / self.wa})
        n3 = WDMWhittleNoise(d, invert_sdm=True)
        self.assertDataEqual(id, n3.invevsdm)
        del d, id, n, n2, n3  # this is to avoid mistakes with reuse of symbols

        # 2x2
        d_upper = WDMData({"aa": self.wa, "ab": self.wb, "bb": self.wd})
        d_lower = WDMData({"aa": self.wa, "ba": self.wb, "bb": self.wd})
        n1 = WDMWhittleNoise(d_upper)
        self.assertDataEqual(d_upper, n1.invevsdm)
        self.assertEqual(["a", "b"], n1.single_channels)
        n2 = WDMWhittleNoise(d_upper, channel_order="ab")
        n3 = WDMWhittleNoise(d_lower, channel_order="ab")
        self.assertDataEqual(d_upper, n2.invevsdm)
        self.assertDataEqual(d_upper, n3.invevsdm)
        self.assertEqual(["a", "b"], n2.single_channels)
        self.assertEqual(["a", "b"], n3.single_channels)
        id_upper = WDMWhittleNoise.invert_sdm(d_upper)
        n4 = WDMWhittleNoise(d_upper, invert_sdm=True)
        self.assertDataEqual(id_upper, n4.invevsdm)
        self.assertEqual(["a", "b"], n4.single_channels)
        del d_upper, d_lower, n1, n2, n3, n4, id_upper

        # 3x3
        # inferred orders upper=123 lower=321 mixed=312 circl -> no inferred order, but error
        wa, wb, wc, wd, we, wf = self.wa, self.wb, self.wc, self.wd, self.we, self.wf
        d_upper = WDMData({"11": wa, "12": wb, "13": wc, "22": wd, "23": we, "33": wf})
        d_lower = WDMData({"11": wa, "21": wb, "31": wc, "22": wd, "32": we, "33": wf})
        d_mixed = WDMData({"11": wa, "12": wb, "31": wc, "22": wd, "32": we, "33": wf})
        d_circl = WDMData({"11": wa, "12": wb, "31": wc, "22": wd, "23": we, "33": wf})
        n1_upper = WDMWhittleNoise(d_upper)
        n1_lower = WDMWhittleNoise(d_lower)
        n1_mixed = WDMWhittleNoise(d_mixed)
        self.assertDataEqual(d_upper, n1_upper.invevsdm)
        self.assertDataEqual(d_lower, n1_lower.invevsdm)
        self.assertDataEqual(d_mixed, n1_mixed.invevsdm)
        self.assertEqual(list("123"), n1_upper.single_channels)
        self.assertEqual(list("321"), n1_lower.single_channels)
        self.assertEqual(list("312"), n1_mixed.single_channels)
        n2_upper = WDMWhittleNoise(d_upper, channel_order="123")
        n2_lower = WDMWhittleNoise(d_lower, channel_order="123")
        n2_mixed = WDMWhittleNoise(d_mixed, channel_order="123")
        n2_circl = WDMWhittleNoise(d_circl, channel_order="123")
        self.assertDataEqual(d_upper, n2_upper.invevsdm)
        self.assertDataEqual(d_upper, n2_lower.invevsdm)
        self.assertDataEqual(d_upper, n2_mixed.invevsdm)
        self.assertDataEqual(d_upper, n2_circl.invevsdm)
        self.assertEqual(list("123"), n2_upper.single_channels)
        self.assertEqual(list("123"), n2_lower.single_channels)
        self.assertEqual(list("123"), n2_mixed.single_channels)
        self.assertEqual(list("123"), n2_circl.single_channels)
        id_upper = WDMWhittleNoise.invert_sdm(d_upper)
        n3 = WDMWhittleNoise(d_upper, invert_sdm=True)
        self.assertDataEqual(id_upper, n3.invevsdm)
        del d_upper, d_lower, d_mixed, d_circl, id_upper
        del n1_upper, n1_lower, n1_mixed, n2_upper, n2_lower, n2_mixed, n2_circl, n3

        ########## now test error behaviors #########
        # keys not length-2
        with self.assertRaises(ValueError):
            WDMWhittleNoise(WDMData({"AAA": wa}))
        with self.assertRaises(ValueError):
            WDMWhittleNoise(WDMData({"A": wa, "E": wa, "T": wa}))

        # empty data
        with self.assertRaises(ValueError):
            WDMWhittleNoise(WDMData({}))

        # more than 3 TDI channels
        datadict = {}
        for n, c1 in enumerate("XYZW"):
            for c2 in "XYZW"[n:]:
                datadict[c1 + c2] = wa
        data = WDMData(datadict)
        with self.assertRaises(ValueError):
            WDMWhittleNoise(data)
        del datadict, data

        # wrong number of keys
        with self.assertRaises(ValueError):
            WDMWhittleNoise(WDMData({"XX": wa, "XY": wb}))
        with self.assertRaises(ValueError):
            WDMWhittleNoise(WDMData({"XX": wa, "XY": wb, "YY": wd, "YZ": wb}))

        # circular permutation -> no possible inferred order
        d_circl = WDMData({"11": wa, "12": wb, "31": wc, "22": wd, "23": we, "33": wf})
        with self.assertRaises(ValueError):
            WDMWhittleNoise(d_circl)
        del d_circl

        # incompatible sdm and channel_order (2 vs 3 TDI channels)
        with self.assertRaises(ValueError):
            WDMWhittleNoise(
                WDMData({"XX": wa, "XY": wb, "YY": wd}), channel_order="XYZ"
            )

    def test_is_valid_sdm(self):
        # the exception-raising paths have already been tested in test_init(),
        # so here I just try to cover the returns.

        # setup
        wa, wb, wc, wd, we, wf = self.wa, self.wb, self.wc, self.wd, self.we, self.wf

        ##### for valid inputs, return True
        self.assertTrue(WDMWhittleNoise.is_valid_sdm(self.c1))
        self.assertTrue(WDMWhittleNoise.is_valid_sdm(self.c2))
        self.assertTrue(WDMWhittleNoise.is_valid_sdm(self.c3))
        self.assertTrue(WDMWhittleNoise.is_valid_sdm(self.c_id))

        ##### for invalid inputs, return False

        # keys not length-2
        self.assertFalse(WDMWhittleNoise.is_valid_sdm(WDMData({"AAA": wa})))
        self.assertFalse(
            WDMWhittleNoise.is_valid_sdm(WDMData({"A": wa, "E": wa, "T": wa}))
        )

        # empty data
        self.assertFalse(WDMWhittleNoise.is_valid_sdm(WDMData({})))

        # more than 3 TDI channels
        datadict = {}
        for n, c1 in enumerate("XYZW"):
            for c2 in "XYZW"[n:]:
                datadict[c1 + c2] = wa
        data = WDMData(datadict)
        self.assertFalse(WDMWhittleNoise.is_valid_sdm(data))
        del datadict, data

        # wrong number of keys
        self.assertFalse(WDMWhittleNoise.is_valid_sdm(WDMData({"XX": wa, "XY": wb})))
        self.assertFalse(
            WDMWhittleNoise.is_valid_sdm(
                WDMData({"XX": wa, "XY": wb, "YY": wd, "YZ": wb})
            )
        )

        # circular permutation -> no possible inferred order
        d_circl = WDMData({"11": wa, "12": wb, "31": wc, "22": wd, "23": we, "33": wf})
        self.assertFalse(WDMWhittleNoise.is_valid_sdm(d_circl))
        del d_circl

        # incompatible sdm and channel_order (2 vs 3 TDI channels)
        self.assertFalse(
            WDMWhittleNoise.is_valid_sdm(
                WDMData({"XX": wa, "XY": wb, "YY": wd}), channel_order="XYZ"
            )
        )

    def test_evsdm(self):
        # 1x1 inversion
        i1 = self.n1.evsdm()
        self.assertArrayEq(self.tgrid, i1["XX"].times)
        self.assertArrayEq(self.fgrid, i1["XX"].frequencies)
        self.assertAllClose(1 / self.wa, i1["XX"].entries)
        # 2x2 inversion
        i2 = self.n2.evsdm()
        self.assertArrayEq(self.tgrid, i2["XX"].times)
        self.assertArrayEq(self.fgrid, i2["XX"].frequencies)
        for m in range(self.Nf):
            for n in range(self.Nt):
                invevsdm = self.mat2_from_tf(self.c2, m, n)
                evsdm = self.mat2_from_tf(i2, m, n)
                # is the inverse matrix actually the inverse matrix?
                self.assertAllClose(np.eye(2), invevsdm @ evsdm)
        # 3x3 inversion
        i3 = self.n3.evsdm()
        self.assertArrayEq(self.tgrid, i3["XX"].times)
        self.assertArrayEq(self.fgrid, i3["XX"].frequencies)
        for m in range(self.Nf):
            for n in range(self.Nt):
                invevsdm = self.mat3_from_tf(self.c3, m, n)
                evsdm = self.mat3_from_tf(i3, m, n)
                self.assertAllClose(np.eye(3), invevsdm @ evsdm)

    def test_inner(self):
        def f(n1, d0, d1, d2):
            # Inner product mathematical properties: symmetric, positive-definite, bilinear
            self.assertAlmostEqual(n1.inner(d1, d2), n1.inner(d2, d1))
            self.assertGreaterEqual(n1.inner(d0, d0), 0)
            self.assertGreaterEqual(n1.inner(d1, d1), 0)
            self.assertGreaterEqual(n1.inner(d2, d2), 0)
            self.assertAlmostEqual(n1.inner(d1, 2.5 * d2), 2.5 * n1.inner(d1, d2))
            self.assertAlmostEqual(
                n1.inner(d1, d1 + d2), n1.inner(d1, d1) + n1.inner(d1, d2)
            )

        f(self.n1, self.d1_0, self.d1_1, self.d1_2)
        f(self.n2, self.d2_0, self.d2_1, self.d2_2)
        f(self.n3, self.d3_0, self.d3_1, self.d3_2)

        # when the data have only one nonzero channel, the result depends only on that channel
        self.assertAlmostEqual(
            self.n1.inner(self.d1_1, self.d1_1), self.n2.inner(self.d2_0, self.d2_0)
        )
        self.assertAlmostEqual(
            self.n1.inner(self.d1_1, self.d1_1), self.n3.inner(self.d3_0, self.d3_0)
        )

        def euclidean2x2(left, right):
            return np.sum((left["X"] * right["X"] + left["Y"] * right["Y"]).entries)

        # normalization: the identity invevsdm leads to the euclidean dot product
        # (NOTE this normalization may change in the future)
        self.assertAlmostEqual(
            self.n_id.inner(self.d2_1, self.d2_2), euclidean2x2(self.d2_1, self.d2_2)
        )

        ###### test error behavior
        # wrong number of TDI channels in one of the operands
        with self.assertRaises(ValueError):
            self.n2.inner(self.d3_1, self.d2_1)

    def test_whitening_matrix(self):
        # 1x1 -- just a square root
        invevsdm = WDMData({"00": self.wa})
        expected_sdm = WDMData({"00": np.sqrt(self.wa)})
        n = WDMWhittleNoise(invevsdm)
        whitening_mat = n.get_whitening_matrix(kind="cholesky")
        self.assertDataAllClose(expected_sdm, whitening_mat)
        del invevsdm, expected_sdm, n, whitening_mat

        # 2x2 -- test that W^T W = C
        def wTw_2x2(w: WDMData):
            # output w^T w
            # [ w00  0  ] [ w00 w01 ]   [  w00^2     w00*w01   ]
            # [ w01 w11 ] [  0  w11 ] = [ w01*w00  w01^2+w11^2 ]
            w00, w01, w11 = w["00"], w["01"], w["11"]
            return WDMData(
                {"00": w00 * w00, "01": w00 * w01, "11": w01 * w01 + w11 * w11}
            )

        invevsdm = WDMData({"00": self.wa, "01": self.wb, "11": self.wd})
        n = WDMWhittleNoise(invevsdm)
        whitening_mat = n.get_whitening_matrix(kind="cholesky")
        product = wTw_2x2(whitening_mat)
        self.assertDataAllClose(invevsdm, product)
        del invevsdm, n, whitening_mat, product

        # 3x3 -- same idea
        def wTw_3x3(w: WDMData):
            # [ w00  0   0  ] [ w00 w01 w02 ]   [ w00^2    w00*w01         w00*w02      ]
            # [ w01 w11  0  ] [  0  w11 w12 ] = [ (sym)  w01^2+w11^2   w01*w02+w11*w12  ]
            # [ w02 w12 w22 ] [  0   0  w22 ]   [ (sym)      (sym)    w02^2+w12^2+w22^2 ]
            w00, w01, w11 = w["00"], w["01"], w["11"]
            w02, w12, w22 = w["02"], w["12"], w["22"]
            return WDMData(
                {
                    "00": w00 * w00,
                    "01": w00 * w01,
                    "02": w00 * w02,
                    "11": w01 * w01 + w11 * w11,
                    "12": w01 * w02 + w11 * w12,
                    "22": w02 * w02 + w12 * w12 + w22 * w22,
                }
            )

        invevsdm = WDMData(
            {
                "00": self.wa,
                "01": self.wb,
                "02": self.wc,
                "11": self.wd,
                "12": self.we,
                "22": self.wf,
            }
        )
        n = WDMWhittleNoise(invevsdm)
        whitening_mat = n.get_whitening_matrix(kind="cholesky")
        product = wTw_3x3(whitening_mat)
        self.assertDataAllClose(invevsdm, product)
        del invevsdm, n, whitening_mat, product

    # TODO test auxiliary methods


if __name__ == "__main__":
    unittest.main()
