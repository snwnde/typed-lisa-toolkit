# import unittest
# from typed_lisa_toolkit.containers.modes import Harmonic, QNM

# class TestModes(unittest.TestCase):

#     def test_harmonic_mode(self):
#         harmonic = Harmonic(l=2, m=1)
#         self.assertEqual(harmonic.degree, 2)
#         self.assertEqual(harmonic.order, 1)

#     def test_qnm_mode(self):
#         qnm = QNM(l=2, m=1, n=0)
#         self.assertEqual(qnm.degree, 2)
#         self.assertEqual(qnm.order, 1)
#         self.assertEqual(qnm.overtone, 0)

#     def test_harmonic_cast(self):
#         harmonic = Harmonic.cast((2, 1))
#         self.assertEqual(harmonic.degree, 2)
#         self.assertEqual(harmonic.order, 1)

#     def test_qnm_cast(self):
#         qnm = QNM.cast((2, 1, 0))
#         self.assertEqual(qnm.degree, 2)
#         self.assertEqual(qnm.order, 1)
#         self.assertEqual(qnm.overtone, 0)


# if __name__ == '__main__':
#     unittest.main()