import numpy as np
import unittest
import pytest

import sophus as sp


class TestRoot(unittest.TestCase):
    def setUp(self):
        self.Rnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787],
                             [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596],
                             [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105]])

        self.Tnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787, -1103.428193030075],
                             [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596, -35.93298768047541],
                             [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105,  798.7780129474496],
                             [                   0,                   0,                    0,                  1]])

    def test_copy_so3_inplace(self):
        R1, R2 = sp.SO3(), sp.SO3(self.Rnp)
        id_old = id(R1)
        sp.copyto(R1, R2)
        id_new = id(R1)
        self.assertTrue(np.allclose(R1.matrix(), R2.matrix()))
        self.assertEqual(id_old, id_new)

    def test_copy_se3_inplace(self):
        T1, T2 = sp.SE3(), sp.SE3(self.Tnp)
        id_old = id(T1)
        sp.copyto(T1, T2)
        id_new = id(T1)
        self.assertTrue(np.allclose(T1.matrix(), T2.matrix()))
        self.assertEqual(id_old, id_new)

    def test_copy_inplace_failure(self):
        R, T = sp.SO3(), sp.SE3()
        with pytest.raises(TypeError) as e:
            sp.copyto(R, T)
        self.assertTrue('SO3-SO3 or SE3-SE3 pair' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.copyto(T, R)
        self.assertTrue('SO3-SO3 or SE3-SE3 pair' in str(e.value))