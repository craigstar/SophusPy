import numpy as np
import unittest

import sophus as sp


class TestSE3(unittest.TestCase):
    def setUp(self):
        self.Rnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787],
                             [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596],
                             [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105]])

        self.Tnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787, -1103.428193030075],
                             [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596, -35.93298768047541],
                             [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105,  798.7780129474496],
                             [                   0,                   0,                    0,                  1]])
        self.t = np.array([-1103.428193030075, -35.93298768047541, 798.7780129474496])

    def test_default_constructor(self):
        T = sp.SE3()
        self.assertTrue(np.allclose(T.matrix(), np.eye(4)))

    def test_copy_constructor(self):
        T1 = sp.SE3(self.Tnp)
        T2 = sp.SE3(T1)
        self.assertTrue(np.allclose(T1.matrix(), T2.matrix()))

    def test_numpy_constructor(self):
        T = sp.SE3(self.Tnp)
        self.assertTrue(np.allclose(T.matrix(), self.Tnp))

    def test_rt_constructor(self):
        T1 = sp.SE3(self.Rnp, self.t)
        self.assertTrue(np.allclose(T1.matrix(), self.Tnp))

        # test input can be any data numpy data type
        T2 = sp.SE3(np.eye(3, dtype=int), np.arange(3))
        T2_groundtruth = np.eye(4, dtype=np.float64)
        T2_groundtruth[:3, 3] = np.arange(3)
        self.assertTrue(np.allclose(T2.matrix(), T2_groundtruth))

    def test_str(self):
        T = sp.SE3(self.Tnp)
        self.assertEqual(T.__str__(), np.array_str(self.Tnp))

    def test_inverse(self):
        T = sp.SE3(self.Tnp)
        T_inv = T.inverse()

        Tnp_inv = np.eye(4)
        Rnp_inv = self.Tnp[:3, :3].T
        tnp = self.Tnp[:3, 3]

        Tnp_inv[:3, :3] = Rnp_inv
        Tnp_inv[:3, 3] = -Rnp_inv.dot(tnp)

        self.assertTrue(np.allclose(T_inv.matrix(), Tnp_inv))

    def test_translation(self):
        T = sp.SE3(self.Tnp)
        self.assertTrue(np.allclose(T.translation(), self.Tnp[:3, 3]))

    def test_rotationMatrix(self):
        T = sp.SE3(self.Tnp)
        self.assertTrue(np.allclose(T.rotationMatrix(), self.Tnp[:3, :3]))

    def test_setRotationMatrix(self):
        T = sp.SE3()
        T.setRotationMatrix(self.Rnp)
        self.Tnp[:3, 3] = np.zeros(3)
        self.assertTrue(np.allclose(T.matrix(), self.Tnp))

        # test input can be any data numpy data type
        T = sp.SE3(self.Tnp)
        R = np.eye(3, dtype=np.float32)
        T.setRotationMatrix(R)
        self.Tnp[:3, :3] = np.eye(3)
        self.assertTrue(np.allclose(T.matrix(), self.Tnp))

    def test_mul_SE3(self):
        T1 = sp.SE3(self.Tnp)
        T2 = sp.SE3()
        T12 = T1 * T2
        self.assertTrue(np.allclose(T12.matrix(), self.Tnp))

    def test_mul_point(self):
        T = sp.SE3()
        pt = T * np.ones(3)
        self.assertTrue(np.allclose(pt, np.ones(3)))

    def test_imul_SE3(self):
        T1 = sp.SE3(self.Tnp)
        T2 = sp.SE3()
        T1 *= T2
        self.assertTrue(np.allclose(T1.matrix(), self.Tnp))

    def test_exp(self):
        T = sp.SE3(self.Tnp)
        Tprime = sp.SE3.exp(T.log())
        self.assertTrue(np.allclose(T.matrix(), Tprime.matrix()))

    def test_setTranslation(self):
        T = sp.SE3()
        T.setTranslation(np.ones(3))
        Tprime = np.eye(4)
        Tprime[:3, 3] = np.ones(3)
        self.assertTrue(np.allclose(T.matrix(), Tprime))

        # test input can be any data numpy data type
        T = sp.SE3()
        T.setTranslation(np.ones(3, dtype=int))
        Tprime = np.eye(4)
        Tprime[:3, 3] = np.ones(3)
        self.assertTrue(np.allclose(T.matrix(), Tprime))

    
