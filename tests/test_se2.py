import numpy as np
import unittest
import pytest
import copy

import sophuspy as sp


class TestSE2(unittest.TestCase):
    def setUp(self):
        self.Rnp = np.array([[0.0, -1.0],
                             [1.0,  0.0]])

        self.Tnp = np.array([[0.0, -1.0, 100.0],
                             [1.0,  0.0, 200.0],
                             [0.0,  0.0, 1.0]])
        self.t = np.array([100.0, 200.0])

    def test_default_constructor(self):
        T = sp.SE2()
        self.assertTrue(np.allclose(T.matrix(), np.eye(3)))

    def test_copy_constructor(self):
        T1 = sp.SE2(self.Tnp)
        T2 = sp.SE2(T1)
        self.assertTrue(np.allclose(T1.matrix(), T2.matrix()))

    def test_numpy_constructor(self):
        T = sp.SE2(self.Tnp)
        self.assertTrue(np.allclose(T.matrix(), self.Tnp))

    def test_rt_constructor(self):
        T1 = sp.SE2(self.Rnp, self.t)
        self.assertTrue(np.allclose(T1.matrix(), self.Tnp))

    def test_str(self):
        T = sp.SE2()
        answer = 'SE2([[ 1, -0,  0],\n     [ 0,  1,  0],\n     [ 0,  0,  1]])'
        self.assertEqual(str(T), answer)

    def test_mul_SE2(self):
        T1 = sp.SE2(self.Tnp)
        T2 = sp.SE2()
        T12 = T1 * T2
        self.assertTrue(np.allclose(T12.matrix(), self.Tnp))

    def test_mul_point(self):
        T = sp.SE2()
        pt = T * np.ones(2)
        self.assertTrue(np.allclose(pt, np.ones(2)))

    def test_mul_points(self):
        T = sp.SE2()
        pt = T * np.ones((4, 2))
        self.assertTrue(np.allclose(pt, np.ones((4, 2))))

    def test_imul_SE2(self):
        T1 = sp.SE2()
        T2 = sp.SE2(self.Tnp)
        T1 *= T2
        self.assertTrue(np.allclose(T1.matrix(), self.Tnp))

    def test_copy_method(self):
        T1 = sp.SE2(self.Tnp)
        T2 = T1.copy()
        self.assertTrue(np.allclose(T1.matrix(), T2.matrix()))

    def test_copy_lib(self):
        T = sp.SE2(self.Tnp)
        T1 = copy.copy(T)
        T2 = copy.deepcopy(T)
        self.assertTrue(np.allclose(T.matrix(), T1.matrix()))
        self.assertTrue(np.allclose(T.matrix(), T2.matrix()))

    def test_so2(self):
        T = sp.SE2(self.Tnp)
        self.assertTrue(np.allclose(T.so2().matrix(), self.Rnp))

    def test_inverse(self):
        T = sp.SE2(self.Tnp)
        T_inv = T.inverse()

        Tnp_inv = np.eye(3)
        Rnp_inv = self.Tnp[:2, :2].T
        tnp = self.Tnp[:2, 2]

        Tnp_inv[:2, :2] = Rnp_inv
        Tnp_inv[:2, 2] = -Rnp_inv.dot(tnp)

        self.assertTrue(np.allclose(T_inv.matrix(), Tnp_inv))

    def test_translation(self):
        T = sp.SE2(self.Tnp)
        self.assertTrue(np.allclose(T.translation(), self.Tnp[:2, 2]))

    def test_rotationMatrix(self):
        T = sp.SE2(self.Tnp)
        self.assertTrue(np.allclose(T.rotationMatrix(), self.Tnp[:2, :2]))

    def test_matrix2x3(self):
        T = sp.SE2(self.Tnp)
        self.assertTrue(np.allclose(T.matrix2x3(), self.Tnp[:2]))

    def test_setRotationMatrix(self):
        T = sp.SE2()
        T.setRotationMatrix(self.Rnp)
        self.Tnp[:2, 2] = np.zeros(2)
        self.assertTrue(np.allclose(T.matrix(), self.Tnp))

    def test_setTranslation(self):
        T = sp.SE2()
        T.setTranslation(np.ones(2))
        Tprime = np.eye(3)
        Tprime[:2, 2] = np.ones(2)
        self.assertTrue(np.allclose(T.matrix(), Tprime))

    def test_static_hat(self):
        v = np.ones(3)
        hat = np.array([[ 0, -1,  1],
                        [ 1,  0,  1],
                        [ 0,  0,  0]], dtype=np.float64)
        self.assertTrue(np.allclose(sp.SE2.hat(v), hat))

    def test_exp(self):
        T = sp.SE2(self.Tnp)
        T_prime = sp.SE2.exp(T.log())
        self.assertTrue(np.allclose(T.matrix(), T_prime.matrix()))

    def test_data_type_compatibility(self):
        T1 = sp.SE2(np.eye(3, dtype=np.float32))
        T2 = sp.SE2(np.eye(3, dtype=int))
        self.assertTrue(T1, np.eye(3))
        self.assertTrue(T2, np.eye(3))

    def test_data_size_compatibility(self):
        T1 = sp.SE2.hat(np.ones(3))
        T2 = sp.SE2.hat(np.ones((3, 1)))
        self.assertTrue(np.allclose(T1, T2))

    def test_size_fault(self):
        with pytest.raises(TypeError) as e:
            sp.SE2(np.eye(4))
        self.assertTrue('incompatible constructor arguments' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.SE2(np.eye(3).flatten())
        self.assertTrue('incompatible constructor arguments' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.SE2.hat(np.ones(((1, 3))))
        self.assertTrue('incompatible function arguments' in str(e.value))
