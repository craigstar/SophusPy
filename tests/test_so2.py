import numpy as np
import unittest
import pytest
import copy

import sophuspy as sp

class TestSO2(unittest.TestCase):
    def setUp(self):
        self.Rnp = np.array([[0.0, -1.0],
                             [1.0,  0.0]])

    def test_default_constructor(self):
        R = sp.SO2()
        self.assertTrue(np.allclose(R.matrix(), np.eye(2)))

    def test_copy_constructor(self):
        R1 = sp.SO2(self.Rnp)
        R2 = sp.SO2(R1)
        self.assertTrue(np.allclose(R1.matrix(), R2.matrix()))

    def test_numpy_constructor(self):
        R = sp.SO2(self.Rnp)
        self.assertTrue(np.allclose(R.matrix(), self.Rnp))

    def test_str_representation(self):
        R = sp.SO2()
        answer = 'SO2([[ 1, -0],\n     [ 0,  1]])'
        self.assertEqual(str(R), answer)

    def test_inverse(self):
        R = sp.SO2(self.Rnp)
        R_inv = R.inverse()
        self.assertTrue(np.allclose(R_inv.matrix(), self.Rnp.T))

    def test_log(self):
        R1 = sp.SO2()
        R2 = sp.SO2(self.Rnp)
        self.assertTrue(type(R1.log()) is float)
        self.assertTrue(np.allclose(R1.log(), np.zeros(1)))
        self.assertTrue(np.allclose(R2.log(), np.array(1.5707963267948966)))

    def test_mul_SO2(self):
        R1 = sp.SO2(self.Rnp)
        R2 = sp.SO2()
        R12 = R1 * R2
        self.assertTrue(np.allclose(R12.matrix(), self.Rnp))

    def test_mul_point(self):
        R = sp.SO2()
        pt = R * np.ones(2)
        self.assertTrue(np.allclose(pt, np.ones(2)))

    def test_mul_points(self):
        R = sp.SO2()
        pts = R * np.ones((4, 2))
        self.assertTrue(np.allclose(pts, np.ones((4, 2))))

    def test_imul_SO2(self):
        R1 = sp.SO2()
        R2 = sp.SO2(self.Rnp)
        R1 *= R2
        self.assertTrue(np.allclose(R1.matrix(), self.Rnp))

    def test_copy_method(self):
        R = sp.SO2(self.Rnp)
        R_copied = R.copy()
        self.assertTrue(np.allclose(R.matrix(), R_copied.matrix()))

    def test_copy_lib(self):
        R = sp.SO2(self.Rnp)
        R1 = copy.copy(R)
        R2 = copy.deepcopy(R)
        self.assertTrue(np.allclose(R.matrix(), R1.matrix()))
        self.assertTrue(np.allclose(R.matrix(), R2.matrix()))

    def test_static_hat(self):
        v1 = np.ones(1)
        v2 = np.ones((1, 1))
        skew_v = np.array([[ 0, -1],
                           [ 1,  0]], dtype=np.float64)
        self.assertTrue(np.allclose(sp.SO2.hat(v1), skew_v))
        self.assertTrue(np.allclose(sp.SO2.hat(v2), skew_v))

    def test_static_exp(self):
        R = sp.SO2(self.Rnp)
        R_prime = sp.SO2.exp(R.log())
        self.assertTrue(np.allclose(R.matrix(), R_prime.matrix()))

    def test_data_type_compatibility(self):
        R1 = sp.SO2(np.eye(2, dtype=np.float32))
        R2 = sp.SO2(np.eye(2, dtype=int))
        self.assertTrue(R1, np.eye(2))
        self.assertTrue(R2, np.eye(2))

    def test_data_size_compatibility(self):
        R1 = sp.SO2.hat(np.ones(1))
        R2 = sp.SO2.hat(np.ones((1, 1)))
        self.assertTrue(np.allclose(R1, R2))

    def test_size_fault(self):
        with pytest.raises(TypeError) as e:
            sp.SO2(np.eye(4))
        self.assertTrue('incompatible constructor arguments' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.SO2(np.eye(2).flatten())
        self.assertTrue('incompatible constructor arguments' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.SO2.exp(np.ones(((1, 2))))
        self.assertTrue('incompatible function arguments' in str(e.value))
