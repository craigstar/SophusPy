import numpy as np
import unittest
import pytest
import copy

import sophuspy as sp

class TestSO3(unittest.TestCase):
    def setUp(self):
        self.Rnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787],
                             [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596],
                             [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105]])

    def test_default_constructor(self):
        R = sp.SO3()
        self.assertTrue(np.allclose(R.matrix(), np.eye(3)))

    def test_copy_constructor(self):
        R1 = sp.SO3(self.Rnp)
        R2 = sp.SO3(R1)
        self.assertTrue(np.allclose(R1.matrix(), R2.matrix()))

    def test_numpy_constructor(self):
        R = sp.SO3(self.Rnp)
        self.assertTrue(np.allclose(R.matrix(), self.Rnp))

    def test_str_representation(self):
        R = sp.SO3()
        answer = 'SO3([[1, 0, 0],\n     [0, 1, 0],\n     [0, 0, 1]])'
        self.assertEqual(str(R), answer)

    def test_inverse(self):
        R = sp.SO3(self.Rnp)
        R_inv = R.inverse()
        self.assertTrue(np.allclose(R_inv.matrix(), self.Rnp.T))

    def test_log(self):
        R1 = sp.SO3()
        R2 = sp.SO3(self.Rnp)
        self.assertTrue(np.allclose(R1.log().shape, (3,)))
        self.assertTrue(np.allclose(R1.log(), np.zeros(3)))
        self.assertTrue(np.allclose(R2.log(), np.array([0.06646925, 1.59563459, 0.04069513])))

    def test_mul_SO3(self):
        R1 = sp.SO3(self.Rnp)
        R2 = sp.SO3()
        R12 = R1 * R2
        self.assertTrue(np.allclose(R12.matrix(), self.Rnp))

    def test_mul_point(self):
        R = sp.SO3()
        pt = R * np.ones(3)
        self.assertTrue(np.allclose(pt, np.ones(3)))

    def test_mul_points(self):
        R = sp.SO3()
        pts = R * np.ones((4, 3))
        self.assertTrue(np.allclose(pts, np.ones((4, 3))))

    def test_imul_SO3(self):
        R1 = sp.SO3()
        R2 = sp.SO3(self.Rnp)
        R1 *= R2
        self.assertTrue(np.allclose(R1.matrix(), self.Rnp))

    def test_copy_method(self):
        R = sp.SO3(self.Rnp)
        R_copied = R.copy()
        self.assertTrue(np.allclose(R.matrix(), R_copied.matrix()))

    def test_copy_lib(self):
        R = sp.SO3(self.Rnp)
        R1 = copy.copy(R)
        R2 = copy.deepcopy(R)
        self.assertTrue(np.allclose(R.matrix(), R1.matrix()))
        self.assertTrue(np.allclose(R.matrix(), R2.matrix()))

    def test_static_hat(self):
        v1 = np.ones(3)
        v2 = np.ones((3, 1))
        skew_v = np.array([[ 0, -1,  1],
                           [ 1,  0, -1],
                           [-1,  1,  0]], dtype=np.float64)
        self.assertTrue(np.allclose(sp.SO3.hat(v1), skew_v))
        self.assertTrue(np.allclose(sp.SO3.hat(v2), skew_v))

    def test_static_exp(self):
        R = sp.SO3(self.Rnp)
        R_prime = sp.SO3.exp(R.log())
        self.assertTrue(np.allclose(R.matrix(), R_prime.matrix()))
        
    def test_data_type_compatibility(self):
        R1 = sp.SO3(np.eye(3, dtype=np.float32))
        R2 = sp.SO3(np.eye(3, dtype=int))
        self.assertTrue(R1, np.eye(3))
        self.assertTrue(R2, np.eye(3))

    def test_data_size_compatibility(self):
        R1 = sp.SO3.hat(np.ones(3))
        R2 = sp.SO3.hat(np.ones((3, 1)))
        self.assertTrue(np.allclose(R1, R2))

    def test_size_fault(self):
        with pytest.raises(TypeError) as e:
            sp.SO3(np.eye(4))
        self.assertTrue('incompatible constructor arguments' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.SO3(np.eye(3).flatten())
        self.assertTrue('incompatible constructor arguments' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.SO3.exp(np.ones(((1, 3))))
        self.assertTrue('incompatible function arguments' in str(e.value))

    # TODO: find a way to test below.
    # def test_R_is_not_orthogonal_fault(self):
    #     with self.assertRaises(SystemExit) as e:
    #         R = np.eye(3)
    #         R[0, 1] = 1e-9
    #         sp.SO3(R)
