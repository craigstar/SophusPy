import numpy as np
import sophus as sp

import unittest

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

	def test_str(self):
		R = sp.SO3(self.Rnp)
		self.assertEqual(R.__str__(), np.array_str(self.Rnp))

	def test_inverse(self):
		R = sp.SO3(self.Rnp)
		R_inv = R.inverse()
		self.assertTrue(np.allclose(R_inv.matrix(), self.Rnp.T))