import numpy as np
import sophus as sp

# print(sp.__file__)

a = sp.pySO3()

m = a.matrix()
print(a)