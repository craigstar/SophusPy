import numpy as np
from eigency.core cimport *

ctypedef SO3[double] SO3d # double precision SO3 

cdef class pySO3:
    cdef SO3d *thisptr

    def __cinit__(self, other=None):
        self.thisptr = new SO3d()

    def __dealloc__(self):
        del self.thisptr
    
    def __str__(self):
        return np.array_str(self.matrix())

    def matrix(self):
        return ndarray(self.thisptr.matrix())
