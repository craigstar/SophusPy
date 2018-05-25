import numpy as np
from eigency.core cimport *
from cython.operator cimport dereference as deref

from sophus_def cimport SO3 as _SO3
from sophus_def cimport SE3 as _SE3

ctypedef _SO3[double] _SO3d # double precision SO3
ctypedef _SE3[double] _SE3d # double precision SE3

cdef class SO3:
    cdef _SO3d *thisptr

    def __cinit__(self, other=None):
        cdef SO3 ostr
        if other is not None and type(other) is SO3:
            # Copy constructor
            ostr = <SO3> other
            self.thisptr = new _SO3d(deref(ostr.thisptr))
        elif other is not None and type(other) is np.ndarray:
             # Eigency expects 'F_CONTIGUOUS' layout, convert if this is not the case
            if other.flags['C_CONTIGUOUS']:
                other = other.copy(order='F')
            self.thisptr = new _SO3d(Map[Matrix3d](other))
        else:
            # default to 3*3 identity matrix
            self.thisptr = new _SO3d()

    def __dealloc__(self):
        del self.thisptr
    
    def __str__(self):
        return np.array_str(self.matrix())

    def matrix(self):
        return ndarray(self.thisptr.matrix())

    def log(self):
        return ndarray(self.thisptr.log())

    def inverse(self):
        so3 = SO3()
        so3.thisptr = new _SO3d(self.thisptr.inverse())
        return so3

cdef class SE3:
    cdef _SE3d *thisptr

    def __cinit__(self, other=None, t=None):
        cdef SE3 ostr
        if t is not None and type(t) is np.ndarray and type(other) is np.ndarray:
            # compose R and t into T (4*4) TODO: use cpp constructor
            T = np.eye(4)
            T[:3, :3] = other
            T[:3, 3] = t.ravel()
            other = T

        if other is not None and type(other) is SE3:
            # Copy constructor
            ostr = <SE3> other
            self.thisptr = new _SE3d(deref(ostr.thisptr))
        elif other is not None and type(other) is np.ndarray:
            if other.flags['C_CONTIGUOUS']:
                other = other.copy(order='F')
            self.thisptr = new _SE3d(Map[Matrix4d](other))
        else:
            # default to 4*4 identity matrix
            self.thisptr = new _SE3d()

    def __dealloc__(self):
        del self.thisptr

    def __str__(self):
        return np.array_str(self.matrix())

    def __mul__(SE3 x, SE3 y):
        """
        Group multiplication operator
        """
        res = SE3()
        res.thisptr[0] = x.thisptr.mul(deref(y.thisptr))
        return res

    def matrix(self):
        return ndarray(self.thisptr.matrix())

    def inverse(self):
        se3 = SE3()
        se3.thisptr = new _SE3d(self.thisptr.inverse())
        return se3

    def translation(self):
        return ndarray(self.thisptr.translation()).ravel()

    def rotationMatrix(self):
        return ndarray(self.thisptr.rotationMatrix())

    def setRotationMatrix(self, R):
        return self.thisptr.setRotationMatrix(Map[Matrix3d](R))
    
