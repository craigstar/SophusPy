import numpy as np
from eigency.core cimport *
from cython.operator cimport dereference as deref

from sophus_def cimport SO3 as _SO3
from sophus_def cimport SE3 as _SE3

DTYPE = np.float64

ctypedef _SO3[double] _SO3d # double precision SO3
ctypedef _SE3[double] _SE3d # double precision SE3
ctypedef np.float64_t DTYPE_t

# helper functions
def __checkfloat64(np.ndarray arr):
    """make sure arr is float64, which corresponds to <double>"""
    if arr.dtype != np.float64:
        raise "arr has to be float64!!!"

def __tofortran(np.ndarray arr):
    """Eigency expects 'F_CONTIGUOUS' layout, convert if this is not the case"""
    if not arr.flags['C_CONTIGUOUS']:
        return arr
    return np.asfortranarray(arr)

cdef class SO3:
    cdef _SO3d *thisptr

    def __cinit__(self, other=None):
        cdef SO3 ostr
        if other is not None and type(other) is SO3:
            # Copy constructor
            ostr = <SO3> other
            self.thisptr = new _SO3d(deref(ostr.thisptr))
        elif other is not None and type(other) is np.ndarray:
            __checkfloat64(other)
            other = __tofortran(other)
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
        del so3.thisptr
        so3.thisptr = new _SO3d(self.thisptr.inverse())
        return so3

cdef class SE3:
    cdef _SE3d *thisptr

    def __cinit__(self, other=None, np.ndarray[DTYPE_t, ndim=1] t=None):
        cdef SE3 ostr
        if t is not None and type(other) is np.ndarray:
            # compose R and t into T (4*4)
            so3 = SO3(other)
            self.thisptr = new _SE3d(deref(so3.thisptr), Map[Vector3d](t))
        elif other is not None and type(other) is SE3:
            # Copy constructor
            ostr = <SE3> other
            self.thisptr = new _SE3d(deref(ostr.thisptr))
        elif other is not None and type(other) is np.ndarray:
            other = __tofortran(other)
            self.thisptr = new _SE3d(Map[Matrix4d](other))
        else:
            # default to 4*4 identity matrix
            self.thisptr = new _SE3d()

    def __dealloc__(self):
        del self.thisptr

    def __str__(self):
        """
        string representation of SE3
        ----------------------------
        Out: str
        ----------------------------
        """
        return np.array_str(self.matrix())

    def __mul__(SE3 x, other):
        """
        SE3 * SE3 or SE3 * point (3*N np.ndarray)
        return None means input type Error
        ------------------------------
        In: SE3, SE3 or np.ndarray (3,) or (N, 3)
        Out: SE3 or (3,) np.ndarray
        ------------------------------
        """
        cdef SE3 ostr

        if type(other) is np.ndarray:
            if other.size == 3:
                __checkfloat64(other)
                return ndarray(x.thisptr.mul(Map[Vector3d](other))).ravel()
            else:
                pass
        elif type(other) is SE3:
            ostr = <SE3> other
            res = SE3()
            res.thisptr[0] = x.thisptr.mul(deref(ostr.thisptr))
            return res

    def __imul__(self, SE3 y):
        self.thisptr[0] = self.thisptr.mul(deref(y.thisptr))
        return self

    def matrix(self):
        return ndarray(self.thisptr.matrix())

    def inverse(self):
        se3 = SE3()
        del se3.thisptr
        se3.thisptr = new _SE3d(self.thisptr.inverse())
        return se3

    def log(self):
        return ndarray(self.thisptr.log()).ravel()

    def translation(self):
        return ndarray(self.thisptr.translation()).ravel()

    def rotationMatrix(self):
        return ndarray(self.thisptr.rotationMatrix())

    def setRotationMatrix(self, np.ndarray[DTYPE_t, ndim=2] R):
        R = __tofortran(R)
        return self.thisptr.setRotationMatrix(Map[Matrix3d](R))

    def setTranslation(self, np.ndarray[DTYPE_t, ndim=1] t):
        so3 = SO3(self.rotationMatrix())
        del self.thisptr
        self.thisptr = new _SE3d(deref(so3.thisptr), Map[Vector3d](t))

    @staticmethod
    def exp(np.ndarray arr):
        res = SE3()
        del res.thisptr
        res.thisptr = new _SE3d(_SE3d.exp(Map[VectorXd](arr)))
        return res
