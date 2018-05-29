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
    if arr.dtype != DTYPE:
        raise "arr has to be float64!!!"

def __tofortran(np.ndarray arr):
    """Eigency expects 'F_CONTIGUOUS' layout, convert if this is not the case"""
    if not arr.flags['C_CONTIGUOUS']:
        return arr
    return np.asfortranarray(arr)


cdef class SO3:
    """Class of SO3"""
    cdef _SO3d *thisptr         # pointer of C++ SO3<double> instance

    def __cinit__(self, other=None):
        """
        Constructor, accept 3 types of input
        ------------------------------------
        In: SO3 or 3*3 np.ndarray or empty
        ------------------------------------
        """
        cdef SO3 ostr           # define an instance of SO3

        if other is not None and type(other) is SO3:
            # Copy constructor
            ostr = <SO3> other
            self.thisptr = new _SO3d(deref(ostr.thisptr))
        elif other is not None and type(other) is np.ndarray:
            # 3*3 np.array constructor
            __checkfloat64(other)
            other = __tofortran(other)
            self.thisptr = new _SO3d(Map[Matrix3d](other))        
        else:
            # default to 3*3 identity matrix
            self.thisptr = new _SO3d()

    def __dealloc__(self):
        """deconstructor"""
        del self.thisptr
    
    def __str__(self):
        """
        string representation of SO3
        ----------------------------
        Out: str
        ----------------------------
        """
        return np.array_str(self.matrix())

    def matrix(self):
        """
        Returns and 3*3 np.ndarray matrix
        ---------------------------------
        Out: np.ndarray (3*3)
        ---------------------------------
        """
        return ndarray(self.thisptr.matrix())

    def log(self):
        """
        Lie algebra log
        --------------------
        Out: np.ndarray (3,)
        --------------------
        """
        return ndarray(self.thisptr.log()).ravel()

    def inverse(self):
        """
        Inverse of a 3*3 othogonal matrix
        is the transpose of it
        ---------------------------------
        Out: SO3
        ---------------------------------
        """
        so3 = SO3()
        del so3.thisptr         # clear pointer before assignment 
        so3.thisptr = new _SO3d(self.thisptr.inverse())
        return so3


cdef class SE3:
    """Class of SE3"""
    cdef _SE3d *thisptr         # pointer of C++ SE3<double> instance

    def __cinit__(self, other=None, np.ndarray[DTYPE_t, ndim=1] t=None):
        """
        Constructor, accept 4 types of input
        ------------------------------------
        In: SE3 or [(3*3) R and (3,) t] or
            (4*4) np.ndarray or empty
        ------------------------------------
        """
        cdef SE3 ostr           # define an instance of SE3
        if t is not None and type(other) is np.ndarray:
            # compose R (3*3) and t (3,) into T (4*4)
            so3 = SO3(other)
            self.thisptr = new _SE3d(deref(so3.thisptr), Map[Vector3d](t))
        elif other is not None and type(other) is np.ndarray:
            # 4*4 np.array constructor
            __checkfloat64(other)
            other = __tofortran(other)
            self.thisptr = new _SE3d(Map[Matrix4d](other))
        elif other is not None and type(other) is SE3:
            # Copy constructor
            ostr = <SE3> other
            self.thisptr = new _SE3d(deref(ostr.thisptr))
        else:
            # default to 4*4 identity matrix
            self.thisptr = new _SE3d()

    def __dealloc__(self):
        """deconstructor"""
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
        SE3 * SE3 or SE3 * point (3, np.ndarray)
        sor SE3 * points (N*3 np.ndarray)
        return None means input type Error
        -----------------------------------------
        In: SE3, SE3 or np.ndarray (3,) or (N, 3)
        Out: SE3 or (3,) np.ndarray
        -----------------------------------------
        """
        cdef SE3 ostr           # define an instance of SE3

        if type(other) is np.ndarray:
            __checkfloat64(other)
            if other.size == 3:
                # SE3 * [x, y, z]
                __checkfloat64(other)
                return ndarray(x.thisptr.mul(Map[Vector3d](other))).ravel()
            else:
                # SE3 * [[x1, y1, z1], ..., [xn, yn, zn]] (N*3) np.array
                other = np.hstack((other, np.ones((len(other), 1)))).T
                return x.matrix3x4().dot(other).T
        elif type(other) is SE3:
            # SE3 * SE3
            ostr = <SE3> other
            res = SE3()
            res.thisptr[0] = x.thisptr.mul(deref(ostr.thisptr))
            return res
        else:
            raise "Input is not the excepted type or dimension"

    def __imul__(self, SE3 y):
        """
        SE3 * SE3 inplace multiplication
        -----------------------------------------
        In: SE3
        Out: SE3
        -----------------------------------------
        """
        self.thisptr[0] = self.thisptr.mul(deref(y.thisptr))
        return self

    def matrix(self):
        """
        Returns and 4*4 np.ndarray matrix
        ---------------------------------
        Out: np.ndarray (4*4)
        ---------------------------------
        """
        return ndarray(self.thisptr.matrix())

    def matrix3x4(self):
        """
        Returns and 3*4 np.ndarray matrix
        ---------------------------------
        Out: np.ndarray (3*4)
        ---------------------------------
        """
        return ndarray(self.thisptr.matrix3x4())

    def inverse(self):
        """
        Inverse of a 4*4 matrix
        ---------------------------------
        Out: SE3
        ---------------------------------
        """
        se3 = SE3()
        del se3.thisptr
        se3.thisptr = new _SE3d(self.thisptr.inverse())
        return se3

    def log(self):
        """
        Lie algebra log
        --------------------
        Out: np.ndarray (6,)
        --------------------
        """
        return ndarray(self.thisptr.log()).ravel()

    def translation(self):
        """
        translation of SE3
        --------------------
        Out: np.ndarray (3,)
        --------------------
        """
        return ndarray(self.thisptr.translation()).ravel()

    def rotationMatrix(self):
        """
        Rotation matrix of SE3
        -----------------------
        Out: np.ndarray (3 * 3)
        -----------------------
        """
        return ndarray(self.thisptr.rotationMatrix())

    def setRotationMatrix(self, np.ndarray[DTYPE_t, ndim=2] R):
        """
        Set rotation matrix of SE3
        --------------------------
        In: np.ndarray (3 * 3)
        --------------------------
        """
        R = __tofortran(R)
        self.thisptr.setRotationMatrix(Map[Matrix3d](R))

    def setTranslation(self, np.ndarray[DTYPE_t, ndim=1] t):
        """
        Set translation matrix of SE3
        -----------------------------
        In: np.ndarray (3,)
        -----------------------------
        """
        so3 = SO3(self.rotationMatrix())
        del self.thisptr
        self.thisptr = new _SE3d(deref(so3.thisptr), Map[Vector3d](t))

    @staticmethod
    def exp(np.ndarray[DTYPE_t, ndim=1] arr):
        """
        Lie algebra exp
        --------------------
        In: np.ndarray (6,)
        Out: SE3
        --------------------
        """
        res = SE3()
        del res.thisptr
        res.thisptr = new _SE3d(_SE3d.exp(Map[VectorXd](arr)))
        return res
