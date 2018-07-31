import numpy as np
from eigency.core cimport *
from cython.operator cimport dereference as deref
from libcpp cimport bool

from sophus_def cimport SO3 as _SO3
from sophus_def cimport SE3 as _SE3
from sophus_def cimport transformPointsByPoses

DTYPE = np.float64

ctypedef _SO3[double] _SO3d # double precision SO3
ctypedef _SE3[double] _SE3d # double precision SE3
ctypedef np.float64_t DTYPE_t


# helper functions
def __checkfloat64(np.ndarray arr):
    """make sure arr is float64, which corresponds to <double>"""
    assert arr.dtype == DTYPE, "arr type should to be float64!!!"

def __tofortran(np.ndarray arr):
    """Eigency expects 'F_CONTIGUOUS' layout, convert if this is not the case"""
    if arr.flags['F_CONTIGUOUS']:
        return arr
    return np.asfortranarray(arr)

def __checksize(np.ndarray arr, int size):
    """make sure arr has expected size"""
    assert arr.size == size, ("arr size %d, expected size %d" % (arr.size, size))

def __checkcols(np.ndarray arr, int ncols):
    """make sure arr has expected number of columns"""
    assert arr.shape[1] == ncols, ("arr cols %d, expected cols %d" % (arr.shape[1], ncols))


def _copytoSO3(SO3 dst, SO3 src):
    """helper function for copying SO3 in place"""
    del dst.thisptr
    dst.thisptr = new _SO3d(deref(src.thisptr))

def _copytoSE3(SE3 dst, SE3 src):
    """helper function for copying SE3 in place"""
    del dst.thisptr
    dst.thisptr = new _SE3d(deref(src.thisptr))

def copyto(dst, src):
    """
    Copy SO3 or SE3 in place
    ----------------------------
    In: (SO3, SO3) or (SE3, SE3)
    ----------------------------
    """
    if type(src) is SO3 and type(dst) is SO3:
        _copytoSO3(dst, src)
    elif type(src) is SE3 and type(dst) is SE3:
        _copytoSE3(dst, src)
    else:
        raise TypeError("input type does not match SO3-SO3 or SE3-SE3 pair")

def transform_points_by_poses(np.ndarray[DTYPE_t, ndim=2] poses, np.ndarray[DTYPE_t, ndim=2] points,
                              bool is_inverse=False):
    """
    Transform points by stack of poses, 
    ----------------------------
    In: (SO3, SO3) or (SE3, SE3)
    ----------------------------
    """
    poses = __tofortran(poses)
    points = __tofortran(points)
    __checkcols(poses, 12)
    __checkcols(points, 3)
    return ndarray(transformPointsByPoses(Map[MatrixXd](poses), Map[MatrixXd](points), is_inverse))


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
            __checksize(other, 9)
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

    def __reduce__(self):
        """
        For deepcopy purpose
        -----------------------------
        Out: tuple
        -----------------------------
        """
        return (self.__class__, (self.matrix(),))
        
    def __copy__(self):
        """
        Return a copy of SO3
        -----------------------------
        Out: SO3
        -----------------------------
        """
        return SO3(self)

    def matrix(self):
        """
        Returns and 3*3 np.ndarray matrix
        ---------------------------------
        Out: np.ndarray (3*3)
        ---------------------------------
        """
        return ndarray(self.thisptr.matrix()).copy()

    def log(self):
        """
        Lie algebra log
        --------------------
        Out: np.ndarray (3,)
        --------------------
        """
        return ndarray(self.thisptr.log()).ravel().copy()

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

    def copy(self):
        """
        Return a copy of SO3
        -----------------------------
        Out: SO3
        -----------------------------
        """
        return self.__copy__()

    @staticmethod
    def hat(np.ndarray[DTYPE_t, ndim=1] v):
        """
        Hat of SO3 is to calculate the skew matrix
        -----------------------
        In: np.ndarray (3,)
        Out: np.ndarray (3 * 3)
        -----------------------
        """
        __checksize(v, 3)
        return ndarray(_SO3d.hat(Map[Vector3d](v))).copy()

    @staticmethod
    def exp(np.ndarray[DTYPE_t, ndim=1] v):
        """
        Computes the exponential map of a 3x1 so3 element
        ----------------------
        In: np.ndarray (3,)
        Out: SO3
        ----------------------
        """
        __checksize(v, 3)
        so3 = SO3()
        del so3.thisptr
        so3.thisptr = new _SO3d(_SO3d.exp(Map[Vector3d](v)))
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
            __checkfloat64(t)
            __checksize(t, 3)
            so3 = SO3(other)
            self.thisptr = new _SE3d(deref(so3.thisptr), Map[Vector3d](t))
        elif other is not None and type(other) is np.ndarray:
            # 4*4 np.array constructor
            __checkfloat64(other)
            __checksize(other, 16)
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
                return ndarray(x.thisptr.mul(Map[Vector3d](other))).ravel()
            else:
                # SE3 * [[x1, y1, z1], ..., [xn, yn, zn]] (N*3) np.array
                poses = x.matrix3x4().reshape((1, 12))
                return transform_points_by_poses(poses, other)
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

    def __reduce__(self):
        """
        For deepcopy purpose
        -----------------------------
        Out: tuple
        -----------------------------
        """
        return (self.__class__, (self.matrix(),))

    def __copy__(self):
        """
        Return a copy of SE3
        -----------------------------
        Out: SE3
        -----------------------------
        """
        return SE3(self)

    def matrix(self):
        """
        Returns and 4*4 np.ndarray matrix
        ---------------------------------
        Out: np.ndarray (4*4)
        ---------------------------------
        """
        return ndarray(self.thisptr.matrix()).copy()

    def matrix3x4(self):
        """
        Returns and 3*4 np.ndarray matrix
        ---------------------------------
        Out: np.ndarray (3*4)
        ---------------------------------
        """
        return ndarray(self.thisptr.matrix3x4()).copy()

    def so3(self):
        """
        Returns a SO3 rotation instance
        ---------------------------------
        Out: SO3
        ---------------------------------
        """
        return SO3(ndarray(self.thisptr.so3().matrix()))

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
        return ndarray(self.thisptr.log()).ravel().copy()

    def translation(self):
        """
        translation of SE3
        --------------------
        Out: np.ndarray (3,)
        --------------------
        """
        return ndarray(self.thisptr.translation()).ravel().copy()  # copy is neccessary

    def rotationMatrix(self):
        """
        Rotation matrix of SE3
        -----------------------
        Out: np.ndarray (3 * 3)
        -----------------------
        """
        return ndarray(self.thisptr.rotationMatrix()).copy()

    def setRotationMatrix(self, np.ndarray[DTYPE_t, ndim=2] R):
        """
        Set rotation matrix of SE3
        --------------------------
        In: np.ndarray (3 * 3)
        --------------------------
        """
        R = __tofortran(R)
        __checksize(R, 9)
        self.thisptr.setRotationMatrix(Map[Matrix3d](R))

    def setTranslation(self, np.ndarray[DTYPE_t, ndim=1] t):
        """
        Set translation matrix of SE3
        -----------------------------
        In: np.ndarray (3,)
        -----------------------------
        """
        __checksize(t, 3)
        so3 = SO3(self.rotationMatrix())
        del self.thisptr
        self.thisptr = new _SE3d(deref(so3.thisptr), Map[Vector3d](t))

    def copy(self):
        """
        Return a copy of SE3
        -----------------------------
        Out: SE3
        -----------------------------
        """
        return self.__copy__()

    @staticmethod
    def exp(np.ndarray[DTYPE_t, ndim=1] arr):
        """
        Lie algebra exp
        --------------------
        In: np.ndarray (6,)
        Out: SE3
        --------------------
        """
        __checksize(arr, 6)
        res = SE3()
        del res.thisptr
        res.thisptr = new _SE3d(_SE3d.exp(Map[VectorXd](arr)))
        return res