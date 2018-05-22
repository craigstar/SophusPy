# -*- coding: utf-8 -*-
from eigency.core cimport *


cdef extern from "<sophus/so3.hpp>" namespace "Sophus":
    cdef cppclass SO3[Scalar]:
        SO3() except +
        # copy constructor
        SO3(const SO3&) except +
        SO3(const Map[Matrix3d]&) except +
        