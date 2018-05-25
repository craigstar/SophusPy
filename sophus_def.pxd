# -*- coding: utf-8 -*-
from eigency.core cimport *


cdef extern from "<sophus/so3.hpp>" namespace "Sophus":
    cdef cppclass SO3[Scalar]:
        SO3() except +
        # copy constructor
        SO3(const SO3&) except +
        SO3(const Map[Matrix3d]&) except +
        
        Matrix3d& matrix()

        Vector3d log()
        SO3 inverse()


cdef extern from "<sophus/se3.hpp>" namespace "Sophus":
    cdef cppclass SE3[Scalar]:
        SE3() except +
        # copy constructor
        SE3(const SE3&) except +
        SE3(const Map[Matrix4d]&) except +
        
        Matrix4d& matrix()

        # Vector3d log()
        SE3 inverse()
        Vector3d translation()
        Matrix4d rotationMatrix()

        void setRotationMatrix(const Map[Matrix3d]&)

        SE3 mul "operator*"(const SE3&)