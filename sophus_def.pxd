# -*- coding: utf-8 -*-
from eigency.core cimport *
from libcpp cimport bool


cdef extern from "<sophus/so3.hpp>" namespace "Sophus":
    cdef cppclass SO3[Scalar]:
        SO3() except +
        # copy constructor
        SO3(const SO3&) except +
        SO3(const Map[Matrix3d]&) except +
        
        Matrix3d& matrix()

        Vector3d log()
        SO3 inverse()
        
        @staticmethod
        Matrix3d hat(const Map[Vector3d]&)
        @staticmethod
        SO3[Scalar] exp(const Map[Vector3d]&)


cdef extern from "<sophus/se3.hpp>" namespace "Sophus":
    cdef cppclass SE3[Scalar]:
        SE3() except +
        # copy constructor
        SE3(const SE3&) except +
        SE3(const Map[Matrix4d]&) except +
        SE3(const SO3&, const Map[Vector3d]&) except +
        
        SE3 mul "operator*"(const SE3&)
        SE3 mul_assign "operator*="(const SE3&)
        Vector3d mul "operator*"(const Map[Vector3d]&)

        Matrix4d& matrix()
        MatrixXd& matrix3x4()

        SO3 so3()
        SE3 inverse()
        VectorXd log()
        Vector3d translation()
        Matrix4d rotationMatrix()

        void setRotationMatrix(const Map[Matrix3d]&)

        @staticmethod
        SE3[Scalar] exp(const Map[VectorXd]&)

cdef extern from "<sophus/useful.hpp>" namespace "Sophus":
    cdef MatrixXd& transformPointsByPoses(const Map[MatrixXd]&, const Map[MatrixXd]&, const bool)