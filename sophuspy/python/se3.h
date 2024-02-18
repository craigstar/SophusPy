#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include "se3.hpp"
#include "so3.hpp"
#include "se3ex.hpp"

namespace py = pybind11;

namespace Sophus
{
void declareSE3(py::module &m)
{
    py::class_<SE3d> cls(m, "SE3");
    
    // initialization, constructor
    cls.def(py::init<>());
    cls.def(py::init<SE3d const &>(), py::arg("other"));
    cls.def(py::init<Eigen::Matrix4d const &>(), py::arg("other"));
    cls.def(py::init<Eigen::Matrix3d const &, Eigen::Vector3d const &>(), py::arg("R"), py::arg("t"));

    // private functions
    cls.def("__repr__", [](SE3d const &self) { return repr(self.matrix()); });
    cls.def("__copy__", [](SE3d const &self) { return SE3d(self); });
    cls.def("__reduce__", [cls](SE3d const &self) { return py::make_tuple(cls, py::make_tuple(self.matrix())); });
    
    // operators
    cls.def(py::self * py::self);
    cls.def(py::self * Eigen::Vector3d());
    cls.def("__mul__", &se3MulPoints);
    cls.def("__imul__", (SE3d & (SE3d::*)(const SE3d &)) &SE3d::operator*=);

    // public functions
    cls.def("matrix", &SE3d::matrix, "Returns a 4 * 4 np.ndarray");
    cls.def("matrix3x4", &SE3d::matrix3x4, "Returns a 3 * 4 np.ndarray");
    cls.def("so3", (SO3d & (SE3d::*)()) & SE3d::so3, "Returns a SO3 rotation instance");
    cls.def("log", &SE3d::log, "Lie algebra log");
    cls.def("inverse", &SE3d::inverse, "Inverse of a 4 * 4 matrix");
    cls.def("copy", [](SE3d const &self) { return SE3d(self); }, "Return a copy of SE3");
    cls.def("translation", (Eigen::Vector3d & (SE3d::*)()) & SE3d::translation, "translation of SE3");
    // cls.def("rotationMatrix", (Eigen::Matrix3d const &(SE3d::*)()) & SE3d::rotationMatrix, "rotation matrix of SE3");
    cls.def("rotationMatrix", &SE3d::rotationMatrix, "rotation matrix of SE3");
    cls.def("setRotationMatrix", &SE3d::setRotationMatrix, "Set rotation matrix of SE3", py::arg("R"));
    cls.def("setTranslation", [](SE3d &self, Eigen::Vector3d const &t) { self.translation() = t; }, "Set translation vector of SE3", py::arg("t"));

    // static methods
    cls.def_static("hat", &SE3d::hat, "Hat of SE3");
    cls.def_static("exp", &SE3d::exp, "Computes the exponential map of a 6x1 se3 element");
}
} // end namespace Sophus
