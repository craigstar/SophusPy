#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include "se3.hpp"
#include "se3_extension.hpp"
#include "useful.hpp"

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
    cls.def("__str__", [](SE3d const &self) { return Eigen::matToStr(self.matrix()); });
    cls.def("__copy__", [](SE3d const &self) { return SE3d(self); });
    cls.def("__reduce__", [cls](SE3d const &self) { return py::make_tuple(cls, py::make_tuple(self.matrix())); });
    
    // operators
    cls.def(py::self * py::self);
    cls.def("__imul__", [](SE3d &self, SE3d const &other) { self *= other; });
    cls.def(py::self * Eigen::Vector3d());
    cls.def("__mul__", &se3MulPoints);

    // public functions
    cls.def("matrix", &SE3d::matrix, "Returns a 4 * 4 np.ndarray");
    // cls.def("log", &SO3d::log, "Lie algebra log");
    // cls.def("inverse", &SO3d::inverse, "Inverse of a 3*3 othogonal matrix is the transpose of it");
    cls.def("copy", [](const SE3d &self) { return SE3d(self); }, "Return a copy of SE3");

    // // static methods
    // cls.def("hat", &SO3d::hat, "Hat of SO3 is to calculate the skew matrix");
    // cls.def("exp", &SO3d::exp, "Computes the exponential map of a 3x1 so3 element");
}
} // end namespace Sophus
