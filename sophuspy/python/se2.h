#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include "se2.hpp"
#include "so2.hpp"
#include "se2ex.hpp"

namespace py = pybind11;

namespace Sophus
{
void declareSE2(py::module &m)
{
    py::class_<SE2d> cls(m, "SE2");
    
    // initialization, constructor
    cls.def(py::init<>());
    cls.def(py::init<SE2d const &>(), py::arg("other"));
    cls.def(py::init<Eigen::Matrix3d const &>(), py::arg("other"));
    cls.def(py::init<Eigen::Matrix2d const &, Eigen::Vector2d const &>(), py::arg("R"), py::arg("t"));

    // private functions
    cls.def("__repr__", [](SE2d const &self) { return reprSE2(self.matrix()); });
    cls.def("__copy__", [](SE2d const &self) { return SE2d(self); });
    cls.def("__reduce__", [cls](SE2d const &self) { return py::make_tuple(cls, py::make_tuple(self.matrix())); });
    
    // operators
    cls.def(py::self * py::self);
    cls.def(py::self * Eigen::Vector2d());
    cls.def("__mul__", &se2MulPoints);
    cls.def("__imul__", (SE2d & (SE2d::*)(const SE2d &)) &SE2d::operator*=);

    // public functions
    cls.def("matrix", &SE2d::matrix, "Returns a 3 * 3 np.ndarray");
    cls.def("matrix2x3", &SE2d::matrix2x3, "Returns a 2 * 3 np.ndarray");
    cls.def("so2", (SO2d & (SE2d::*)()) & SE2d::so2, "Returns a SO2 rotation instance");
    cls.def("log", &SE2d::log, "Lie algebra log");
    cls.def("inverse", &SE2d::inverse, "Inverse of a 3 * 3 matrix");
    cls.def("copy", [](SE2d const &self) { return SE2d(self); }, "Return a copy of SE2");
    cls.def("translation", (Eigen::Vector2d & (SE2d::*)()) & SE2d::translation, "translation of SE2");
    cls.def("rotationMatrix", &SE2d::rotationMatrix, "rotation matrix of SE2");
    cls.def("setRotationMatrix", &SE2d::setRotationMatrix, "Set rotation matrix of SE2", py::arg("R"));
    cls.def("setTranslation", [](SE2d &self, Eigen::Vector2d const &t) { self.translation() = t; }, "Set translation vector of SE2", py::arg("t"));

    // static methods
    cls.def_static("hat", &SE2d::hat, "Hat of SE2");
    cls.def_static("exp", &SE2d::exp, "Computes the exponential map of a 3x1 se2 element");
}
} // end namespace Sophus
