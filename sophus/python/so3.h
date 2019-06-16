#include <pybind11/operators.h>
#include "so3ex.hpp"
#include "so3.hpp"

namespace py = pybind11;

namespace Sophus
{
void declareSO3(py::module &m)
{
    py::class_<SO3d> cls(m, "SO3");
    
    // initialization, constructor
    cls.def(py::init<>());
    cls.def(py::init<SO3d const &>(), py::arg("other"));
    cls.def(py::init<Eigen::Matrix3d const &>(), py::arg("other"));

    // private functions
    cls.def("__repr__", [](SO3d const &so3) { return repr(so3.matrix()); });
    cls.def("__copy__", [](SO3d const &so3) { return SO3d(so3); });
    cls.def("__reduce__", [cls](SO3d const &so3) { return py::make_tuple(cls, py::make_tuple(so3.matrix())); });

    // operators
    cls.def(py::self * py::self);
    cls.def(py::self * Eigen::Vector3d());
    cls.def("__mul__", &so3MulPoints);
    cls.def("__imul__", (SO3d & (SO3d::*)(const SO3d &)) &SO3d::operator*=);

    // public functions
    cls.def("matrix", &SO3d::matrix, "Returns a 3 * 3 np.ndarray");
    cls.def("log", &SO3d::log, "Lie algebra log");
    cls.def("inverse", &SO3d::inverse, "Inverse of a 3*3 othogonal matrix is the transpose of it");
    cls.def("copy", [](const SO3d &so3) { return SO3d(so3); }, "Return a copy of SO3");

    // static methods
    cls.def_static("hat", &SO3d::hat, "Hat of SO3 is to calculate the skew matrix");
    cls.def_static("exp", &SO3d::exp, "Computes the exponential map of a 3x1 so3 element");
}
} // end namespace Sophus
