#include <pybind11/operators.h>
#include "so2ex.hpp"
#include "so2.hpp"

namespace py = pybind11;

namespace Sophus
{
void declareSO2(py::module &m)
{
    py::class_<SO2d> cls(m, "SO2");
    
    // initialization, constructor
    cls.def(py::init<>());
    cls.def(py::init<SO2d const &>(), py::arg("other"));
    cls.def(py::init<Eigen::Matrix2d const &>(), py::arg("other"));

    // private functions
    cls.def("__repr__", [](SO2d const &so2) { return reprSO2(so2.matrix()); });
    cls.def("__copy__", [](SO2d const &so2) { return SO2d(so2); });
    cls.def("__reduce__", [cls](SO2d const &so2) { return py::make_tuple(cls, py::make_tuple(so2.matrix())); });

    // operators
    cls.def(py::self * py::self);
    cls.def(py::self * Eigen::Vector2d());
    cls.def("__mul__", &so2MulPoints);
    cls.def("__imul__", (SO2d & (SO2d::*)(const SO2d &)) &SO2d::operator*=);

    // public functions
    cls.def("matrix", &SO2d::matrix, "Returns a 2 * 2 np.ndarray");
    cls.def("log", &SO2d::log, "Lie algebra log");
    cls.def("inverse", &SO2d::inverse, "Inverse of a 2*2 othogonal matrix which is the transpose of it");
    cls.def("copy", [](const SO2d &so2) { return SO2d(so2); }, "Return a copy of SO2");

    // static methods
    cls.def_static("hat", &SO2d::hat, "Hat of SO2 is to calculate the skew matrix");
    cls.def_static("exp", &SO2d::exp, "Computes the exponential map of a 2x1 so2 element");
}
} // end namespace Sophus
