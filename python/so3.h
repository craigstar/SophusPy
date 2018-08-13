#include "so3.hpp"
#include "useful.hpp"

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
    cls.def("__str__", [](SO3d const &so3) { return Eigen::matToStr(so3.matrix()); });
    cls.def("__copy__", [](SO3d const &so3) { return SO3d(so3); });
    cls.def("__reduce__", [cls](SO3d const &so3) { return py::make_tuple(cls, py::make_tuple(so3.matrix())); });

    // public functions
    cls.def("matrix", &SO3d::matrix, "Returns a 3 * 3 np.ndarray");
    cls.def("log", &SO3d::log, "Lie algebra log");
    cls.def("inverse", &SO3d::inverse, "Inverse of a 3*3 othogonal matrix is the transpose of it");
    cls.def("copy", [](const SO3d &so3) { return SO3d(so3); }, "Return a copy of SO3");

    // static methods
    cls.def("hat", &SO3d::hat, "Hat of SO3 is to calculate the skew matrix");
    cls.def("exp", &SO3d::exp, "Computes the exponential map of a 3x1 so3 element");
}
} // end namespace Sophus
