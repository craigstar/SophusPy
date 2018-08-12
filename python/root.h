#include "useful.hpp"

namespace py = pybind11;

namespace Sophus
{
void declareRoot(py::module &m)
{
    m.def("invert_poses", &invertPoses, "Inverse a batch of poses together",
          py::arg("poses"));
}
} // end namespace Sophus