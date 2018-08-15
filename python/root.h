#include "rootex.hpp"

namespace py = pybind11;

namespace Sophus
{
void declareRoot(py::module &m)
{
    m.def("invert_poses", &invertSinglePose, "Inverse a batch of poses together", py::arg("pose"));
    m.def("invert_poses", &invertPoses, "Inverse a batch of poses together", py::arg("poses"));
    m.def("copyto", &copytoSO3, "Copy one SO3d to another", py::arg("dst"), py::arg("src"));
    m.def("copyto", &copytoSE3, "Copy one SE3d to another", py::arg("dst"), py::arg("src"));
    m.def("to_orthogonal", &toOrthogonal, "Convert matrix3d to orthogonal", py::arg("R"));
    m.def("transform_points_by_poses",
          &transformPointsByPoses,
          "Transform 3d points to new position by sequence of poses. New points are stacked points of poses order.",
          py::arg("poses"), py::arg("points"), py::arg("need_inverse") = false);
}
} // end namespace Sophus