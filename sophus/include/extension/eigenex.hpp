#ifndef SOPHUS_EIGEN_HPP
#define SOPHUS_EIGEN_HPP

#include <Eigen/Dense>

namespace Eigen
{
typedef Matrix<double, Dynamic, 3> PointsXd;
typedef Matrix<double, Dynamic, 12> PosesXd;
typedef Matrix<double, 3, 4, RowMajor> RowPose34d;
typedef Map<const RowPose34d> MapRowPose34d;
typedef Matrix<double, 1, 12> Vector12d;
typedef Matrix<double, 1, 12, RowMajor> RowVector12d;
typedef Map<const RowVector12d> MapRowVector12d;

} // namespace Eigen

#endif