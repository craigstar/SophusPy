#ifndef SOPHUS_SO3_EXTENSION_HPP
#define SOPHUS_SO3_EXTENSION_HPP

#include "eigenex.hpp"

namespace Sophus
{
Eigen::PointsXd so3MulPoints(const SO3d &self, const Eigen::PointsXd &pts)
{
    Eigen::PointsXd newPts(pts.rows(), 3);
    for (int i = 0; i<pts.rows(); ++i)
    {
        newPts.row(i) = self * pts.row(i);
    }
    return newPts;
}
} // namespace Sophus
#endif
