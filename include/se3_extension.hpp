#ifndef SOPHUS_SE3_EXTENSION_HPP
#define SOPHUS_SE3_EXTENSION_HPP

#include "eigen.hpp"

namespace Sophus
{
Eigen::PointsXd se3MulPoints(const SE3d &self, const Eigen::PointsXd &pts)
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
