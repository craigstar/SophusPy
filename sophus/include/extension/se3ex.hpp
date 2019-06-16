#ifndef SOPHUS_SE3_EXTENSION_HPP
#define SOPHUS_SE3_EXTENSION_HPP

#include "eigenex.hpp"

namespace Sophus
{

/** @brief Convert matrix to string representation

@param Eigen::Matrix4d matrix

@return std::string string representation of the input matrix
 */
std::string repr(const Eigen::Matrix4d &mat)
{
    std::stringstream ss;
	Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "    [", "]", "[", "]");
	ss << "SE3(" << mat.format(HeavyFmt) << ")";
	std::string out = ss.str();
	out.erase(5, 4);
    return out;
}

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
