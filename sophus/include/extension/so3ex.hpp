#ifndef SOPHUS_SO3_EXTENSION_HPP
#define SOPHUS_SO3_EXTENSION_HPP

#include "eigenex.hpp"

namespace Sophus
{
/** @brief Convert matrix to string representation

@param Eigen::Matrix3d matrix

@return std::string string representation of the input matrix
 */
std::string repr(const Eigen::Matrix3d &mat)
{
    std::stringstream ss;
	Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "    [", "]", "[", "]");
	ss << "SO3(" << mat.format(HeavyFmt) << ")";
	std::string out = ss.str();
	out.erase(5, 4);
    return out;
}

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
