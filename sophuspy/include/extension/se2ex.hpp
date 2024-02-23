#ifndef SOPHUS_SE2_EXTENSION_HPP
#define SOPHUS_SE2_EXTENSION_HPP

#include "eigenex.hpp"

namespace Sophus
{

/** @brief Convert matrix to string representation

@param Eigen::Matrix4d matrix

@return std::string string representation of the input matrix
 */
std::string reprSE2(const Eigen::Matrix3d &mat)
{
    std::stringstream ss;
	Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "    [", "]", "[", "]");
	ss << "SE2(" << mat.format(HeavyFmt) << ")";
	std::string out = ss.str();
	out.erase(5, 4);
    return out;
}

Eigen::MatrixX2d se2MulPoints(const SE2d &self, const Eigen::MatrixX2d &pts)
{
    Eigen::MatrixX2d newPts(pts.rows(), 2);
    for (int i = 0; i<pts.rows(); ++i)
    {
        newPts.row(i) = self * pts.row(i);
    }
    return newPts;
}
} // namespace Sophus
#endif
