#ifndef SOPHUS_SO2_EXTENSION_HPP
#define SOPHUS_SO2_EXTENSION_HPP

#include "eigenex.hpp"
#include "so2.hpp"

namespace Sophus
{
/** @brief Convert matrix to string representation

@param Eigen::Matrix2d matrix

@return std::string string representation of the input matrix
 */
std::string reprSO2(const Eigen::Matrix2d &mat)
{
    std::stringstream ss;
	Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, ", ", ",\n", "    [", "]", "[", "]");
	ss << "SO2(" << mat.format(HeavyFmt) << ")";
	std::string out = ss.str();
	out.erase(5, 4);
    return out;
}

Eigen::MatrixX2d so2MulPoints(const SO2d &self, const Eigen::MatrixX2d &pts)
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
