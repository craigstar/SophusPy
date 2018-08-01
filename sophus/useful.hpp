#ifndef SOPHUS_USEFUL_HPP
#define SOPHUS_USEFUL_HPP

#include <Eigen/Dense>

namespace Eigen 
{
	typedef Matrix<double, Dynamic, 3> PointsXd;
	typedef Matrix<double, Dynamic, 12> PosesXd;
	typedef Matrix<double, 3, 4, RowMajor> RowPose34d;
	typedef Map<const RowPose34d> MapRowPose34d;
	typedef Matrix<double, 1, 12, RowMajor> RowVector12d;
}


namespace Sophus {

/** @brief Transform 3d points to new position by sequence of poses.
		   New points are stacked points of poses order.

@param poses (N, 12) matrix, each row is a 3 * 4 transform. Row order
       points (M, 3) 3d points
       bInv flag of inverting pose or not

@return PointsXd new position of (M * N, 3) matrix
 */
Eigen::PointsXd transformPointsByPoses(const Eigen::PosesXd &poses, const Eigen::PointsXd &points, const bool bInv)
{
	const int nPoints = points.rows();
	const int nPoses = poses.rows();
	Eigen::PointsXd newPoints(nPoints * nPoses, 3);

	if (0 >= nPoses || 0 >= nPoints)
	{
		return newPoints;
	}

	// transform points by pose
	for (int i = 0; i < poses.rows(); ++i)
	{
		Eigen::RowVector12d p(poses.row(i));
		Eigen::MapRowPose34d pose(p.data(), 3, 4);
		Eigen::Matrix3d R = pose.leftCols(3);
		Eigen::Vector3d t = pose.col(3);

		// invert pose
		if (bInv)
		{
			R.transposeInPlace();
			t = -R * t;
		}
		
		for (int j = 0; j < points.rows(); ++j)
		{
			Eigen::Vector3d pt = R * Eigen::Vector3d(points.row(j)) + t;
			newPoints.row(i * nPoints + j) = pt;
		}
	}
	return newPoints;
}
}  // namespace Sophus

#endif