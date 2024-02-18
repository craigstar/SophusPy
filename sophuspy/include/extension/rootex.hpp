#ifndef SOPHUS_USEFUL_HPP
#define SOPHUS_USEFUL_HPP

#include "so3.hpp"
#include "se3.hpp"
#include "eigenex.hpp"

namespace Sophus
{
/** @brief Transform 3d points to new position by sequence of poses.
		   New points are stacked points of poses order.

@param poses (N, 12) matrix, each row is a 3 * 4 transform. Row order
       points (M, 3) 3d points
       bInv flag of inverting pose or not

@return PointsXd new position of (M * N, 3) matrix
 */
Eigen::PointsXd transformPointsByPoses(const Eigen::PosesXd &poses, const Eigen::PointsXd &points, const bool bInv=false)
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

/** @brief Inverse a batch of poses together

@param poses (N, 12) matrix, each row is a 3 * 4 transform. Row order

@return PosesXd new inverted poses of (N, 12) matrix
 */
Eigen::PosesXd invertPoses(const Eigen::PosesXd &poses)
{
	const int nPoses = poses.rows();
	Eigen::PosesXd newPoses(nPoses, 12);

	if (0 >= nPoses)
	{
		return newPoses;
	}

	// invert poses
	Eigen::RowPose34d newPose;
	for (int i = 0; i < poses.rows(); ++i)
	{
		Eigen::RowVector12d p(poses.row(i));
		Eigen::MapRowPose34d pose(p.data(), 3, 4);
		Eigen::Matrix3d R = pose.leftCols(3);
		Eigen::Vector3d t = pose.col(3);

		newPose.leftCols(3) = R.transpose();
		newPose.col(3) = -R.transpose() * t;
		newPoses.row(i) = Eigen::MapRowVector12d(newPose.data(), 12);
	}
	return newPoses;
}

/** @brief Inverse a single of pose

@param poses Vector12d, is a 3 * 4 transform. Row order

@return Vector12d new inverted pose
 */
Eigen::Vector12d invertSinglePose(const Eigen::Vector12d &pose) { return invertPoses(pose); }

/** @brief Copy one SO3d to another

@param dst SO3d
	   src SO3d

@return void
 */
void copytoSO3(SO3d &dst, const SO3d &src) { dst = src; }

/** @brief Copy one SE3d to another

@param dst SE3d
	   src SE3d

@return void
 */
void copytoSE3(SE3d &dst, const SE3d &src) { dst = src; }

/** @brief convert matrix to orthogonal

@param R Eigen::Matrix3d

@return Eigen::Matrix3d
 */
Eigen::Matrix3d toOrthogonal(const Eigen::Matrix3d &R)
{	
	Eigen::Quaterniond q(R);
	return q.normalized().toRotationMatrix();
}
} // namespace Sophus

#endif