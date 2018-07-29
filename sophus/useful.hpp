#ifndef SOPHUS_USEFUL_HPP
#define SOPHUS_USEFUL_HPP

#include <Eigen/Dense>

typedef Eigen::Matrix<double, Eigen::Dynamic, 3> PointsXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 12> PosesXd;
typedef Eigen::Matrix<double, 3, 4, Eigen::RowMajor> RowPose34d;
typedef Eigen::Map<const RowPose34d> MapRowPose34d;
typedef Eigen::Matrix<double, 1, 12, Eigen::RowMajor> RowVector12d;

namespace Sophus {

PointsXd transformPointsByPoses(const PosesXd &poses, const PointsXd &points)
{
	const clock_t begin_time = std::clock();

	int nPoints = points.rows();
	int nPoses = poses.rows();
	PointsXd newPoints(nPoints * nPoses, 3);

	if (nPoses <= 0 || nPoints <= 0 || poses.cols() != 12 || points.cols() != 3)
	{
		return newPoints;
	}

	// all points need to map all poses 
	Eigen::Vector3d pt;

	// transform points by pose
	for (int i = 0; i < poses.rows(); ++i)
	{
		RowVector12d p(poses.row(i));
		MapRowPose34d pose(p.data(), 3, 4);
		
		for (int j = 0; j < points.rows(); ++j)
		{
			pt = pose * Eigen::Vector4d(points.row(j).homogeneous());
			newPoints.row(i * nPoints + j) = pt;
		}
	}
	std::cout << float( std::clock () - begin_time ) /  CLOCKS_PER_SEC;
	return newPoints;
}
}  // namespace Sophus

#endif