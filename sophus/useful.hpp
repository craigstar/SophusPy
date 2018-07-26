#ifndef SOPHUS_USEFUL_HPP
#define SOPHUS_USEFUL_HPP

#include <Eigen/Core>


typedef Eigen::Matrix<double, Eigen::Dynamic, 3> PointsXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 12> PosesXd;
typedef Eigen::Matrix<double, 3, 4> Pose34d;

namespace Sophus {

PointsXd transformPointsByPoses(const PosesXd poses, const PointsXd points)
{
    PointsXd newPts = PointsXd(points.rows(), points.cols());
    Pose34d pose;
    for (int i = 0; i < poses.rows(); ++i)
    {
        memcpy(pose.data(), poses.row(i).data(), 6*sizeof(double))
    }

    

    return newPts;
}

}  // namespace Sophus

#endif