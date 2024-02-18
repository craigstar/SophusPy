import numpy as np
import unittest
import pytest

import sophuspy as sp


class TestRoot(unittest.TestCase):
    def setUp(self):
        self.Rnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787],
                             [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596],
                             [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105]])

        self.Tnp = np.array([[-0.02495988040066277, 0.01720436961811805,   0.9995404014027787, -1103.428193030075],
                             [ 0.06813350186490005,   0.997556284701166, -0.01546883243273596, -35.93298768047541],
                             [ -0.9973639407428014, 0.06771608759556018, -0.02607107950853105,  798.7780129474496],
                             [                   0,                   0,                    0,                  1]])

    def test_copy_so3_inplace(self):
        R1, R2 = sp.SO3(), sp.SO3(self.Rnp)
        id_old = id(R1)
        sp.copyto(R1, R2)
        id_new = id(R1)
        self.assertTrue(np.allclose(R1.matrix(), R2.matrix()))
        self.assertEqual(id_old, id_new)

    def test_copy_se3_inplace(self):
        T1, T2 = sp.SE3(), sp.SE3(self.Tnp)
        id_old = id(T1)
        sp.copyto(T1, T2)
        id_new = id(T1)
        self.assertTrue(np.allclose(T1.matrix(), T2.matrix()))
        self.assertEqual(id_old, id_new)

    def test_copy_inplace_failure(self):
        R, T = sp.SO3(), sp.SE3()
        with pytest.raises(TypeError) as e:
            sp.copyto(R, T)
        self.assertTrue('incompatible function arguments' in str(e.value))

        with pytest.raises(TypeError) as e:
            sp.copyto(T, R)
        self.assertTrue('incompatible function arguments' in str(e.value))

    def _prepare_points_and_poses(self):
        points = np.array([[1, -1, 1],
                           [2,  3, 4]], dtype=np.float64)
        pose1 = self.Tnp[:3]
        pose2 = np.linalg.inv(self.Tnp)[:3]
        poses = np.vstack((pose1.ravel(), pose2.ravel()))
        return (poses, points, pose1, pose2)

    def test_transform_points_by_poses_multi_success(self):
        poses, points, pose1, pose2 = self._prepare_points_and_poses()
        points_homo = np.hstack((points, np.ones((2, 1))))

        new_points1 = pose1.dot(points_homo.T).T
        new_points2 = pose2.dot(points_homo.T).T
        new_points = np.vstack((new_points1, new_points2))

        sp_new_points = sp.transform_points_by_poses(poses, points)
        self.assertTrue(np.allclose(sp_new_points, new_points))

    def test_transform_points_by_poses_single_success(self):
        poses, points, pose1, _ = self._prepare_points_and_poses()
        points_homo = np.hstack((points, np.ones((2, 1))))

        new_points = pose1.dot(points_homo.T).T
        sp_new_points = sp.transform_points_by_poses(poses[0], points)
        self.assertTrue(np.allclose(sp_new_points, new_points))

    def test_transform_points_by_poses_inverse_multi_success(self):
        poses, points, pose2_inv, pose1_inv = self._prepare_points_and_poses()
        points_homo = np.hstack((points, np.ones((2, 1))))

        new_points1 = pose1_inv.dot(points_homo.T).T
        new_points2 = pose2_inv.dot(points_homo.T).T
        new_points = np.vstack((new_points1, new_points2))

        sp_new_points = sp.transform_points_by_poses(poses, points, True)
        self.assertTrue(np.allclose(sp_new_points, new_points))

    def test_transform_points_by_poses_inverse_single_success(self):
        poses, points, _, pose1_inv = self._prepare_points_and_poses()
        points_homo = np.hstack((points, np.ones((2, 1))))

        new_points = pose1_inv.dot(points_homo.T).T
        sp_new_points = sp.transform_points_by_poses(poses[0], points, True)
        self.assertTrue(np.allclose(sp_new_points, new_points))

    def test_transform_points_by_poses_empty_poses_points_success(self):
        poses, points, pose1, pose2 = self._prepare_points_and_poses()
        sp_new_points = sp.transform_points_by_poses(np.zeros((0, 12)), points)
        self.assertEqual(sp_new_points.shape, (0, 3))

        poses, points, pose1, pose2 = self._prepare_points_and_poses()
        sp_new_points = sp.transform_points_by_poses(poses, np.zeros((0, 3)))
        self.assertEqual(sp_new_points.shape, (0, 3))

    def test_invert_poses_multi_success(self):
        poses, _, pose2_inv, pose1_inv = self._prepare_points_and_poses()
        new_poses = np.vstack((pose1_inv.ravel(), pose2_inv.ravel()))
        sp_new_poses = sp.invert_poses(poses)
        self.assertTrue(np.allclose(sp_new_poses, new_poses))

    def test_invert_poses_single_success(self):
        poses, _, pose2_inv, pose1_inv = self._prepare_points_and_poses()   
        sp_new_pose = sp.invert_poses(poses[0])
        self.assertTrue(np.allclose(sp_new_pose, pose1_inv.ravel()))
        self.assertEqual(sp_new_pose.shape, (12,))

    def test_invert_poses_empty_poses_success(self):
        sp_new_poses = sp.invert_poses(np.zeros((0, 12)))
        self.assertEqual(sp_new_poses.shape, (0, 12))

    def test_to_orthogonal_success(self):
        R = np.eye(3)
        R[0, 1] = 1e-3
        R = sp.to_orthogonal(R)
        ans = np.array([[ 9.99999875e-01,  5.00000000e-04,  0.00000000e+00],
                        [-5.00000000e-04,  9.99999875e-01, -0.00000000e+00],
                        [-0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        self.assertTrue(np.allclose(R, ans))