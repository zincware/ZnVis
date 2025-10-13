"""
Test for the Base Trajectory module
"""

import unittest

import numpy as np

from znvis.cameras.trajectories import BaseTrajectory


class BaseTrajectoryTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the InterpolationCamera class for testing
        """
        cls.camera_trajectory = BaseTrajectory(total_frames=10)

    def test_initialization(self):
        """
        Test the initialization of the KeyframeCamera class.
        """
        self.assertEqual(self.camera_trajectory.total_frames, 10)
        with self.assertRaises(NotImplementedError):
            self.camera_trajectory.get_center_eye_up(0)

    def test_get_center_eye_up_from_view_matrix(self):
        """
        Test the get_center_eye_up_from_view_matrix method.
        """
        view_matrix = np.array(
            [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]]
        )
        center, eye, up = self.camera_trajectory.get_center_eye_up_from_view_matrix(
            view_matrix
        )

        expected_eye = np.array([1, 2, 3])
        expected_up = np.array([0, 1, 0])
        expected_center = np.array([1, 2, 2])
        self.assertTrue(np.allclose(eye, expected_eye))
        self.assertTrue(np.allclose(up, expected_up))
        self.assertTrue(np.allclose(center, expected_center))
