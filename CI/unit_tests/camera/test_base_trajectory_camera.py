"""
Test for the particle following camera module
"""

import unittest

import numpy as np

from znvis.camera_trajectories import CircularTrajectory
from znvis.cameras import TrajectoryCamera


class TrajectoryCameraTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the InterpolationCamera class for testing
        """

        cls.circular_trajectory = CircularTrajectory(
            total_frames=10,
            center=np.array([0, 0, 0]),
            radius=1,
            frames_per_rotation=10,
        )
        cls.rotating_camera = TrajectoryCamera(cls.circular_trajectory)

    def test_initialization(self):
        """
        Test the initialization of the KeyframeCamera class.
        """
        self.assertEqual(
            self.rotating_camera.trajectory is self.circular_trajectory, True
        )

    def test_get_view_matrix(self):
        """
        Test the get_view_matrix method.
        """
        # Test the get_view_matrix method with a valid frame index
        self.rotating_camera.get_view_matrix(0)
