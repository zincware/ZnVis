"""
Test for the Zooming Trajectory module
"""

import unittest

import numpy as np

from znvis.camera_trajectories import ZoomingTrajectory


class ZoomingTrajectoryTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the InterpolationCamera class for testing
        """
        cls.camera_trajectory = ZoomingTrajectory(
            total_frames=10,
            center=np.array([0, 0, 0]),
            initial_eye=np.array([0, 0, 10]),
            zoom_distance=5,
            up=np.array([0, 1, 0]),
            zoom_fraction=1.0,
        )

        cls.camera_trajectory_None = ZoomingTrajectory(
            total_frames=10,
            center=np.array([0, 0, 0]),
            initial_eye=np.array([0, 0, 10]),
            zoom_distance=5,
            up=np.array([0, 1, 0]),
            zoom_fraction=None,
        )

    def test_initialization(self):
        """
        Test the initialization of the camera trajectory
        """
        self.assertEqual(self.camera_trajectory.total_frames, 10)
        self.assertTrue(
            np.array_equal(self.camera_trajectory.center, np.array([0, 0, 0]))
        )
        self.assertTrue(
            np.array_equal(self.camera_trajectory.initial_eye, np.array([0, 0, 10]))
        )
        self.assertTrue(np.array_equal(self.camera_trajectory.zoom_distance, 5))
        self.assertTrue(np.array_equal(self.camera_trajectory.up, np.array([0, 1, 0])))
        self.assertEqual(self.camera_trajectory.zoom_fraction, 1.0)

        with self.assertRaises(ValueError):
            ZoomingTrajectory(
                total_frames=0,
                center=np.array([0, 0, 0]),
                initial_eye=np.array([0, 0, 10]),
                zoom_distance=5,
                up=np.array([0, 1, 0]),
                zoom_fraction=1.0,
            )

    def test_get_center_eye_up(self):
        """
        Test the get_center_eye_up method
        """
        assert np.allclose(
            self.camera_trajectory.get_center_eye_up(0),
            self.camera_trajectory.get_center_eye_up(-1),
        )
        assert np.allclose(
            self.camera_trajectory.get_center_eye_up(20),
            self.camera_trajectory.get_center_eye_up(10),
        )
