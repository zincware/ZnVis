"""
Test for the circular trajectory module
"""

import unittest

import numpy as np

from znvis.cameras.trajectories import CircularTrajectory


class CircularTrajectoryTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the InterpolationCamera class for testing
        """
        cls.camera_trajectory = CircularTrajectory(
            total_frames=20,
            center=np.array([0, 0, 0]),
            radius=5,
            frames_per_rotation=10,
            start_angle=0,
            rotation_angle=2 * np.pi,
            loop=False,
            ping_pong=False,
            rotation_axis="y",
            clockwise=True,
            smoothing=True,
        )
        cls.loop_pingpong_camera_trajectory = CircularTrajectory(
            total_frames=25,
            center=np.array([0, 0, 0]),
            radius=5,
            frames_per_rotation=10,
            start_angle=0,
            rotation_angle=2 * np.pi,
            loop=False,
            ping_pong=False,
            rotation_axis="y",
            clockwise=True,
            smoothing=False,
        )
        cls.not_clockwise_camera_trajectory = CircularTrajectory(
            total_frames=25,
            center=np.array([0, 0, 0]),
            radius=5,
            frames_per_rotation=10,
            start_angle=0,
            rotation_angle=2 * np.pi,
            loop=False,
            ping_pong=False,
            rotation_axis="y",
            clockwise=False,
            smoothing=True,
        )

        cls.camera_trajectory_axis_check = CircularTrajectory(
            total_frames=25,
            center=np.array([0, 0, 0]),
            radius=5,
            frames_per_rotation=10,
            start_angle=0,
            rotation_angle=2 * np.pi,
            loop=False,
            ping_pong=False,
            rotation_axis="x",
            clockwise=True,
            smoothing=True,
        )

    def test_initialization(self):
        """
        Test the initialization of the camera trajectory
        """
        self.assertEqual(self.camera_trajectory.total_frames, 20)
        self.assertTrue(
            np.array_equal(self.camera_trajectory.center, np.array([0, 0, 0]))
        )
        self.assertTrue(np.array_equal(self.camera_trajectory.radius, 5))
        self.assertTrue(np.array_equal(self.camera_trajectory.frames_per_rotation, 10))
        self.assertTrue(np.array_equal(self.camera_trajectory.start_angle, 0))
        self.assertTrue(
            np.array_equal(self.camera_trajectory.rotation_angle, 2 * np.pi)
        )
        self.assertTrue(np.array_equal(self.camera_trajectory.loop, False))
        self.assertTrue(np.array_equal(self.camera_trajectory.ping_pong, False))
        self.assertTrue(np.array_equal(self.camera_trajectory.rotation_axis, "y"))
        self.assertTrue(np.array_equal(self.camera_trajectory.clockwise, True))
        self.assertTrue(np.array_equal(self.camera_trajectory.smoothing, True))

        # Raise a frames_per_rotation error
        zero_frames = CircularTrajectory(
            total_frames=20,
            center=np.array([0, 0, 0]),
            radius=5,
            frames_per_rotation=0,
            start_angle=0,
            rotation_angle=2 * np.pi,
            loop=False,
            ping_pong=False,
            rotation_axis="x",
            clockwise=True,
            smoothing=True,
        )
        self.assertTrue(np.array_equal(zero_frames.frames_per_rotation, 20))

        with self.assertRaises(ValueError):
            # Raise the xyz axis error
            CircularTrajectory(
                total_frames=20,
                center=np.array([0, 0, 0]),
                radius=5,
                frames_per_rotation=10,
                start_angle=0,
                rotation_angle=2 * np.pi,
                loop=False,
                ping_pong=False,
                rotation_axis="a",
                clockwise=True,
                smoothing=True,
            )

        with self.assertRaises(ValueError):
            # Raise a frames_per_rotation error
            zero_frames = CircularTrajectory(
                total_frames=20,
                center=np.array([0, 0, 0]),
                radius=5,
                frames_per_rotation=-4,
                start_angle=0,
                rotation_angle=2 * np.pi,
                loop=False,
                ping_pong=False,
                rotation_axis="x",
                clockwise=True,
                smoothing=True,
            )

    def test_get_loop_and_ping_pong_frame_index(self):
        """
        Test the get_loop_and_ping_pong_frame_index method
        """
        # Test with loop = True and ping_pong = False
        self.loop_pingpong_camera_trajectory.loop = True
        self.loop_pingpong_camera_trajectory.ping_pong = False
        for i in range(20):
            frame_index = (
                self.loop_pingpong_camera_trajectory.get_loop_and_ping_pong_frame_index(
                    i
                )
            )
            self.assertEqual(frame_index, i % 10)

        # Test with loop = False and ping_pong = False
        self.loop_pingpong_camera_trajectory.loop = False
        self.loop_pingpong_camera_trajectory.ping_pong = False

        for i in range(20):
            frame_index = (
                self.loop_pingpong_camera_trajectory.get_loop_and_ping_pong_frame_index(
                    i
                )
            )
            if i < 10:
                self.assertEqual(frame_index, i)
            elif i >= 10:
                self.assertEqual(frame_index, 10 - 1)

        # Test with loop = False and ping_pong = True
        self.loop_pingpong_camera_trajectory.loop = False
        self.loop_pingpong_camera_trajectory.ping_pong = True

        for i in range(20):
            frame_index = (
                self.loop_pingpong_camera_trajectory.get_loop_and_ping_pong_frame_index(
                    i
                )
            )
            if i <= 10:
                self.assertEqual(frame_index, i)
            else:
                self.assertEqual(frame_index, 20 - i)

        self.assertEqual(
            self.loop_pingpong_camera_trajectory.get_loop_and_ping_pong_frame_index(21),
            self.loop_pingpong_camera_trajectory.get_loop_and_ping_pong_frame_index(0),
        )

        # Test with loop = True and ping_pong = True
        self.loop_pingpong_camera_trajectory.loop = True
        self.loop_pingpong_camera_trajectory.ping_pong = True

        for i in range(20):
            frame_index = (
                self.loop_pingpong_camera_trajectory.get_loop_and_ping_pong_frame_index(
                    i
                )
            )
            print(i, frame_index)
            if i <= 10:
                self.assertEqual(frame_index, i)
            else:
                self.assertEqual(frame_index, 20 - i)

    def test_get_center_eye_up(self):
        """
        Test the get_center_eye_up method
        """
        center, eye, up = self.not_clockwise_camera_trajectory.get_center_eye_up(0)
        print(center, eye, up)
        assert np.allclose(center, self.not_clockwise_camera_trajectory.center)
        assert np.allclose(eye, np.array([0, 0, 5]))
        assert np.allclose(up, np.array([0, 1, 0]))

    def test_other_planes(self):
        """
        Test the get_center_eye_up method for other planes
        """
        self.camera_trajectory_axis_check.rotation_axis = "x"
        up_x = np.array([1, 0, 0])
        up_y = np.array([0, 1, 0])
        up_z = np.array([0, 0, 1])

        for axis in ["x", "y", "z"]:
            self.camera_trajectory_axis_check.rotation_axis = axis
            _, _, up = self.camera_trajectory_axis_check.get_center_eye_up(0)
            if axis == "x":
                assert np.allclose(up, up_x)
            elif axis == "y":
                assert np.allclose(up, up_y)
            elif axis == "z":
                assert np.allclose(up, up_z)
