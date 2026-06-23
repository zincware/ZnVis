"""
Test Module for Camera
"""

import unittest

import numpy as np

from znvis.cameras.static_camera import StaticCamera


class StaticCameraTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the StaticCamera class for testing.
        """

        cls.center = np.array([119, 175, 0])
        cls.eye = np.array([200 / 2, 200 / 2, 240])
        cls.up = np.array([0, 1, 0])

        cls.view_matrix = np.array(
            [
                [-1.0000000e00, 0.0000000e00, 0.0000000e00, -50.0000000e00],
                [0.0000000e00, 1.0000000e00, 0.0000000e00, -50.0000000e00],
                [0.0000000e00, 0.0000000e00, -1.0000000e00, -240.0000000e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000],
            ]
        )

        cls.look_at_camera = StaticCamera(center=cls.center, eye=cls.eye, up=cls.up)
        cls.view_matrix_camera = StaticCamera(view_matrix=cls.view_matrix)

    def test_look_at(self):
        """
        Test the look_at method.
        """
        correct_view_matrix = np.array(
            [
                [9.96880973e-01, -0.00000000e00, 7.89197437e-02, -1.18628836e02],
                [-2.34728691e-02, 9.54744537e-01, 2.96499399e-01, -1.64287023e02],
                [-7.53481941e-02, -2.97427082e-01, 9.51766663e-01, -1.91146471e02],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        self.assertTrue(
            np.allclose(self.look_at_camera.view_matrix, correct_view_matrix)
        )

    def test_view_matrix(self):
        """
        Test the view_matrix attribute.
        """
        self.assertTrue(
            np.allclose(self.view_matrix_camera.view_matrix, self.view_matrix)
        )

    def test_false_initialized_camera(self):
        """
        Test if the camera is correctly initialized with a false parameter.
        """
        with self.assertRaises(ValueError):
            StaticCamera(center=None, eye=None, up=None)
        with self.assertRaises(ValueError):
            StaticCamera(view_matrix=None)

    def test_get_view_matrix_from_particle_positions(self):
        """
        Test the get_view_matrix_from_particle_positions method.
        """
        particle_positions = np.array([[1, 2, 0], [1, 2, 1], [1, 2, 2]])
        view_matrix = self.look_at_camera.get_view_matrix_from_particle_positions(
            particle_positions
        )
        expected_center = np.array([1.0, 2.0, 1.0])
        expected_eye = np.array([1.0, 2.0, 3.0])
        expected_up = np.array([0.0, 1.0, 0.0])
        expected_view_matrix = self.look_at_camera.look_at(
            expected_center, expected_eye, expected_up
        )
        self.assertTrue(np.allclose(view_matrix, expected_view_matrix))

    def test_get_view_matrix_from_box_size(self):
        """
        Test the get_view_matrix_from_box_size method.
        """
        box_size = np.array([1, 1, 1])
        view_matrix = self.look_at_camera.get_view_matrix_from_box_size(box_size)
        expected_view_matrix = np.array(
            [[1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, 0, 1, -1.6], [0, 0, 0, 1]]
        )
        self.assertTrue(np.allclose(view_matrix, expected_view_matrix))

    def test_get_minimal_view_matrix(self):
        """
        Test the get_minimal_view_matrix method.
        """
        box_size = np.array([1, 1, 1])
        renderer_resolution = np.array([800, 600])
        view_matrix = self.look_at_camera.get_minimal_view_matrix(
            box_size, renderer_resolution
        )

        expected_view_matrix = np.array(
            [
                [
                    1,
                    0,
                    0,
                    -0.5,
                ],
                [0, 1, 0, -0.5],
                [0, 0, 1, -1.3660254],
                [0, 0, 0, 1],
            ]
        )
        self.assertTrue(np.allclose(view_matrix, expected_view_matrix))

    def test_get_view_matrix(self):
        """
        Test the get_view_matrix method.
        """
        view_matrix_frame0 = self.look_at_camera.get_view_matrix(frame_index=0)
        view_matrix_frame1 = self.look_at_camera.get_view_matrix(frame_index=1)
        expected_view_matrix = np.array(
            [
                [9.96880973e-01, -0.00000000e00, 7.89197437e-02, -1.18628836e02],
                [-2.34728691e-02, 9.54744537e-01, 2.96499399e-01, -1.64287023e02],
                [-7.53481941e-02, -2.97427082e-01, 9.51766663e-01, -1.91146471e02],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        self.assertTrue(np.allclose(view_matrix_frame0, expected_view_matrix))
        self.assertTrue(np.allclose(view_matrix_frame1, expected_view_matrix))

    def test_set_view_matrix(self):
        """
        Test the set_view_matrix method.
        """
        new_view_matrix = np.array(
            [[1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, 0, 1, -1.3660254], [0, 0, 0, 1]]
        )
        self.look_at_camera.set_view_matrix(new_view_matrix)
        self.assertTrue(np.allclose(self.look_at_camera.view_matrix, new_view_matrix))

    def test_get_center_eye_up_from_view_matrix(self):
        """
        Test the get_center_eye_up_from_view_matrix method.
        NOTE: Center cannot be extracted accurately from the view matrix
        in the sense that the information about the exact center is lost due
        to normalization of the forward vector in the process of creating the view
        matrix. Therefore, this is only an approximation for the center!
        """
        view_matrix = np.array(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, -3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        _center, eye, up = self.look_at_camera.get_center_eye_up_from_view_matrix(
            view_matrix
        )
        expected_eye = np.array([1.0, 2.0, 3.0])
        expected_up = np.array([0.0, 1.0, 0.0])

        self.assertTrue(np.allclose(eye, expected_eye))
        self.assertTrue(np.allclose(up, expected_up))
