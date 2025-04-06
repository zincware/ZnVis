"""
Test for the particle following camera module
"""

import unittest

import numpy as np

from znvis.cameras import ParticleFollowingCamera


class ParticleFollowingCameraTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the InterpolationCamera class for testing
        """
        particle_positions = np.array([[1, 2, 0], [1, 1, 0]])

        cls.particle_camera = ParticleFollowingCamera(
            particle_positions=particle_positions,
            camera_particle_vector=np.array([0, 0, 20]),
            camera_up_vector=np.array([0, 1, 0]),
        )

    def test_initialization(self):
        """
        Test the initialization of the KeyframeCamera class.
        """
        np.allclose(self.particle_camera.camera_particle_vector, np.array([0, 0, 20]))
        np.allclose(self.particle_camera.camera_up_vector, np.array([0, 1, 0]))
        self.assertEqual(self.particle_camera.particle_positions.shape, (2, 3))
        self.assertEqual(self.particle_camera.view_matrix.shape, (4, 4))

    def test_get_view_matrix_from_particle_positions(self):
        """
        Test the get_view_matrix_from_particle_positions method.
        """
        view_matrix = self.particle_camera.get_view_matrix(0)
        expected_view_matrix = np.array(
            [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -20], [0, 0, 0, 1]]
        )
        np.allclose(view_matrix, expected_view_matrix)
