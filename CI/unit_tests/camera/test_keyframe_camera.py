"""
Test for the keyframe camera module
"""

import unittest
from pathlib import Path

import numpy as np

from znvis.cameras import KeyframeCamera


class KeyframeCameraTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the KeyframeCamera class for testing
        """

        project_root = Path(__file__).resolve().parents[2]
        path_to_interpolated_view_matrices = (
            project_root
            / "test_files"
            / "keyframe_camera"
            / "interpolated_view_matrices.npy"
        )

        cls.keyframe_camera = KeyframeCamera(
            view_matrices_path=path_to_interpolated_view_matrices
        )

    def test_interpolate_view_matrices(self):
        """
        Test the interpolation of view matrices.
        By this, the smoothing is also tested.
        """
        view_matrices_dict = {
            0: np.array(
                [
                    [1.0000000e00, -1.8031891e-08, 5.3372737e-09, -8.3287910e01],
                    [-2.0071333e-08, 9.9999309e-01, -3.7090916e-03, -8.6779434e01],
                    [1.0577577e-07, 3.7090357e-03, 9.9999303e-01, -2.3617999e02],
                    [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
                ]
            ),
            20: np.array(
                [
                    [-1.4584593e-01, 1.5409360e-02, -9.8918748e-01, 2.0602367e01],
                    [2.6641423e-02, 9.9957752e-01, 1.1643190e-02, -8.9114258e01],
                    [9.8894864e-01, -2.4655251e-02, -1.4619486e-01, -3.0473773e02],
                    [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
                ]
            ),
            199: np.array(
                [
                    [-1.4584593e-01, 1.5409360e-02, -9.8918748e-01, 2.0602367e01],
                    [2.6641423e-02, 9.9957752e-01, 1.1643190e-02, -8.9114258e01],
                    [9.8894864e-01, -2.4655251e-02, -1.4619486e-01, -3.0473773e02],
                    [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
                ]
            ),
        }
        interpolated_view_matrices = self.keyframe_camera.interpolate_view_matrices(
            view_matrices_dict
        )

        assert len(view_matrices_dict) == 3
        assert len(interpolated_view_matrices) == 200
        assert np.allclose(view_matrices_dict[0], interpolated_view_matrices[0])

    def test_get_view_matrix(self):
        """
        Test the get_view_matrix method.
        """
        view_matrix = np.array(
            [
                [1.0000000e00, -1.8031891e-08, 5.3372737e-09, -8.3287910e01],
                [-2.0071333e-08, 9.9999309e-01, -3.7090916e-03, -8.6779434e01],
                [1.0577577e-07, 3.7090357e-03, 9.9999303e-01, -2.3617999e02],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
        print(self.keyframe_camera.get_view_matrix(0))
        self.assertTrue(
            np.allclose(
                self.keyframe_camera.get_view_matrix(0),
                view_matrix,
                rtol=1e-1,
                atol=1e-1,
            )
        )
