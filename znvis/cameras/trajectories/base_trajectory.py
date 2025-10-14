"""
ZnVis: A Zincwarecode package.
License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0
Copyright Contributors to the Zincwarecode Project.
Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/
Citation
--------
If you use this module please cite us with:

Summary
-------
Module for the BaseCameraTrajectory parent class.
"""

import numpy as np


class BaseTrajectory:
    def __init__(self, total_frames: int):
        """
        Base class for camera trajectories.

        Parameters
        ----------
        total_frames : int
            The number of frames in which the trajectory should be recorded.
        """
        self.total_frames = total_frames

    def get_center_eye_up(self, frame_index: int) -> tuple:
        """
        Provides the view matrix for the given frame index.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @staticmethod
    def get_center_eye_up_from_view_matrix(view_matrix: np.ndarray) -> tuple:
        """
        Extracts center, up and eye vector from the view matrix.
        NOTE: Center cannot be extracted accurately from the view matrix
        in the sense that the information about the exact center is lost due to
        normalization of the forward vector in the process of creating the view matrix.
        Therefore, this is only an approximation for the center!
        Parameters
        ----------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        Returns
        -------
        center : np.ndarray
            The center of the view matrix.
        eye : np.ndarray
            The eye of the view matrix.
        up : np.ndarray
            The up vector of the view matrix.
        """

        rotation = view_matrix[:3, :3]
        translation = view_matrix[:3, 3]

        inv_rotation = rotation.T
        inv_translation = -inv_rotation @ translation

        eye = inv_translation
        center = eye + inv_rotation @ np.array([0, 0, -1])
        up = inv_rotation @ np.array([0, 1, 0])

        return center, eye, up
