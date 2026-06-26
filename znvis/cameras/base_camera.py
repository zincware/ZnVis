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
Module for the BaseCamera parent class.
"""

import numpy as np


class BaseCamera:
    def __init__(
        self,
        center: np.ndarray = None,
        eye: np.ndarray = None,
        up: np.ndarray = None,
        view_matrix: np.ndarray = None,
    ) -> None:
        """
        This is a base class for cameras. The specific implementation of how
        the camera is initialized should be provided in a child class.

        Parameters
        ----------
        center : np.ndarray shape=(3,)
                The spatial point the camera looks at.
        eye : np.ndarray shape=(3,)
                The spatial position of the camera.
        up : np.ndarray shape=(3,)
                The up vector of the camera.
        ↑ (up)
        |
        |       camera looks from the eye to the center
        |
        ●------>●
        eye   center
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera. Only needed,
                if the camera is not defined by the center, eye and up vectors.

        Returns
        -------
        None
        """
        raise NotImplementedError("This method should be implemented in a child class.")

    @staticmethod
    def get_view_matrix_from_particle_positions(
        particle_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates a good initial view matrix for the camera based on
        the particle positions.
        The function calculates the center of the particle positions and
        sets the camera to look at the center of the particles.
        The camera is positioned at a distance of 2 times the maximum distance
        of the particles from the center.

        Parameters
        ----------
        particle_positions : np.ndarray shape=(n_particles, 3)
                The positions of the particles at a given frame.
        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        """
        center = np.mean(particle_positions, axis=0)
        max_distance = np.max(np.linalg.norm(particle_positions - center, axis=1)) * 2
        eye = center + np.array([0, 0, max_distance])
        up = np.array([0, 1, 0])

        return BaseCamera.look_at(center, eye, up)

    @staticmethod
    def get_view_matrix_from_box_size(box_size: np.ndarray) -> np.ndarray:
        """
        Calculates a good initial view matrix for the camera based on the box size.
        The function sets the camera to look at the center of the box.
        The camera is positioned at a distance of 2 times the maximum distance
        of the box size from the center.

        Parameters
        ----------
        box_size : np.ndarray shape=(3,)
                The size of the box.

        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        """
        center = box_size / 2
        max_distance = np.max(box_size) * 1.1
        eye = center + np.array([0, 0, max_distance])
        up = np.array([0, 1, 0])
        return BaseCamera.look_at(center, eye, up)

    @staticmethod
    def get_minimal_view_matrix(
        box_size: np.ndarray, renderer_resolution=None
    ) -> np.ndarray:
        """
        Calculates a minimal view matrix for the camera based on the box size.

        Parameters
        ----------
        box_size : np.ndarray shape=(3,)
                The size of the box.
        renderer_resolution : np.ndarray, optional
                The chosen renderer resolution to calculate the aspect ratio.
        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        """
        # Default value in Open3D
        fov = 60
        if renderer_resolution is None:
            renderer_resolution = np.array([4, 3])
        else:
            renderer_resolution = np.asarray(renderer_resolution)
        aspect_ratio = renderer_resolution[0] / renderer_resolution[1]
        dx, dy, dz = box_size
        center = np.array([dx / 2, dy / 2, 0])
        up = np.array([0, 1, 0])

        tan_fov_y = np.tan(np.radians(fov) / 2)
        tan_fov_x = tan_fov_y * aspect_ratio

        dist_y = dy / (2 * tan_fov_y)
        dist_x = dx / (2 * tan_fov_x)

        camera_distance = max(dist_x, dist_y) + dz / 2

        eye = center + np.array([0, 0, camera_distance])
        return BaseCamera.look_at(center, eye, up)

    @staticmethod
    def look_at(center: np.ndarray, eye: np.ndarray, up: np.ndarray) -> np.ndarray:
        """
        Calculates a view matrix out of the center,
        eye and up vectors adapted to open3D conventions.

        Parameters
        ----------
        center : np.ndarray shape=(3,)
                The center of the camera.
        eye : np.ndarray shape=(3,)
                The eye of the camera.
        up : np.ndarray shape=(3,)
                The up vector of the camera.

        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        """
        z = eye - center
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-7:
            z = np.array([0.0, 0.0, 1.0])
        else:
            z = z / np.linalg.norm(z)

        x = np.cross(up, z)
        x_norm = np.linalg.norm(x)

        if x_norm < 1e-7:
            x = np.array([1.0, 0.0, 0.0])
        else:
            x = x / np.linalg.norm(x)

        y = np.cross(z, x)

        view_matrix = np.array(
            [
                [x[0], x[1], x[2], -np.dot(x, eye)],
                [y[0], y[1], y[2], -np.dot(y, eye)],
                [z[0], z[1], z[2], -np.dot(z, eye)],
                [0, 0, 0, 1],
            ]
        )

        return view_matrix

    def get_view_matrix(self, frame_index: int) -> np.ndarray:
        """
        Get the current view matrix of the camera.

        Parameters
        ----------
        frame_index : int
                The frame index of the view matrix that should be returned.

        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        """
        return self.view_matrix

    def set_view_matrix(self, view_matrix: np.ndarray) -> None:
        """
        Sets the view matrix of the camera.

        Parameters
        ----------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.

        Returns
        -------
        None
        """
        self.view_matrix = view_matrix

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
        # NOTE: Here the exact distance is missing,
        # hence center is only an approximation.
        # The direction will be correct.
        center = eye + inv_rotation @ np.array([0, 0, -1])
        up = inv_rotation @ np.array([0, 1, 0])

        return center, eye, up

    def verify_camera_setup_for_rendering(self):
        """
        Ensures the camera is ready for rendering and has a valid view matrix.
        If no view matrix is set, assigns a default view matrix.
        Returns True if the camera is ready, False otherwise.
        """
        if (
            self.view_matrix is not None
            and isinstance(self.view_matrix, np.ndarray)
            and self.view_matrix.shape == (4, 4)
        ):
            return True
        else:
            print(
                "Couldn't find a valid view matrix in the camera object. "
                "Using default."
            )
            self.view_matrix = np.array(
                [[1, 0, 0, -100], [0, 1, 0, -90], [0, 0, 1, -230], [0, 0, 0, 1]]
            )
            return False
