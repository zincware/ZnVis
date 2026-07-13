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
Module for the ParticleFollowingCamera class.
"""

import numpy as np

from znvis.cameras.base_camera import BaseCamera


class ParticleFollowingCamera(BaseCamera):
    """
    A class to provide a camera that follows a given particle in a simulation.
    Three possible cases are considered:
    1. The camera follows the particle with a fixed distance vector.
       Activated by passing a single vector of shape (3,) to the
       camera_particle_vector parameter.
    2. The camera follows the particle with a dynamic distance vector given by
       the user. Activated by passing an array of shape (n_frames, 3) to the
       camera_particle_vector parameter.
    3. The camera follows the particle while always looking at the same direction
       as the particle. Activated by passing the particle_directions parameter.
    """

    def __init__(
        self,
        particle_positions: np.ndarray,
        particle_directions: np.ndarray = None,
        camera_particle_vector=None,
        camera_up_vector=None,
    ) -> None:
        """
        Initialize the ParticleFollowingCamera object.
        Parameters
        ----------
        particle_positions : np.ndarray shape=(n_frames, 3)
                The positions of the particles in the simulation.
        particle_directions : np.ndarray shape=(n_frames, 3), optional
                The directions of the particles in the simulation.
                If None, the resulting camera will use the camera_particle_vector
                as an orientation.
                This will supercede the camera_particle_vector input.
        camera_particle_vector : np.ndarray shape=(3,) or (n_frames, 3)
                The distance vector(s) between the camera and the particle.
                If the camera should follow the particle, this vector's length
                is used to determine the distance from the particle.
                If the camera particle vector is (n_frames, 3), the camera will
                follow the particle dynamically with the given distance vector.
                If the camera particle vector is (3,), the camera will follow
                the particle with the given distance vector for all frames.
        camera_up_vector : np.ndarray shape=(3,) or (n_frames, 3)
                The up vector(s) of the camera.
                This vector defines the orientation of the camera.
                It should be a unit vector.
                The camera will be rotated around the particle to look at it.
                Default is [0,1,0], which is the y-axis.

        """

        self.particle_positions = particle_positions
        if camera_particle_vector is None:
            camera_particle_vector = np.array([0, 0, 20])
        if camera_up_vector is None:
            camera_up_vector = np.array([0, 1, 0])
        if particle_directions is not None:
            self.particle_directions = particle_directions
        else:
            self.particle_directions = None

        self.camera_particle_vector = camera_particle_vector
        self.camera_up_vector = camera_up_vector
        self.view_matrix = self.get_view_matrix(0)

    def get_view_matrix(self, frame_index: int) -> np.ndarray:
        """
        Provides the view matrix for the given frame index, where
        the camera is shifted by the camera_particle_vector from
        the particle position and looks at the particle.
        Parameters
        ----------
        frame_index : int,
            The index of the frame for which to get the view matrix.

        Returns
        -------
        view_matrix: np.ndarray
            The view matrix for the given frame index.
        """
        # can support (3,) shape and (n_frames, 3)
        up_vec = (
            self.camera_up_vector[frame_index]
            if getattr(self.camera_up_vector, "ndim", 1) == 2
            else self.camera_up_vector
        )
        up_norm = np.linalg.norm(up_vec)
        if up_norm == 0.0:
            raise ValueError(
                '"camera_up_vector" must be non-zero. '
                f"Error occured for frame index {frame_index}."
            )
        up = up_vec / up_norm

        center = self.particle_positions[frame_index]

        if self.particle_directions is None:
            if self.camera_particle_vector.ndim == 1:
                eye = center + self.camera_particle_vector
            elif self.camera_particle_vector.ndim == 2:
                eye = center + self.camera_particle_vector[frame_index]
            else:
                raise ValueError(
                    '"camera_particle_vector" must be shape (3,) or (n_frames, 3).'
                )
        else:
            camera_particle_vector = (
                self.camera_particle_vector[frame_index]
                if self.camera_particle_vector.ndim == 2
                else self.camera_particle_vector
            )
            distance_from_particle = float(np.linalg.norm(camera_particle_vector))
            if distance_from_particle == 0:
                distance_from_particle = 1e-8  # avoid eye == center
            eye = (
                center - self.particle_directions[frame_index] * distance_from_particle
            )

        view_matrix = self.look_at(center, eye, up)
        self.view_matrix = view_matrix
        return view_matrix
