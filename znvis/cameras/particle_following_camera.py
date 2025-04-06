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

from .camera import Camera


class ParticleFollowingCamera(Camera):
    """
    A class to provide a camera that follows a given particle in a simulation.
    The camera is positioned at a fixed distance from the particle and can look
    in the direction of the particle's director. If no director is provided,
    the camera will look in the direction of the particle.
    """

    def __init__(
        self,
        particle_positions: np.ndarray,
        camera_particle_vector: np.ndarray = np.array([0, 0, 20]),
        camera_up_vector: np.ndarray = np.array([0, 1, 0]),
    ) -> None:
        """
        Initialize the ParticleFollowingCamera object.
        Parameters
        ----------
        particle_positions : np.ndarray shape=(n_frames, 3)
                The positions of the particles in the simulation.
        camera_particle_vector : np.ndarray shape=(3,)
                The distance vector between the camera and the particle.
        camera_up_vector : np.ndarray shape=(3,)
                The up vector of the camera.
                This vector defines the orientation of the camera.
                It should be a unit vector.
                The camera will be rotated around the particle to look at it.
                Default is [0,1,0], which is the y-axis.

        """
        self.particle_positions = particle_positions
        self.camera_particle_vector = camera_particle_vector
        self.camera_up_vector = camera_up_vector
        self.view_matrix = self.get_view_matrix(0)

    def get_view_matrix(self, frame_index=None):
        """
        Provides the view matrix for the given frame index, where
        the amera is shifted by the camera_particle_vector from
        the particle position and looks at the particle.
        Parameters
        ----------
        frame_index : int, optional
            The index of the frame for which to get the view matrix, by default None.

        Returns
        -------
        view_matrix: np.ndarray
            The view matrix for the given frame index.
        """
        center = self.particle_positions[frame_index]
        eye = center + self.camera_particle_vector
        up = self.camera_up_vector
        view_matrix = self.look_at(center, eye, up)
        self.view_matrix = view_matrix
        return view_matrix
