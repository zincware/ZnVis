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
Module for the particle parent class
"""
from dataclasses import dataclass

import numpy as np

from znvis.mesh import Mesh
from znvis.transformations.rotation_matrices import rotation_matrix


@dataclass
class Particle:
    """
    Parent class for a ZnVis particle.

    Attributes
    ----------
    name : str
            Name of the particle
    mesh : Mesh
            Mesh to use e.g. sphere
    position : np.ndarray
            Position tensor of the shape (n_confs, n_particles, n_dims)
    velocity : np.ndarray
            Velocity tensor of the shape (n_confs, n_particles, n_dims)
    force : np.ndarray
            Force tensor of the shape (n_confs, n_particles, n_dims)
    director: np.ndarray
            Director tensor of the shape (n_confs, n_particles, n_dims)
    mesh_dict : dict
            Mesh dict to store names of meshes and the mesh objects. e.g.
            {'name_1': TriangleMesh, 'name_2': TraingleMesh, ...}
    """

    name: str
    mesh: Mesh = None
    position: np.ndarray = None
    velocity: np.ndarray = None
    force: np.ndarray = None
    director: np.ndarray = None
    mesh_dict: dict = None

    def construct_mesh_dict(self):
        """
        Constructor the mesh dict for the class.

        Returns
        -------
        Updates the class attributes mesh_dict

        Notes
        -----
        #TODO allow for no position data.
        """
        self.mesh_dict = {}
        try:
            n_particles = int(self.position.shape[1])
        except ValueError:
            raise ValueError("There is no data for these particles.")

        for i in range(n_particles):
            if self.director is not None:
                self.mesh_dict[f"{self.name}_{i}"] = self.mesh.create_mesh(
                    self.position[0][i], starting_orientation=self.director[0][i]
                )
            else:
                self.mesh_dict[f"{self.name}_{i}"] = self.mesh.create_mesh(
                    self.position[0][i],
                )

    def update_position_data(self, step: int):
        """
        Update the positions of each particle.

        Parameters
        ----------
        step : int
                Step to update to.

        Returns
        -------
        Updates the position of the mesh in the mesh_dict

        Notes
        -----
        TODO: Allow for no position data.
        """
        for i, item in enumerate(self.mesh_dict):
            self.mesh_dict[item].translate(self.position[step][i], relative=False)
            if self.director is not None:
                if step == 0:
                    current = self.director[-1][i]
                    matrix = rotation_matrix(current, self.director[step][i])
                    self.mesh_dict[item].rotate(matrix)
                else:
                    current = self.director[step - 1][i]
                    matrix = rotation_matrix(current, self.director[step][i])
                    self.mesh_dict[item].rotate(matrix)
