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
from rich.progress import track

from znvis.mesh import Mesh


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
            A list of mesh objects, one for each time step.
    """

    name: str
    mesh: Mesh = None
    position: np.ndarray = None
    velocity: np.ndarray = None
    force: np.ndarray = None
    director: np.ndarray = None
    mesh_dict: dict = None

    def _create_mesh(self, position, director):
        """
        Create a mesh object for the particle.

        Parameters
        ----------
        position : np.ndarray
                Position of the particle
        director : np.ndarray
                Director of the particle

        Returns
        -------
        mesh : o3d.geometry.TriangleMesh
                A mesh object
        """
        if director is not None:
            mesh = self.mesh.create_mesh(position, starting_orientation=director)
        else:
            mesh = self.mesh.create_mesh(position)
        return mesh

    def construct_mesh_dict(self):
        """
        Constructor the mesh dict for the class.

        The mesh dict is a list of mesh objects for each
        time step in the parsed trajectory.

        Returns
        -------
        Updates the class attributes mesh_dict
        """
        self.mesh_dict = []
        try:
            n_particles = int(self.position.shape[1])
            n_time_steps = int(self.position.shape[0])
        except ValueError:
            raise ValueError("There is no data for these particles.")

        for i in track(range(n_time_steps), description=f"Building {self.name} Mesh"):
            for j in range(n_particles):
                if j == 0:
                    if self.director is not None:
                        mesh = self._create_mesh(
                            self.position[i][j], self.director[i][j]
                        )
                    else:
                        mesh = self._create_mesh(self.position[i][j], None)
                else:
                    if self.director is not None:
                        mesh += self._create_mesh(
                            self.position[i][j], self.director[i][j]
                        )
                    else:
                        mesh += self._create_mesh(self.position[i][j], None)

            self.mesh_dict.append(mesh)
