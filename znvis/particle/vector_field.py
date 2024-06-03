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

import typing
from dataclasses import dataclass

import numpy as np
from rich.progress import track

from znvis.mesh.arrow import Arrow


@dataclass
class VectorField:
    """
    A class to represent a vector field.

    Attributes
    ----------
    name : str
            Name of the vector field
    mesh : Mesh
            Mesh to use 
    position : np.ndarray
            Position tensor of the shape (n_steps, n_vectors, n_dims)
    direction : np.ndarray
            Direction tensor of the shape (n_steps, n_vectors, n_dims)
    mesh_list : list
            A list of mesh objects, one for each time step.
    static : bool (default=False)
            If true, only render the mesh once at initialization. Be careful
            as this changes the shape of the required position and direction 
            to (n_particles, n_dims) 
    smoothing : bool (default=False)
            If true, apply smoothing to each mesh object as it is rendered.
            This will slow down the initial construction of the mesh objects
            but not the deployment.
    """

    name: str
    mesh: Arrow = None # Should be an instance of the Arrow class
    position: np.ndarray = None
    direction: np.ndarray = None
    mesh_list: typing.List[Arrow] = None
    static: bool = False
    smoothing: bool = False

    def _create_mesh(self, position: np.ndarray, direction: np.ndarray):
        """
        Create a mesh object for the vector field.

        Parameters
        ----------
        position : np.ndarray
                Position of the arrow
        direction : np.ndarray
                Direction of the arrow

        Returns
        -------
        mesh : o3d.geometry.TriangleMesh
                A mesh object
        """
        
        mesh = self.mesh.create_mesh(position, direction)
        if self.smoothing:
            return mesh.filter_smooth_taubin(100)
        else:
            return mesh

    def construct_mesh_list(self):
        """
        Constructor the mesh list for the class.

        The mesh list is a list of mesh objects for each
        time step in the parsed trajectory.

        Returns
        -------
        Updates the class attributes mesh_list
        """
        self.mesh_list = []
        try:
            if not self.static: 
                n_particles = int(self.position.shape[1])
                n_time_steps = int(self.position.shape[0])
            else:
                n_particles = int(self.position.shape[0])
                n_time_steps = 1
                self.position = self.position[np.newaxis, :, :]
                self.direction = self.direction[np.newaxis, :, :]

        except ValueError:
            raise ValueError("There is no data for this vector field.")

        for i in track(range(n_time_steps), description=f"Building {self.name} Mesh"):
            for j in range(n_particles):
                if j == 0:
                    mesh = self._create_mesh(self.position[i][j], self.direction[i][j])
                else:
                    mesh += self._create_mesh(self.position[i][j], self.direction[i][j])
            self.mesh_list.append(mesh)
