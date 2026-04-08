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
    mesh: Arrow = None  # Should be an instance of the Arrow class
    position: typing.Union[np.ndarray, list] = None
    direction: typing.Union[np.ndarray, list] = None
    mesh_list: typing.List[Arrow] = None
    static: bool = False
    smoothing: bool = False

    def _create_mesh(
        self, position: np.ndarray, direction: np.ndarray, time_step: int, index: int
    ):
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

        mesh = self.mesh.instantiate_mesh(position, direction)
        if self.smoothing:
            mesh = mesh.filter_smooth_taubin(100)

        if self.mesh.material.colour.ndim == 3:
            mesh = mesh.paint_uniform_color(
                self.mesh.material.colour[time_step, index, :]
            )

        return mesh

    def construct_mesh_list(self):
        """
        Constructor the mesh list for the class.

        The mesh list is a list of mesh objects for each
        time step in the parsed trajectory. Position and direction data
        can be either numpy arrays of shape (n_confs, n_vectors, n_dims)
        for a fixed vector count, or lists of arrays where each array has
        shape (n_vectors_i, n_dims) to support a variable number of vectors
        per time step.

        Returns
        -------
        Updates the class attributes mesh_list
        """
        self.mesh_list = []

        if self.position is None:
            raise ValueError("Position data cannot be None.")
        if self.direction is None:
            raise ValueError("Director data cannot be None.")

        variable_particle_count = isinstance(self.position, list)

        try:
            if variable_particle_count:
                n_time_steps = len(self.position)
            elif not self.static:
                n_particles = int(self.position.shape[1])
                n_time_steps = int(self.position.shape[0])
            else:
                n_particles = int(self.position.shape[0])
                n_time_steps = 1
                self.position = self.position[np.newaxis, :, :]
                self.direction = self.direction[np.newaxis, :, :]
        except IndexError:
            raise IndexError("The provided data has an incompatible shape.") from None

        if variable_particle_count:
            for pos_arr, dir_arr in zip(self.position, self.direction):
                if np.isnan(pos_arr).any() or np.isnan(dir_arr).any():
                    raise ValueError("The provided data contains NaNs.")
        else:
            if np.isnan(self.position).any() or np.isnan(self.direction).any():
                raise ValueError("The provided data contains NaNs.")

        for i in track(range(n_time_steps), description=f"Building {self.name} Mesh"):
            n_particles_i = (
                len(self.position[i])
                if variable_particle_count
                else n_particles
            )
            mesh = None
            for j in range(n_particles_i):
                if np.max(np.abs(self.direction[i][j])) > 0:
                    new_mesh = self._create_mesh(
                        self.position[i][j], self.direction[i][j], i, j
                    )
                    if mesh is None:
                        mesh = new_mesh
                    else:
                        mesh += new_mesh

            self.mesh_list.append(mesh)
