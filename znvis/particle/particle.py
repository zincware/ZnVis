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
    position : Union(np.ndarray, list of np.ndarray)
            Position tensor of either (n_confs, n_particles, n_dims) or
            list of (n_particles_t, n_dims) with len n_confs
    velocity : Union[np.ndarray, list of np.ndarray]
            Velocity tensor of either (n_confs, n_particles, n_dims) or
            list of (n_particles_t, n_dims) with len n_confs
    force : Union[np.ndarray, list of np.ndarray]
            Force tensor of either (n_confs, n_particles, n_dims) or
            list of (n_particles_t, n_dims) with len n_confs
    director: Union[np.ndarray, list of np.ndarray]
            Director tensor of either (n_confs, n_particles, n_dims) or
            list of (n_particles_t, n_dims) with len n_confs
    mesh_list : list
            A list of mesh objects, one for each time step.
    static : bool (default=False)
            If true, only render the mesh once at initialization. Be careful
            as this changes the shape of the required position and director
            to (n_particles, n_dims).

    smoothing : bool (default=False)
            If true, apply smoothing to each mesh object as it is rendered.
            This will slow down the initial construction of the mesh objects
            but not the deployment.
    """

    name: str
    mesh: Mesh = None
    position: typing.Union[np.ndarray, typing.List[np.ndarray]] = None
    velocity: typing.Union[np.ndarray, typing.List[np.ndarray]] = None
    force: typing.Union[np.ndarray, typing.List[np.ndarray]] = None
    director: typing.Union[np.ndarray, typing.List[np.ndarray]] = None
    mesh_list: typing.List[Mesh] = None
    static: bool = False
    smoothing: bool = False

    def _create_mesh(self, position, director, time_step=None, index=None):
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
            mesh = self.mesh.instantiate_mesh(position, starting_orientation=director)
        else:
            mesh = self.mesh.instantiate_mesh(position)

        if self.smoothing:
            mesh = mesh.filter_smooth_taubin(100)

        if self.mesh.material.colour.ndim == 3:
            mesh.paint_uniform_color(self.mesh.material.colour[time_step, index, :])

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

        # Convert ndarrays into lists
        if isinstance(self.position, np.ndarray):
            if self.position.size == 0:
                raise ValueError(
                    "The provided position array is empty."
                    "Please provide a valid position array."
                )
            if self.director is not None and self.director.size == 0:
                self.director = None
                print(
                    "-------\nWARNING: The provided director array is empty."
                    "Setting to None.\n-------",
                )
            # Static case differentiation
            if not self.static:
                n_time_steps = self.position.shape[0]
                self.position = [self.position[i] for i in range(n_time_steps)]
                if self.director is not None:
                    self.director = [self.director[i] for i in range(n_time_steps)]
            else:
                n_time_steps = 1
                if self.position.ndim == 3:
                    print(
                        "-------\nWARNING: The provided position array has an ",
                        "unexpected shape. Using the first entry as the static array."
                        "\n-------",
                    )
                    self.position = [self.position[0, :, :]]
                elif self.position.ndim == 2:
                    self.position = [self.position]

                if self.director is not None:
                    if self.director.ndim == 3:
                        self.director = [self.director[0, :, :]]
                        print(
                            "-------\nWARNING: The provided director array has an "
                            "unexpected shape. Using the first entry as the static "
                            "array.\n-------",
                        )
                    elif self.director.ndim == 2:
                        self.director = [self.director]
        # List case
        else:
            n_time_steps = len(self.position)
            # Normalize director to list form if provided as ndarray
            if self.director is not None and isinstance(self.director, np.ndarray):
                if self.director.ndim == 3 and self.director.shape[0] == n_time_steps:
                    self.director = [self.director[i] for i in range(n_time_steps)]
                elif self.director.ndim == 2 and n_time_steps == 1:
                    self.director = [self.director]
                else:
                    raise ValueError(
                        "Director shape does not match number of position frames."
                    )

        # Check data for consistency
        if self.position is None:
            raise ValueError("Position data must be not None.")
        for i, position in enumerate(self.position):
            if np.isnan(position).any():
                raise ValueError(
                    f"The provided position data contains at least one "
                    f"NaN value at time step {i}."
                )
        if self.director is not None:
            for i, director in enumerate(self.director):
                if director is not None and np.isnan(director).any():
                    raise ValueError(
                        "The provided director data contains at least one "
                        f"NaN value at time step {i}.",
                    )
        # Create the mesh
        for frame_index in track(
            range(n_time_steps), description=f"Building {self.name} Mesh"
        ):
            frame_pos = self.position[frame_index]
            frame_dir = (
                self.director[frame_index] if self.director is not None else None
            )
            n_particles = frame_pos.shape[0]
            meshes = []
            for particle_index in range(n_particles):
                pos = frame_pos[particle_index]
                dir = frame_dir[particle_index] if frame_dir is not None else None
                mesh = self._create_mesh(pos, dir, frame_index, particle_index)
                meshes.append(mesh)

            # Combine all meshes into one
            if not meshes:
                raise ValueError(f"No particles found at time step {frame_index}.")
            combined_mesh = meshes[0]
            for m in meshes[1:]:
                combined_mesh += m

            self.mesh_list.append(combined_mesh)
