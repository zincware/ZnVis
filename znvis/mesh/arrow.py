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
Create a sphere mesh.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from znvis.transformations.rotation_matrices import rotation_matrix

from znvis.mesh import Mesh


@dataclass
class Arrow(Mesh):
    """
    A class to produce arrow meshes. Arrow meshes are a special case and need to
    overwrite the create_mesh object of the parent mesh class.

    Attributes
    ----------
    scale : float
            Scale of the arrow
    resolution : int
            Resolution of the mesh.
    """
    scale: float = 1.0
    resolution: int = 10

    def create_mesh(
        self, starting_position: np.ndarray, starting_orientation: np.ndarray = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Create and correctly orient an arrow mesh. Overwrites the parent class
        """
        mesh = self.create_mesh_object(starting_orientation)
        mesh.compute_vertex_normals()
        if starting_orientation is not None:
            matrix = rotation_matrix(np.array([0, 0, 1]), starting_orientation)
            mesh.rotate(matrix, center=(0, 0, 0))

        # Translate the arrow to the starting position and center the origin
        mesh.translate(starting_position.astype(float))

        return mesh

    def create_mesh_object(self, direction: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Creates an arrow mesh object.
        """
        direction_length = np.linalg.norm(direction)

        cylinder_radius = 0.06 * direction_length * self.scale
        cylinder_height = 0.85 * direction_length * self.scale
        cone_radius = 0.15 * direction_length * self.scale
        cone_height = 0.15 * direction_length * self.scale

        return o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius, 
            cylinder_height=cylinder_height, 
            cone_radius=cone_radius, 
            cone_height=cone_height,
            resolution=self.resolution
        )
