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

from .mesh import Mesh


@dataclass
class Arrow(Mesh):
    """
    A class to produce arrow meshes.

    Attributes
    ----------
    scale : float
            Scale of the arrow
    """
    scale: float = 1.0
    resolution = 10

    def create_mesh(
        self, starting_position: np.ndarray, direction: np.ndarray = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Create a mesh object defined by the dataclass.

        Parameters
        ----------
        starting_position : np.ndarray shape=(3,)
                Starting position of the mesh.
        direction : np.ndarray shape=(3,) (default = None)
                Direction of the mesh.

        Returns
        -------
        mesh : o3d.geometry.TriangleMesh
        """

        direction_length = np.linalg.norm(direction)

        cylinder_radius = 0.06 * direction_length * self.scale
        cylinder_height = 0.85 * direction_length * self.scale
        cone_radius = 0.15 * direction_length * self.scale
        cone_height = 0.15 * direction_length * self.scale

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius, 
            cylinder_height=cylinder_height, 
            cone_radius=cone_radius, 
            cone_height=cone_height,
            resolution=self.resolution
        )

        arrow.compute_vertex_normals()
        matrix = rotation_matrix(np.array([0, 0, 1]), direction)
        arrow.rotate(matrix)

        # Translate the arrow to the starting position and center the origin
        arrow.translate(starting_position.astype(float) + direction * 0.5 * self.scale)

        return arrow