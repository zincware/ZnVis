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
Create a cylinder mesh
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from znvis.transformations.rotation_matrices import rotation_matrix

from .mesh import Mesh


@dataclass
class Cylinder(Mesh):
    """
    A class to produce cylinder meshes.

    Attributes
    ----------
    radius : float
            Radius of the sphere.
    height : float
            Height of the cylinder.
    split : int
            Number of segment the mesh will be split into.
    resolution : int
            Resolution of the sphere.
    """

    radius: float = 1.0
    height: float = 3.0
    split: int = 1
    resolution: int = 10

    def create_mesh(
        self, starting_position: np.ndarray, starting_orientation: np.ndarray = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Create a mesh object defined by the dataclass.

        Parameters
        ----------
        starting_position : np.ndarray shape=(3,)
                Starting position of the mesh.
        starting_orientation : np.ndarray shape=(3,) (default = None)
                Starting orientation of the mesh.

        Returns
        -------
        mesh : o3d.geometry.TriangleMesh
        """
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=self.radius,
            height=self.height,
            split=self.split,
            resolution=self.resolution,
        )
        cylinder.compute_vertex_normals()
        cylinder.translate(starting_position.astype(float))
        if starting_orientation is not None:
            matrix = rotation_matrix(np.array([0, 0, 1]), starting_orientation)
            cylinder.rotate(matrix)
        cylinder.paint_uniform_color(self.colour)

        return cylinder
