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
Create a cone mesh
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from znvis.transformations.rotation_matrices import rotation_matrix

from .mesh import Mesh


@dataclass
class Cone(Mesh):
    """
    A class to produce cone meshes.

    Attributes
    ----------
    radius : float
            Radius of the cone.
    height : float
            Height of the cone.
    resolution : int
            Resolution of the cone.
    split : int
            The height will be split into this many segments.
    """

    radius: float = 1.0
    height: float = 2.0
    resolution: int = 20
    split: int = 1

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
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius = self.radius
        )
        cone.compute_vertex_normals()
        cone.translate(starting_position.astype(float))
        if starting_orientation is not None:
            matrix = rotation_matrix(self.base_direction, starting_orientation)
            cone.rotate(matrix)

        return cone
