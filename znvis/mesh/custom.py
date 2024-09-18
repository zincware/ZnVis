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
Create a custom mesh
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from znvis.mesh import Mesh
from znvis.transformations.rotation_matrices import rotation_matrix


@dataclass
class CustomMesh(Mesh):
    """
    A class to produce custom meshes. Custom meshes are special and need to override
    the create_mesh method.

    Attributes
    ----------
    file : str
            Path to mesh file.
    """

    file: str = None
    scale: float = 1.0

    def instantiate_mesh(
        self, starting_position: np.ndarray, starting_orientation: np.ndarray = None
    ) -> o3d.geometry.TriangleMesh:
        """
        Create a mesh object defined by the dataclass.
        """
        mesh = o3d.io.read_triangle_mesh(self.file)
        mesh.compute_vertex_normals()
        mesh.scale(self.scale, center=mesh.get_center())
        mesh.translate(starting_position.astype(float))
        if starting_orientation is not None:
            matrix = rotation_matrix(self.base_direction, starting_orientation)
            mesh.rotate(matrix)

        return mesh
