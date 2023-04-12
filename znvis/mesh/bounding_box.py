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
Create  bounding box mesh.
"""
from dataclasses import dataclass

import numpy as np
import open3d as o3d

from .mesh import Mesh


@dataclass
class BoundingBox(Mesh):
    """
    A class to produce sphere meshes.

    Attributes
    ----------
    center : np.ndarray shape=(3,)
            Center of the bounding box
    rotation_matrix : np.ndarray shape=(3, 3)
            Rotation matrix of the bounding box.
    box_size : np.ndarray shape=(3,)
            Size of the bounding box.
    scale : np.ndarray shape=(3,)
            Scale of the bounding box vectors.
    """

    center: np.ndarray = np.array([0, 0, 0])
    box_size: np.ndarray = np.array([1.0, 1.0, 1.0])
    rotation_matrix: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    scale: np.ndarray = np.array([1.0, 1.0, 1.0])

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
        bounding_box = o3d.geometry.OrientedBoundingBox(
            self.center, self.rotation_matrix, self.box_size
        )
        # bounding_box = bounding_box.get_axis_aligned_bounding_box()
        # box = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(
        #     bounding_box, scale=self.scale, create_uv_map=False
        # )
        # box = o3d.geometry.TriangleMesh.create_from
        # box.compute_vertex_normals()
        # # sphere.translate(starting_position.astype(float))
        # # if starting_orientation is not None:
        # #     matrix = rotation_matrix(np.array([0, 0, 1]), starting_orientation)
        # #     sphere.rotate(matrix)
        # box.paint_uniform_color((1, 0, 0))

        return bounding_box
