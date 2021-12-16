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
from .mesh import Mesh
import open3d as o3d
import numpy as np


@dataclass
class Sphere(Mesh):
    """
    A class to produce sphere meshes.
    """
    radius: int
    resolution: int = 10

    def create_mesh(self, starting_position: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Create a mesh object defined by the dataclass.

        Parameters
        ----------
        starting_position : np.ndarray
                Starting position of the mesh.

        Returns
        -------
        mesh : o3d.geometry.TriangleMesh
        """
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.radius, resolution=self.resolution
        )
        sphere.compute_vertex_normals()
        sphere.translate(starting_position.astype(float))
        sphere.paint_uniform_color(self.colour)

        return sphere

