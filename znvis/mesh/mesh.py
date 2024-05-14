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
Module for the mesh parent class.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from znvis.material.material import Material


@dataclass
class Mesh:
    """
    Parent class for the ZnVis meshes.

    Attributes
    ----------
    material : Material
            A ZnVis material class.
    """

    material: Material = Material()
    base_direction = np.array([1, 0, 0])

    def __post_init__(self):
        """
        Post init function to create materials.
        """
        material = rendering.MaterialRecord()
        material.base_color = np.hstack((self.material.colour, self.material.alpha))
        material.shader = "defaultLitTransparency"
        material.base_metallic = self.material.metallic
        material.base_roughness = self.material.roughness
        material.base_reflectance = self.material.reflectance
        material.base_anisotropy = self.material.anisotropy

        self.o3d_material = material

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
        raise NotImplementedError("Implemented in child class.")
