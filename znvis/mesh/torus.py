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
Create a torus mesh
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from znvis.transformations.rotation_matrices import rotation_matrix

from znvis.mesh import Mesh


@dataclass
class Torus(Mesh):
    """
    A class to produce torus meshes.

    Attributes
    ----------
    torus_radius : float
            The radius from the center of the torus to the center of the tube.
    tube_radius : float
            The radius of the torus tube.
    tubular_resolution : int
            The number of segments along the tubular direction.    
    radial_resolution : int
            The number of segments along the radial direction.
    """

    torus_radius: float = 1.0
    tube_radius: float = 0.3
    tubular_resolution: int = 20
    radial_resolution: int = 30

    def create_mesh(self) -> o3d.geometry.TriangleMesh:

        return o3d.geometry.TriangleMesh.create_torus(
            torus_radius=self.torus_radius,
            tube_radius=self.tube_radius,
            tubular_resolution=self.tubular_resolution,
            radial_resolution=self.radial_resolution,
        )

