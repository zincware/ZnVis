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
Create a box mesh
"""

from dataclasses import dataclass

import open3d as o3d

from znvis.mesh import Mesh


@dataclass
class Box(Mesh):
    """
    A class to produce box meshes.

    Attributes
    ----------
    width : float
            Width of the box.
    height : float
            Height of the box.
    depth : float
            Depth of the box.
    """

    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0

    def create_mesh(self) -> o3d.geometry.TriangleMesh:

        return o3d.geometry.TriangleMesh.create_box(
            width=self.width, height=self.height, depth=self.depth
        )
