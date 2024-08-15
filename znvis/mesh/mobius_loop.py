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
Create a mobius loop mesh
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from znvis.transformations.rotation_matrices import rotation_matrix

from znvis.mesh import Mesh


@dataclass
class MobiusLoop(Mesh):
    """
    A class to produce mobius loop meshes.

    Attributes
    ----------
    length_split : int
            The number of segments along the Mobius strip.
    width_split : int
            The number of segments along the width of the Mobius strip.
    twists : int
            Number of twists of the Mobius strip.
    radius : float
            Radius of the Mobius strip.
    flatness : float
            Controls the flatness/height of the Mobius strip.
    width : float
            Width of the Mobius strip.
    scale : float
            Scale the complete Mobius strip.
    """

    length_split: int = 70
    width_split: int = 15
    twists: int = 1
    radius: float = 1
    flatness: float = 1
    width: float = 1
    scale: float = 1

    def create_mesh(self) -> o3d.geometry.TriangleMesh:

        return o3d.geometry.TriangleMesh.create_mobius(
            length_split=self.length_split,
            width_split=self.width_split,
            twists=self.twists,
            raidus=self.radius, # typo in open3d
            flatness=self.flatness,
            width=self.width,
            scale=self.scale,
        )
