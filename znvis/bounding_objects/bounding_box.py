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
Create  bounding box.
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class BoundingBox:
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
    colour : np.ndarray shape=(3,)
    """

    center: np.ndarray = np.array([0, 0, 0])
    box_size: np.ndarray = np.array([1.0, 1.0, 1.0])
    rotation_matrix: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    colour: np.ndarray = np.array([0.0, 0.0, 0.0])

    def __call__(self) -> o3d.geometry.TriangleMesh:
        """
        Create the bounding box.

        Returns
        -------
        bounding_box : o3d.geometry.OrientationBoundingBox
                The bounding box object.
        """
        bounding_box = o3d.geometry.OrientedBoundingBox(
            self.center, self.rotation_matrix, self.box_size
        )
        bounding_box.color = self.colour

        return bounding_box
