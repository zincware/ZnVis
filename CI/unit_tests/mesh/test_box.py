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
Run unit tests on the box module.
"""

import unittest

import numpy as np
import open3d as o3d

from znvis import Material
from znvis.mesh.box import Box


class TestBox(unittest.TestCase):
    """
    A test class for the Box class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a box instance for testing
        """
        cls.material = Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)
        cls.box = Box(
            material=cls.material,
            width=1.0,
            height=5.0,
            depth=10.0
        )

    def test_instantiation(self):
        """
        Test the instantiation of a Mesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        self.assertEqual(self.box.material, self.material)
        self.assertEqual(self.box.width, 1.0)
        self.assertEqual(self.box.height, 5.0)
        self.assertEqual(self.box.depth, 10.0)

    def test_build_box(self):
        """
        Test the construction of a box mesh.

        Returns
        -------
        Test if a box mesh is constructed correctly.
        """
        box = self.box.instantiate_mesh(
            starting_position=np.array([1, 1, 1]),
            starting_orientation=np.array([1, 1, 1]),
        )
        self.assertEqual(box.has_vertex_normals(), True)
        self.assertEqual(type(box), o3d.geometry.TriangleMesh)
        np.testing.assert_almost_equal(box.get_center(), [1.5, 3.5, 6.0])
