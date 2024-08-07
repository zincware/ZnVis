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
Run unit tests on the sphere module.
"""

import unittest

import numpy as np
import open3d as o3d

from znvis import Material
from znvis.mesh.cylinder import Cylinder


class TestCylinder(unittest.TestCase):
    """
    A test class for the Particle class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a sphere instance for testing
        """
        cls.material = Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)
        cls.cylinder = Cylinder(
            material=cls.material,
            radius=10,
            height=20,
            split=6,
            resolution=10,
        )

    def test_instantiation(self):
        """
        Test the instantiation of a Mesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        self.assertEqual(self.cylinder.material, self.material)
        self.assertEqual(self.cylinder.radius, 10)
        self.assertEqual(self.cylinder.height, 20)
        self.assertEqual(self.cylinder.split, 6)
        self.assertEqual(self.cylinder.resolution, 10)

    def test_build_cylinder(self):
        """
        Test the construction of a sphere mesh.

        Returns
        -------
        Test if a sphere mesh is constructed correctly.
        """
        cylinder = self.cylinder.instantiate_mesh(
            starting_position=np.array([1, 1, 1]),
            starting_orientation=np.array([1, 1, 1]),
        )
        self.assertEqual(cylinder.has_vertex_normals(), True)
        self.assertEqual(type(cylinder), o3d.geometry.TriangleMesh)
        np.testing.assert_almost_equal(cylinder.get_center(), [1.0, 1.0, 1.0])
