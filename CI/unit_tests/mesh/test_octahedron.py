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
Run unit tests on the octahedron module.
"""

import unittest

import numpy as np
import open3d as o3d

from znvis import Material
from znvis.mesh.octahedron import Octahedron


class TestOctahedron(unittest.TestCase):
    """
    A test class for the Octahedron class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a octahedron instance for testing
        """
        cls.material = Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)
        cls.octahedron = Octahedron(
            material=cls.material,
            radius=1.0,
        )

    def test_instantiation(self):
        """
        Test the instantiation of a Mesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        self.assertEqual(self.octahedron.material, self.material)
        self.assertEqual(self.octahedron.radius, 1.0)

    def test_build_octahedron(self):
        """
        Test the construction of a octahedron mesh.

        Returns
        -------
        Test if a octahedron mesh is constructed correctly.
        """
        octahedron = self.octahedron.instantiate_mesh(
            starting_position=np.array([1, 1, 1]),
            starting_orientation=np.array([1, 1, 1]),
        )
        self.assertEqual(octahedron.has_vertex_normals(), True)
        self.assertEqual(type(octahedron), o3d.geometry.TriangleMesh)
        np.testing.assert_almost_equal(octahedron.get_center(), [1, 1, 1])
