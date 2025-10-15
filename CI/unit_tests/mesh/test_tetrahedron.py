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
Run unit tests on the tetrahedron module.
"""

import unittest

import numpy as np
import open3d as o3d

from znvis import Material
from znvis.mesh.tetrahedron import Tetrahedron


class TestTetrahedron(unittest.TestCase):
    """
    A test class for the Tetrahedron class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a tetrahedron instance for testing
        """
        cls.material = Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)
        cls.tetrahedron = Tetrahedron(
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
        self.assertEqual(self.tetrahedron.material, self.material)
        self.assertEqual(self.tetrahedron.radius, 1.0)

    def test_build_tetrahedron(self):
        """
        Test the construction of a tetrahedron mesh.

        Returns
        -------
        Test if a tetrahedron mesh is constructed correctly.
        """
        tetrahedron = self.tetrahedron.instantiate_mesh(
            starting_position=np.array([1, 1, 1]),
            starting_orientation=np.array([1, 1, 1]),
        )
        self.assertEqual(tetrahedron.has_vertex_normals(), True)
        self.assertEqual(type(tetrahedron), o3d.geometry.TriangleMesh)
        np.testing.assert_almost_equal(tetrahedron.get_center(), [1, 1, 1])
