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
Run unit tests on the torus module.
"""

import unittest

import numpy as np
import open3d as o3d

from znvis import Material
from znvis.mesh.torus import Torus


class TestTorus(unittest.TestCase):
    """
    A test class for the Torus class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a torus instance for testing
        """
        cls.material = Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)
        cls.torus = Torus(
            material=cls.material,
            torus_radius=1.0,
            tube_radius=0.3,
            tubular_resolution=20,
            radial_resolution=30,
        )

    def test_instantiation(self):
        """
        Test the instantiation of a Mesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        self.assertEqual(self.torus.material, self.material)
        self.assertEqual(self.torus.torus_radius, 1.0)
        self.assertEqual(self.torus.tube_radius, 0.3)
        self.assertEqual(self.torus.tubular_resolution, 20)
        self.assertEqual(self.torus.radial_resolution, 30)

    def test_build_torus(self):
        """
        Test the construction of a torus mesh.

        Returns
        -------
        Test if a torus mesh is constructed correctly.
        """
        torus = self.torus.instantiate_mesh(
            starting_position=np.array([1, 1, 1]),
            starting_orientation=np.array([1, 1, 1]),
        )
        self.assertEqual(torus.has_vertex_normals(), True)
        self.assertEqual(type(torus), o3d.geometry.TriangleMesh)
        np.testing.assert_almost_equal(torus.get_center(), [1, 1, 1])
