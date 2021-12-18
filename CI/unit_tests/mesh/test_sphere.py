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

from znvis.mesh.sphere import Sphere


class TestSphere(unittest.TestCase):
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
        cls.sphere = Sphere(colour=np.array([0.7, 0.3, 0.1]), radius=10, resolution=10)

    def test_instantiation(self):
        """
        Test the instantiation of a Mesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        np.testing.assert_array_equal(self.sphere.colour, np.array([0.7, 0.3, 0.1]))
        self.assertEqual(self.sphere.radius, 10)
        self.assertEqual(self.sphere.resolution, 10)

    def test_build_sphere(self):
        """
        Test the construction of a sphere mesh.

        Returns
        -------
        Test if a sphere mesh is constructed correctly.
        """
        sphere = self.sphere.create_mesh(starting_position=np.array([1, 1, 1]))
        self.assertEqual(sphere.has_vertex_normals(), True)
        self.assertEqual(sphere.has_vertex_colors(), True)
        self.assertEqual(type(sphere), o3d.geometry.TriangleMesh)
        np.testing.assert_almost_equal(sphere.get_center(), [1., 1., 1.])
