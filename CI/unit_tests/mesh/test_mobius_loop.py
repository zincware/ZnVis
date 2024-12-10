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
Run unit tests on the mobiusloop module.
"""

import unittest

import numpy as np
import open3d as o3d

from znvis import Material
from znvis.mesh.mobius_loop import MobiusLoop


class TestMobius(unittest.TestCase):
    """
    A test class for the MobiusLoop class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a mobius loop instance for testing
        """
        cls.material = Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)
        cls.mobius_loop = MobiusLoop(
            material=cls.material,
            length_split=70,
            width_split=15,
            twists=1,
            radius=1,
            flatness=1.0,
            width=1.0,
            scale=1.0,
        )

    def test_instantiation(self):
        """
        Test the instantiation of a Mesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        self.assertEqual(self.mobius_loop.material, self.material)
        self.assertEqual(self.mobius_loop.length_split, 70)
        self.assertEqual(self.mobius_loop.width_split, 15)
        self.assertEqual(self.mobius_loop.twists, 1)
        self.assertEqual(self.mobius_loop.radius, 1)
        self.assertEqual(self.mobius_loop.flatness, 1.0)
        self.assertEqual(self.mobius_loop.width, 1.0)
        self.assertEqual(self.mobius_loop.scale, 1.0)

    def test_build_mobiusloop(self):
        """
        Test the construction of a mobiusloop mesh.

        Returns
        -------
        Test if a mobiusloop mesh is constructed correctly.
        """
        mobius_loop = self.mobius_loop.instantiate_mesh(
            starting_position=np.array([1, 1, 1]),
            starting_orientation=np.array([1, 1, 1]),
        )
        self.assertEqual(mobius_loop.has_vertex_normals(), True)
        self.assertEqual(type(mobius_loop), o3d.geometry.TriangleMesh)
        np.testing.assert_almost_equal(mobius_loop.get_center(), [1.0, 1.0, 1.0])