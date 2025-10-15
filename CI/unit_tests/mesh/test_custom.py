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
Run unit tests on the CustomMesh module.
"""

import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

from znvis import Material
from znvis.mesh.custom import CustomMesh


class TestCustomMesh(unittest.TestCase):
    """
    A test class for the Custom class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a custom instance for testing
        """
        cls.material = Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)

        project_root = Path(__file__).resolve().parents[2]
        obj_file_path = project_root / "test_files" / "red_blood_particle.obj"

        cls.custom = CustomMesh(
            material=cls.material,
            file=str(obj_file_path),
            scale=1.0,
        )

    def test_instantiation(self):
        """
        Test the instantiation of a CustomMesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        self.assertEqual(self.custom.material, self.material)
        self.assertEqual(self.custom.scale, 1.0)

    def test_build_custom(self):
        """
        Test the importing and construction of a custom mesh.

        Returns
        -------
        Test if a custom mesh is constructed correctly.
        """
        custom = self.custom.instantiate_mesh(
            starting_position=np.array([2, 2, 2]),
            starting_orientation=np.array([1, 1, 1]),
        )
        self.assertEqual(custom.has_vertex_normals(), True)
        self.assertEqual(type(custom), o3d.geometry.TriangleMesh)
        # THIS IS COMPLETELY DEPENDENT ON THE FILE PROVIDED...

        np.testing.assert_almost_equal(
            custom.get_center(), [1.9000875, 2.0035554, 1.9946454]
        )
