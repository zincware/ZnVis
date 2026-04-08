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
Test the Material dataclass.
"""

import unittest

import numpy as np

from znvis.material.material import Material


class TestMaterial(unittest.TestCase):
    """
    A test class for the Material class.
    """

    def test_default_values(self):
        """
        Test that the default values are set correctly.
        """
        material = Material()
        np.testing.assert_array_almost_equal(
            material.colour, np.array([59.0, 53.0, 97.0]) / 255
        )
        self.assertEqual(material.alpha, 1.0)
        self.assertEqual(material.roughness, 0.5)
        self.assertEqual(material.metallic, 0.0)
        self.assertEqual(material.reflectance, 0.4)
        self.assertEqual(material.anisotropy, 0.4)
        self.assertIsNone(material.mitsuba_bsdf)

    def test_custom_colour(self):
        """
        Test creation with a custom colour.
        """
        colour = np.array([1.0, 0.0, 0.0])
        material = Material(colour=colour)
        np.testing.assert_array_equal(material.colour, colour)

    def test_custom_properties(self):
        """
        Test creation with custom material properties.
        """
        material = Material(
            colour=np.array([0.5, 0.5, 0.5]),
            alpha=0.8,
            roughness=0.3,
            metallic=1.0,
            reflectance=0.9,
            anisotropy=0.1,
        )
        self.assertEqual(material.alpha, 0.8)
        self.assertEqual(material.roughness, 0.3)
        self.assertEqual(material.metallic, 1.0)
        self.assertEqual(material.reflectance, 0.9)
        self.assertEqual(material.anisotropy, 0.1)

    def test_3d_colour_array(self):
        """
        Test that a 3D colour array (per-particle-per-timestep) is stored correctly.
        """
        colour = np.random.uniform(0, 1, (10, 5, 3))
        material = Material(colour=colour)
        self.assertEqual(material.colour.ndim, 3)
        self.assertEqual(material.colour.shape, (10, 5, 3))

    def test_colour_ndim(self):
        """
        Test the ndim property used for per-particle colouring checks.
        """
        material_1d = Material(colour=np.array([1.0, 0.0, 0.0]))
        self.assertEqual(material_1d.colour.ndim, 1)

        material_3d = Material(colour=np.zeros((5, 3, 3)))
        self.assertEqual(material_3d.colour.ndim, 3)
