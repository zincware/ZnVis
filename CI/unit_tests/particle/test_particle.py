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
Test the particle dataclass operations.
"""
import unittest

import numpy as np

from znvis.mesh.sphere import Sphere
from znvis.particle.particle import Particle


class TestParticle(unittest.TestCase):
    """
    A test class for the Particle class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the Particle class for the test.

        Returns
        -------

        """
        name = "my_particle"
        position = np.random.uniform(-5, 5, (10, 2, 3))
        cls.particle = Particle(name=name, position=position, mesh=Sphere())

    def test_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.particle.mesh), Sphere)
        self.assertEqual(self.particle.name, "my_particle")
        np.testing.assert_array_equal(self.particle.position.shape, (10, 2, 3))

    def test_construct_mesh_dict(self):
        """
        Test the construct_mesh_dict method.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Build the mesh dict
        self.particle.construct_mesh_dict()

        # Check that all particle are in the dict.
        self.assertEqual(len(self.particle.mesh_dict), self.particle.position.shape[0])
