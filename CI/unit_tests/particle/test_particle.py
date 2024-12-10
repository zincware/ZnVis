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

from znvis import Material
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
        cls.material = Material(colour=np.zeros((10, 10, 3)))
        name = "my_particle"
        position = np.random.uniform(-5, 5, (10, 2, 3))

        cls.particle = Particle(
            name=name, 
            position=position,
            mesh=Sphere(material=cls.material),
        )

        static_name="my_static_particle"
        static_position = np.random.uniform(-5, 5, (2, 3))

        cls.static_particle = Particle(
            name=static_name, 
            position=static_position,
            mesh=Sphere(),
            static=True,
            smoothing=True,
            director=np.random.uniform(-5, 5, (2, 3))
        )

        empty_name = "my_empty_particle"

        cls.empty_particle = Particle(
            name=empty_name,
            mesh=Sphere(),
            position=np.array([])
        )

        nan_name = "my_nan_particle"

        cls.nan_name = Particle(
            name=nan_name,
            mesh=Sphere(),
            position=np.full_like(position, np.nan)
        )


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
        self.particle.construct_mesh_list()

        # Check that all time steps are in the dict.
        self.assertEqual(len(self.particle.mesh_list), self.particle.position.shape[0])


    def test_static_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.static_particle.mesh), Sphere)
        self.assertEqual(self.static_particle.name, "my_static_particle")
        #NOTE
        # What behavior is needed here? 
        #np.testing.assert_array_equal(self.static_particle.position.shape, (1, 2, 3))
        self.assertEqual(self.static_particle.static, True)
        self.assertEqual(self.static_particle.smoothing, True)

    def test_construct_static_mesh_dict(self):
            # Build the mesh dict
            np.testing.assert_array_equal(self.static_particle.position.shape, (2, 3))
            self.static_particle.construct_mesh_list()
            np.testing.assert_array_equal(self.static_particle.position.shape, (1, 2, 3))
            # Check that all time steps are in the dict.
            self.assertEqual(len(self.static_particle.mesh_list), self.static_particle.position.shape[0])

    def test_empty_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.empty_particle.mesh), Sphere)
        self.assertEqual(self.empty_particle.name, "my_empty_particle")

    def test_construct_empty_mesh_dict(self):
        """
        Test the construct_mesh_dict method for a static mesh.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Attempt to build the mesh dict

        with self.assertRaises((IndexError)) as context:
            self.empty_particle.construct_mesh_list()
        # Check if error message is correct
        self.assertEqual(
            str(context.exception),
           "The provided data has an incompatible shape."
        )

    def test_construct_nan_mesh_dict(self):
        """
        Test the construct_mesh_dict method for a particle with nans in the trajectory.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Attempt to build the mesh dict

        with self.assertRaises((ValueError)) as context:
            self.nan_name.construct_mesh_list()
        # Check if error message is correct
        self.assertEqual(
            str(context.exception),
           "The provided data contains NaN values."
        )