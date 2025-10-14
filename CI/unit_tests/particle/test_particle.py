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
        director = np.random.uniform(-5, 5, (10, 2, 3))
        cls.particle = Particle(
            name=name,
            position=position,
            director=director,
            mesh=Sphere(material=cls.material),
        )

        static_name = "my_static_particle"
        static_position = np.random.uniform(-5, 5, (2, 3))
        static_director = np.random.uniform(-5, 5, (2, 3))
        cls.static_particle = Particle(
            name=static_name,
            position=static_position,
            director=static_director,
            mesh=Sphere(),
            static=True,
            smoothing=True,
        )

        empty_name = "my_empty_particle"
        cls.empty_particle = Particle(
            name=empty_name, mesh=Sphere(), position=np.array([]), director=np.array([])
        )

        nan_name = "my_nan_particle"
        nan_position = [np.full((2, 3), np.nan) for _ in range(10)]
        cls.nan_particle = Particle(
            name=nan_name,
            mesh=Sphere(),
            position=nan_position,
            director=nan_position.copy(),
        )

        configuration_warning_name = "my_static_mixing_particle"
        configuration_warning_position = np.random.uniform(-5, 5, (1, 2, 3))
        configuration_warning_director = np.random.uniform(-5, 5, (1, 2, 3))
        cls.configuration_warning_particle = Particle(
            name=configuration_warning_name,
            position=configuration_warning_position,
            director=configuration_warning_director,
            mesh=Sphere(),
            static=True,
            smoothing=True,
        )

    def test_configuration_warning(self):
        """
        Test the construct_mesh_list method for a static mesh.

        Returns
        -------
        Tests whether the list was created properly.
        """
        self.configuration_warning_particle.construct_mesh_list()
        assert self.configuration_warning_particle.position[0].shape == (2, 3)
        assert self.configuration_warning_particle.director[0].shape == (2, 3)

    def test_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.particle.mesh), Sphere)
        self.assertEqual(self.particle.name, "my_particle")
        self.assertEqual(len(self.particle.position), 10)
        np.testing.assert_array_equal(self.particle.position[0].shape, (2, 3))

    def test_construct_mesh_list(self):
        """
        Test the construct_mesh_list method.

        Returns
        -------
        Tests whether the list was created properly.
        """
        # Build the mesh list
        self.particle.construct_mesh_list()

        # Check that all time steps are in the list.
        self.assertEqual(len(self.particle.mesh_list), len(self.particle.position))

    def test_static_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.static_particle.mesh), Sphere)
        self.assertEqual(self.static_particle.name, "my_static_particle")
        # NOTE
        # What behavior is needed here?
        # np.testing.assert_array_equal(self.static_particle.position.shape, (1, 2, 3))
        self.assertTrue(self.static_particle.static)
        self.assertTrue(self.static_particle.smoothing)

    def test_construct_static_mesh_list(self):
        """
        Test the construct_mesh_list method for a static mesh.

        Returns
        -------
        Tests whether the list was created properly.
        """
        self.assertEqual(np.asarray(self.static_particle.position).shape, (2, 3))
        self.static_particle.construct_mesh_list()

        self.assertEqual(np.asarray(self.static_particle.position)[0].shape, (2, 3))

    def test_empty_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.empty_particle.mesh), Sphere)
        self.assertEqual(self.empty_particle.name, "my_empty_particle")

    def test_construct_empty_mesh_list(self):
        """
        Test the construct_mesh_list method for an empty mesh.

        Returns
        -------
        Tests if the ValueError is thrown.
        """
        with self.assertRaises((ValueError)) as context:
            self.empty_particle.construct_mesh_list()
        # Check if error message is correct
        self.assertIn("The provided position array is empty.", str(context.exception))

        self.empty_particle.position = np.random.uniform(-5, 5, (2, 2, 3))
        self.empty_particle.construct_mesh_list()
        assert self.empty_particle.director is None

    def test_construct_nan_mesh_list(self):
        """
        Test the construct_mesh_list method for a particle with nans in the trajectory.

        Returns
        -------
        Tests whether the list was created properly.
        """
        with self.assertRaises((ValueError)) as context:
            self.nan_particle.construct_mesh_list()
        self.assertIn(
            "The provided position data contains at least one "
            "NaN value at time step 0.",
            str(context.exception),
        )

        self.nan_particle.position = np.random.uniform(-5, 5, (2, 2, 3))
        self.nan_particle.director = np.full((2, 2, 3), np.nan)
        with self.assertRaises((ValueError)) as context:
            self.nan_particle.construct_mesh_list()
        self.assertIn(
            "The provided director data contains at least one "
            "NaN value at time step 0.",
            str(context.exception),
        )
