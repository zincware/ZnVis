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
Test the Vector Fields dataclass operations.
"""

import unittest

import numpy as np

from znvis import Material
from znvis.mesh.arrow import Arrow
from znvis.particle.vector_field import VectorField


class TestVectorField(unittest.TestCase):
    """
    A test class for the VectorField class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare an instance of the VectorField class for the test.

        Returns
        -------

        """
        cls.material = Material(
            colour=np.full((10, 10, 3), np.array([0, 0, 255]) / 255)
        )
        name = "my_vector_field"

        position = np.random.uniform(-5, 5, (10, 2, 3))
        static_position = np.random.uniform(-5, 5, (2, 3))
        directors = np.random.uniform(-5, 5, (10, 2, 3))
        static_directors = np.random.uniform(-5, 5, (2, 3))

        cls.vector_field = VectorField(
            name=name,
            position=position,
            mesh=Arrow(scale=10, material=cls.material),
            direction=directors,
        )

        static_name = "my_static_vector_field"

        cls.static_vector_field = VectorField(
            name=static_name,
            position=static_position,
            mesh=Arrow(scale=10, material=cls.material),
            direction=static_directors,
            smoothing=True,
            static=True,
        )

        empty_name = "my_empty_vector_field"
        cls.empty_vector_field = VectorField(
            name=empty_name,
            position=np.array([]),
            mesh=Arrow(scale=10, material=cls.material),
            direction=np.array([]),
            static=False,
        )
        nan_name = "my_nan_vector_field"
        cls.nan_vector_field = VectorField(
            name=nan_name,
            position=np.full_like(position, np.nan),
            mesh=Arrow(scale=10, material=cls.material),
            direction=np.full_like(directors, np.nan),
        )
        none_pos_name = "my_none_pos_vector_field"
        cls.none_pos_vector_field = VectorField(
            name=none_pos_name,
            position=None,
            mesh=Arrow(scale=10, material=cls.material),
            direction=directors,
        )
        none_dir_name = "my_none_dir_vector_field"
        cls.none_dir_vector_field = VectorField(
            name=none_dir_name,
            position=position,
            mesh=Arrow(scale=10, material=cls.material),
            direction=None,
        )

    def test_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.vector_field.mesh), Arrow)
        self.assertEqual(self.vector_field.name, "my_vector_field")
        np.testing.assert_array_equal(self.vector_field.position.shape, (10, 2, 3))
        np.testing.assert_array_equal(self.vector_field.direction.shape, (10, 2, 3))

    def test_construct_mesh_dict(self):
        """
        Test the construct_mesh_dict method.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Build the mesh dict
        self.vector_field.construct_mesh_list()

        # Check that all time steps are in the dict.
        self.assertEqual(
            len(self.vector_field.mesh_list), self.vector_field.position.shape[0]
        )

    def test_static_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.static_vector_field.mesh), Arrow)
        self.assertEqual(self.static_vector_field.name, "my_static_vector_field")

        # NOTE Same problem as in test_particle.
        # This yields an error despite the construct mesh is called after this!
        # np.testing.assert_array_equal(
        #   self.static_vector_field.position.shape,
        #   (10, 2, 3))
        # np.testing.assert_array_equal(
        #   self.static_vector_field.direction.shape,
        #   (10, 2, 3))

    def test_construct_static_mesh_dict(self):
        """
        Test the construct_mesh_dict method.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Build the mesh dict NOTE
        np.testing.assert_array_equal(self.static_vector_field.position.shape, (2, 3))
        np.testing.assert_array_equal(self.static_vector_field.direction.shape, (2, 3))
        self.static_vector_field.construct_mesh_list()
        np.testing.assert_array_equal(
            self.static_vector_field.position.shape, (1, 2, 3)
        )
        np.testing.assert_array_equal(
            self.static_vector_field.direction.shape, (1, 2, 3)
        )

        # Check that all time steps are in the dict.
        self.assertEqual(
            len(self.static_vector_field.mesh_list),
            self.static_vector_field.position.shape[0],
        )

    def test_empty_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------
        Check if the attributes are set correctly.
        """
        self.assertEqual(type(self.empty_vector_field.mesh), Arrow)
        self.assertEqual(self.empty_vector_field.name, "my_empty_vector_field")

    def test_construct_empty_mesh_dict(self):
        """
        Test the construct_mesh_dict method for an empty mesh.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Attempt to build the mesh dict

        with self.assertRaises(IndexError) as context:
            self.empty_vector_field.construct_mesh_list()
        # Check if error message is correct
        self.assertEqual(
            str(context.exception), "The provided data has an incompatible shape."
        )

    def test_construct_nan_mesh_dict(self):
        """
        Test the construct_mesh_dict method for a mesh containing nans.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Attempt to build the mesh dict

        with self.assertRaises(ValueError) as context:
            self.nan_vector_field.construct_mesh_list()
        # Check if error message is correct
        self.assertEqual(str(context.exception), "The provided data contains NaNs.")

    def test_construct_none_pos_mesh_dict(self):
        """
        Test the construct_mesh_dict method for a mesh with position=None.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Attempt to build the mesh dict

        with self.assertRaises(ValueError) as context:
            self.none_pos_vector_field.construct_mesh_list()
        # Check if error message is correct
        self.assertEqual(str(context.exception), "Position data cannot be None.")

    def test_construct_none_dir_mesh_dict(self):
        """
        Test the construct_mesh_dict method for a mesh with directors=None.

        Returns
        -------
        Tests whether the dict was created properly.
        """
        # Attempt to build the mesh dict

        with self.assertRaises(ValueError) as context:
            self.none_dir_vector_field.construct_mesh_list()
        # Check if error message is correct
        self.assertEqual(str(context.exception), "Director data cannot be None.")


class TestVariableVectorFieldCount(unittest.TestCase):
    """
    Test class for vector fields with a variable number of vectors per time step.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare vector fields with list-of-arrays data.
        """
        cls.positions = [
            np.random.uniform(-5, 5, (2, 3)),
            np.random.uniform(-5, 5, (5, 3)),
            np.random.uniform(-5, 5, (3, 3)),
        ]
        cls.directions = [
            np.random.uniform(-1, 1, (2, 3)),
            np.random.uniform(-1, 1, (5, 3)),
            np.random.uniform(-1, 1, (3, 3)),
        ]
        cls.vector_field = VectorField(
            name="variable_vf",
            position=cls.positions,
            mesh=Arrow(scale=10),
            direction=cls.directions,
        )

    def test_variable_count_initialization(self):
        """
        Test that a vector field with list data initializes correctly.
        """
        self.assertEqual(self.vector_field.name, "variable_vf")
        self.assertIsInstance(self.vector_field.position, list)
        self.assertEqual(len(self.vector_field.position), 3)

    def test_variable_count_mesh_list(self):
        """
        Test that construct_mesh_list produces the correct number of meshes.
        """
        self.vector_field.construct_mesh_list()
        self.assertEqual(len(self.vector_field.mesh_list), 3)

    def test_variable_count_nan_check(self):
        """
        Test that NaN detection works for list-based data.
        """
        nan_positions = [
            np.random.uniform(-5, 5, (2, 3)),
            np.full((3, 3), np.nan),
        ]
        nan_directions = [
            np.random.uniform(-1, 1, (2, 3)),
            np.random.uniform(-1, 1, (3, 3)),
        ]
        nan_vf = VectorField(
            name="nan_variable_vf",
            position=nan_positions,
            mesh=Arrow(scale=10),
            direction=nan_directions,
        )
        with self.assertRaises(ValueError) as context:
            nan_vf.construct_mesh_list()
        self.assertEqual(str(context.exception), "The provided data contains NaNs.")
