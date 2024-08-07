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
Unit test for the bounding box.
"""

import numpy as np
import open3d as o3d

from znvis.bounding_objects.bounding_box import BoundingBox


class TestBoundingBox:
    """
    A test class for the Particle class.
    """

    @classmethod
    def setup_class(cls) -> None:
        """
        Set up the test.

        Returns
        -------
        Sets up a sphere instance for testing
        """
        cls.box = BoundingBox(
            colour=np.array([0.7, 0.3, 0.1]),
            center=np.array([0, 0, 0]),
            box_size=np.array([20, 20, 20]),
        )

    def test_instantiation(self):
        """
        Test the instantiation of a Mesh.

        Returns
        -------
        Check that parameters are set correctly.
        """
        np.testing.assert_array_equal(self.box.colour, np.array([0.7, 0.3, 0.1]))
        np.testing.assert_array_equal(self.box.center, np.array([0, 0, 0]))
        np.testing.assert_array_equal(self.box.box_size, np.array([20, 20, 20]))
        np.testing.assert_array_equal(
            self.box.rotation_matrix, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        )

    def test_call(self):
        """
        Test the construction of a sphere mesh.

        Returns
        -------
        Test if a sphere mesh is constructed correctly.
        """
        box_object = self.box()
        assert isinstance(box_object, o3d.geometry.OrientedBoundingBox)
