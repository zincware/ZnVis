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
Test the rotation matrices module.
"""
import unittest

import numpy as np

from znvis.transformations.rotation_matrices import rotation_matrix


class TestRotationMatrix(unittest.TestCase):
    """
    A test class for the Particle class.
    """

    def test_rotation(self):
        """
        Test a rotation on two vectors.
        """
        reference = np.array([1, 0, 0])
        new_direction = np.array([0, 1, 0])

        rotation = rotation_matrix(reference, new_direction)

        rotated_vector = np.dot(rotation, reference)
        print(rotated_vector)
