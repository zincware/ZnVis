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
from znvis.particle.particle import Particle


class TestParticle(unittest.TestCase):
    """
    A test class for the Particle class.
    """
    def test_initialization(self):
        """
        Test the initialization of the class.

        Returns
        -------

        """
        # Most simple instantiation
        name = "my_particle"
        particle = Particle(name=name)
        self.assertEqual(particle.name, name)


