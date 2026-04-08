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
Test the Mitsuba rendering module.
"""

import unittest

import numpy as np

from znvis.rendering.mitsuba import Mitsuba, default_scene_dict


class TestMitsuba(unittest.TestCase):
    """
    A test class for the Mitsuba renderer.
    """

    def test_default_initialization(self):
        """
        Test that the default scene dict is used when none is provided.
        """
        renderer = Mitsuba()
        self.assertEqual(renderer.scene_dict, default_scene_dict)
        self.assertTrue(renderer.update_camera)

    def test_custom_scene_dict(self):
        """
        Test that a custom scene dict is stored correctly.
        """
        custom_dict = {"type": "scene", "integrator": {"type": "direct"}}
        renderer = Mitsuba(scene_dict=custom_dict)
        self.assertEqual(renderer.scene_dict, custom_dict)

    def test_no_camera_update(self):
        """
        Test that camera updates can be disabled.
        """
        renderer = Mitsuba(update_camera=False)
        self.assertFalse(renderer.update_camera)

    def test_update_camera(self):
        """
        Test that _update_camera modifies the scene dict sensor transform.
        """
        renderer = Mitsuba()
        view_matrix = np.eye(4)
        renderer._update_camera(view_matrix)
        self.assertIn("to_world", renderer.scene_dict["sensor"])

    def test_default_scene_dict_structure(self):
        """
        Test that the default scene dict has the expected structure.
        """
        self.assertEqual(default_scene_dict["type"], "scene")
        self.assertIn("integrator", default_scene_dict)
        self.assertIn("light", default_scene_dict)
        self.assertIn("sensor", default_scene_dict)
        self.assertIn("thefilm", default_scene_dict["sensor"])
        self.assertIn("thesampler", default_scene_dict["sensor"])
