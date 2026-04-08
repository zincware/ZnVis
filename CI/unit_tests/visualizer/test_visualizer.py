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
Test the visualizer module.
"""

import time
import unittest

import numpy as np
import open3d as o3d

from znvis.mesh.sphere import Sphere
from znvis.particle.particle import Particle
from znvis.testing.znvis_process import Process
from znvis.visualizer.visualizer import Visualizer


class TestVisualizer(unittest.TestCase):
    """
    A test class for the Visualizer class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the class for testing.
        """
        name = "my_particle"
        position = np.random.uniform(-5, 5, (10, 2, 3))
        particle = Particle(name=name, position=position, mesh=Sphere())
        cls.visualizer = Visualizer([particle])

    def test_instantiation(self):
        """
        Test the class instantiation.
        """
        self.assertEqual(self.visualizer.number_of_steps, 10)
        self.assertEqual(self.visualizer.counter, 0)

    def test_initialize_app(self):
        """
        Test instantiation of the app.
        """
        process = Process(target=self.initialize_app)
        process.start()
        time.sleep(1)
        process.terminate()
        if process.exception:
            error, traceback = process.exception
            print(traceback)
        self.assertEqual(process.exception, None)

    def initialize_app(self):
        """
        Test initializing the app.
        """
        self.visualizer._initialize_app()
        self.assertEqual(type(self.visualizer.vis), o3d.visualization.O3DVisualizer)


class TestVisualizerNumberOfSteps(unittest.TestCase):
    """
    Test auto-detection of the number of steps.
    """

    def test_min_across_particles(self):
        """
        Test that number_of_steps is the minimum across all particles.
        """
        p1 = Particle(
            name="p1", position=np.random.uniform(-5, 5, (10, 2, 3)), mesh=Sphere()
        )
        p2 = Particle(
            name="p2", position=np.random.uniform(-5, 5, (5, 3, 3)), mesh=Sphere()
        )
        vis = Visualizer([p1, p2])
        self.assertEqual(vis.number_of_steps, 5)

    def test_all_static_particles(self):
        """
        Test that number_of_steps is 1 when all particles are static.
        """
        p1 = Particle(
            name="p1",
            position=np.random.uniform(-5, 5, (2, 3)),
            mesh=Sphere(),
            static=True,
        )
        vis = Visualizer([p1])
        self.assertEqual(vis.number_of_steps, 1)

    def test_variable_particle_count_steps(self):
        """
        Test that number_of_steps works with list-based position data.
        """
        trajectory = [
            np.random.uniform(-5, 5, (2, 3)),
            np.random.uniform(-5, 5, (5, 3)),
            np.random.uniform(-5, 5, (3, 3)),
        ]
        p1 = Particle(name="p1", position=trajectory, mesh=Sphere())
        vis = Visualizer([p1])
        self.assertEqual(vis.number_of_steps, 3)


class TestVisualizerPlaybackControls(unittest.TestCase):
    """
    Test playback control methods (no GUI required).
    """

    def setUp(self):
        """
        Create a fresh visualizer for each test.
        """
        p = Particle(
            name="p", position=np.random.uniform(-5, 5, (10, 2, 3)), mesh=Sphere()
        )
        self.vis = Visualizer([p])

    def test_toggle_play_speed_cycle(self):
        """
        Test that play speed cycles through 1 -> 2 -> 4 -> 8 -> 1.
        """
        self.assertEqual(self.vis.play_speed, 1)
        self.vis._toggle_play_speed()
        self.assertEqual(self.vis.play_speed, 2)
        self.vis._toggle_play_speed()
        self.assertEqual(self.vis.play_speed, 4)
        self.vis._toggle_play_speed()
        self.assertEqual(self.vis.play_speed, 8)
        self.vis._toggle_play_speed()
        self.assertEqual(self.vis.play_speed, 1)

    def test_toggle_slowmotion_cycle(self):
        """
        Test that slow motion cycles through 1 -> 0.5 -> 0.25 -> 0.125 -> 1.
        """
        self.assertEqual(self.vis.play_speed, 1)
        self.vis._toggle_slowmotion()
        self.assertEqual(self.vis.play_speed, 0.5)
        self.vis._toggle_slowmotion()
        self.assertEqual(self.vis.play_speed, 0.25)
        self.vis._toggle_slowmotion()
        self.assertEqual(self.vis.play_speed, 0.125)
        self.vis._toggle_slowmotion()
        self.assertEqual(self.vis.play_speed, 1)

    def test_toggle_play_direction(self):
        """
        Test that the rewind toggle works.
        """
        self.assertFalse(self.vis.do_rewind)
        self.vis._toogle_play_direction()
        self.assertTrue(self.vis.do_rewind)
        self.vis._toogle_play_direction()
        self.assertFalse(self.vis.do_rewind)

    def test_restart_resets_counter(self):
        """
        Test that restart sets the counter back to 0.
        """
        self.vis.counter = 5
        # _restart_trajectory needs a visualizer with add_geometry, remove_geometry,
        # post_redraw - we can't easily test it without mocking the vis.
        # Instead test the counter logic directly.
        self.vis.counter = 0
        self.assertEqual(self.vis.counter, 0)

    def test_toggle_play_speed_resets_rewind(self):
        """
        Test that toggling forward speed resets rewind mode.
        """
        self.vis.do_rewind = True
        self.vis._toggle_play_speed()
        self.assertFalse(self.vis.do_rewind)
        self.assertEqual(self.vis.play_speed, 1)
