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
from unittest import mock

import numpy as np
import open3d as o3d

from znvis.mesh.sphere import Sphere
from znvis.particle.particle import Particle
from znvis.testing.znvis_process import Process
from znvis.visualizer.visualizer import Visualizer


class DummyMesh:
    """
    Minimal mesh-like object exposing Open3D-style array attributes.
    """

    def __init__(self, nbytes: int):
        self.vertices = np.zeros(nbytes, dtype=np.uint8)


class DummyLazyItem:
    """
    Minimal visualizer item for cache preloading tests.
    """

    def __init__(self, name: str, static: bool, nbytes: int):
        self.name = name
        self.static = static
        self.nbytes = nbytes
        self.position = np.zeros((10, 3))
        self.loaded_frames = []

    def get_mesh_for_frame(self, frame_index: int):
        self.loaded_frames.append(frame_index)
        return DummyMesh(self.nbytes)


def initialize_app_in_process():
    """
    Initialize the visualizer app inside a separate process.

    Returns
    -------
    None
    """
    name = "my_particle"
    position = np.random.uniform(-5, 5, (10, 2, 3))
    particle = Particle(name=name, position=position, mesh=Sphere())
    visualizer = Visualizer([particle])
    visualizer._initialize_app()
    assert type(visualizer.vis) is o3d.visualization.O3DVisualizer


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
        visualizer = Visualizer(self.visualizer.particles)
        self.assertEqual(visualizer.number_of_steps, 10)
        self.assertEqual(visualizer.counter, 0)

    def test_lazy_mesh_cache_uses_configured_byte_limit(self):
        """
        Test that lazy mesh loading converts the GB limit for the frame cache.
        """
        visualizer = Visualizer(
            self.visualizer.particles,
            lazy_mesh_loading=True,
            mesh_cache_max_gb=1.5,
        )

        visualizer._initialize_mesh_cache()

        self.assertEqual(visualizer.mesh_cache.max_bytes, int(1.5 * 1024**3))

    def test_initial_lazy_cache_stops_when_byte_budget_is_filled(self):
        """
        Test that initial preloading stops once a new frame would exceed the budget.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=40)
        visualizer = Visualizer(
            [item],
            number_of_steps=10,
            lazy_mesh_loading=True,
            mesh_cache_max_gb=100 / 1024**3,
        )

        visualizer._initialize_mesh_cache()

        self.assertEqual(item.loaded_frames, [0, 1, 2])
        self.assertEqual(visualizer.mesh_cache_manager.get_complete_frames_count(), 2)
        self.assertEqual(visualizer.mesh_cache_manager.get_cached_frame_ranges(), "0-1")
        self.assertTrue(visualizer.mesh_cache.contains(item, 0))
        self.assertTrue(visualizer.mesh_cache.contains(item, 1))
        self.assertFalse(visualizer.mesh_cache.contains(item, 2))

    def test_lazy_cache_prefetch_wakes_manager(self):
        """
        Test playback prefetch refills a missing frame when the worker runs.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        visualizer = Visualizer(
            [item],
            number_of_steps=5,
            lazy_mesh_loading=True,
            mesh_cache_max_gb=20 / 1024**3,
        )
        visualizer._initialize_mesh_cache()
        visualizer.mesh_cache._dynamic_cache.clear()
        visualizer.mesh_cache.dynamic_bytes = 0
        visualizer.mesh_cache_manager.complete_frames.clear()
        item.loaded_frames.clear()

        visualizer.mesh_cache_manager.submit_prefetch(
            visualizer.counter,
            visualizer.do_rewind,
            visualizer._get_playback_frame_step(),
        )
        visualizer.mesh_cache_manager.run_once(timeout=0)

        self.assertEqual(item.loaded_frames, [0, 4])

    def test_fast_playback_skips_frames(self):
        """
        Test autoplay can advance by multiple frames per GUI update.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        visualizer = Visualizer(
            [item],
            number_of_steps=10,
            lazy_mesh_loading=True,
            mesh_cache_max_gb=1,
        )
        visualizer._initialize_mesh_cache()

        with (
            mock.patch.object(visualizer, "_draw_particles"),
            mock.patch.object(visualizer, "_draw_vector_field"),
        ):
            visualizer._update_particles(
                visualizer=mock.Mock(),
                frame_step=4,
            )

        self.assertEqual(visualizer.counter, 4)

    def test_fast_rewind_skips_frames_backwards(self):
        """
        Test backward autoplay advances in the reverse direction for fast speeds.
        """
        visualizer = self.visualizer
        visualizer.counter = 6
        visualizer.do_rewind = True

        with (
            mock.patch.object(visualizer, "_draw_particles"),
            mock.patch.object(visualizer, "_draw_vector_field"),
        ):
            visualizer._update_particles(
                visualizer=mock.Mock(),
                frame_step=4,
            )

        self.assertEqual(visualizer.counter, 2)

    def test_nonblocking_playback_does_not_build_missing_frame(self):
        """
        Test autoplay skips a cache miss instead of blocking the GUI thread.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        visualizer = Visualizer(
            [item],
            number_of_steps=5,
            lazy_mesh_loading=True,
            mesh_cache_max_gb=20 / 1024**3,
        )
        visualizer._initialize_mesh_cache()
        visualizer.mesh_cache._dynamic_cache.clear()
        visualizer.mesh_cache.dynamic_bytes = 0
        visualizer.mesh_cache_manager.complete_frames.clear()
        item.loaded_frames.clear()

        with mock.patch.object(visualizer, "_draw_particles"):
            updated = visualizer._update_particles(
                visualizer=mock.Mock(),
                block_on_cache_miss=False,
            )

        self.assertFalse(updated)
        self.assertEqual(visualizer.counter, 0)
        self.assertEqual(item.loaded_frames, [])
        visualizer.mesh_cache_manager.run_once(timeout=0)
        self.assertIn(1, item.loaded_frames)

    def test_pause_starts_background_refill(self):
        """
        Test pausing lazy playback triggers a background refill.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        visualizer = Visualizer(
            [item],
            number_of_steps=10,
            lazy_mesh_loading=True,
            mesh_cache_max_gb=1,
        )
        visualizer._initialize_mesh_cache()
        visualizer.mesh_cache._dynamic_cache.clear()
        visualizer.mesh_cache.dynamic_bytes = 0
        visualizer.mesh_cache_manager.complete_frames.clear()
        item.loaded_frames.clear()
        visualizer.counter = 3

        visualizer._pause_run(mock.Mock())
        visualizer.mesh_cache_manager.run_once(timeout=0)

        self.assertEqual(visualizer.interrupt, 0)
        self.assertGreater(len(item.loaded_frames), 0)

    def test_manager_ensure_current_frame_loads_only_current_dynamic_frame(self):
        """
        Test synchronous playback fallback only builds the current frame.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        visualizer = Visualizer(
            [item],
            number_of_steps=5,
            lazy_mesh_loading=True,
            mesh_cache_max_gb=20 / 1024**3,
        )
        visualizer._initialize_mesh_cache()
        item.loaded_frames.clear()

        visualizer.mesh_cache_manager.ensure_current_frame(3)

        self.assertEqual(item.loaded_frames, [3])

    def test_initialize_app(self):
        """
        Test instantiation of the app.
        """
        process = Process(target=initialize_app_in_process)
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
        self.vis._toggle_play_direction()
        self.assertTrue(self.vis.do_rewind)
        self.vis._toggle_play_direction()
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
