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
Tests for lazy mesh cache policy and worker management.
"""

import unittest

import numpy as np

from znvis.mesh_cache.mesh_cache_manager import MeshCacheManager
from znvis.mesh_cache.mesh_frame_cache import MeshFrameCache


class DummyMesh:
    """
    Minimal mesh-like object exposing Open3D-style array attributes.
    """

    def __init__(self, nbytes: int):
        self.vertices = np.zeros(nbytes, dtype=np.uint8)


class DummyLazyItem:
    """
    Minimal visualizer item for cache manager tests.
    """

    def __init__(self, name: str, static: bool, nbytes: int):
        self.name = name
        self.static = static
        self.nbytes = nbytes
        self.loaded_frames = []

    def get_mesh_for_frame(self, frame_index: int):
        self.loaded_frames.append(frame_index)
        return DummyMesh(self.nbytes)


class TestMeshCacheManager(unittest.TestCase):
    """
    Test rolling lazy mesh cache policy independent of the visualizer UI.
    """

    def make_manager(self, item, number_of_steps=10, max_bytes=100):
        return MeshCacheManager(
            items=[item],
            number_of_steps=number_of_steps,
            cache=MeshFrameCache(max_bytes=max_bytes),
            future_fraction=2 / 3,
        )

    def test_initialize_stops_when_byte_budget_is_filled(self):
        """
        Test that initial loading stops once a new frame would exceed the budget.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=40)
        manager = self.make_manager(item, number_of_steps=10, max_bytes=100)

        manager.initialize(
            current_frame=0,
            do_rewind=False,
            frame_step=1,
            start_worker=False,
        )

        self.assertEqual(item.loaded_frames, [0, 1, 2])
        self.assertEqual(manager.get_complete_frames_count(), 2)
        self.assertEqual(manager.get_cached_frame_ranges(), "0-1")
        self.assertTrue(manager.cache.contains(item, 0))
        self.assertTrue(manager.cache.contains(item, 1))
        self.assertFalse(manager.cache.contains(item, 2))

    def test_ensure_current_frame_loads_only_current_dynamic_frame(self):
        """
        Test synchronous playback fallback only builds the current frame.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        manager = self.make_manager(item, number_of_steps=5, max_bytes=20)

        manager.ensure_current_frame(3)

        self.assertEqual(item.loaded_frames, [3])
        self.assertEqual(manager.complete_frames, {3})

    def test_urgent_cache_miss_is_filled_first(self):
        """
        Test an urgent display miss wins over normal refill priority.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        manager = self.make_manager(item, number_of_steps=20, max_bytes=1)
        manager.initial_cached_frame_count = 6

        manager.submit_prefetch(8, False, 1, urgent_frame=42)
        manager.run_once(timeout=0)

        self.assertEqual(item.loaded_frames[0], 42)
        self.assertIn(42, item.loaded_frames)

    def test_urgent_cache_miss_outside_policy_window_is_filled_first(self):
        """
        Test an urgent display miss is honored even outside the rolling window.
        """
        item = DummyLazyItem("dynamic", static=False, nbytes=10)
        manager = self.make_manager(item, number_of_steps=100, max_bytes=1)
        manager.initial_cached_frame_count = 10

        manager.submit_prefetch(10, False, 1, urgent_frame=90)
        manager.run_once(timeout=0)

        self.assertEqual(item.loaded_frames[0], 90)
        self.assertIn(90, item.loaded_frames)

    def test_pause_and_running_refill_use_same_policy(self):
        """
        Test paused and running refills choose the same missing policy frames.
        """
        running_item = DummyLazyItem("dynamic", static=False, nbytes=10)
        running_manager = self.make_manager(
            running_item, number_of_steps=20, max_bytes=1
        )
        running_manager.initial_cached_frame_count = 6
        running_manager.submit_prefetch(8, False, 1)
        running_manager.run_once(timeout=0)

        paused_item = DummyLazyItem("dynamic", static=False, nbytes=10)
        paused_manager = self.make_manager(paused_item, number_of_steps=20, max_bytes=1)
        paused_manager.initial_cached_frame_count = 6
        paused_manager.submit_pause_refill(8, False, 1)
        paused_manager.run_once(timeout=0)

        self.assertEqual(paused_item.loaded_frames, running_item.loaded_frames)


if __name__ == "__main__":
    unittest.main()
