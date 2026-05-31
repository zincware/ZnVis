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
Tests for interactive visualizer mesh frame caching.
"""

import unittest

import numpy as np

from znvis.visualizer.cache.mesh_frame_cache import MeshFrameCache, estimate_mesh_nbytes


class DummyMesh:
    """
    Minimal mesh-like object exposing Open3D-style array attributes.
    """

    def __init__(self, nbytes: int):
        self.vertices = np.zeros(nbytes, dtype=np.uint8)


class DummyItem:
    """
    Minimal cache item with static/dynamic behavior.
    """

    def __init__(self, name: str, static: bool, nbytes: int):
        self.name = name
        self.static = static
        self.nbytes = nbytes
        self.builds = 0

    def get_mesh_for_frame(self, frame_index: int):
        self.builds += 1
        return DummyMesh(self.nbytes)


class TestMeshFrameCache(unittest.TestCase):
    """
    Test LRU cache behavior for static and dynamic visualizer meshes.
    """

    def test_estimate_mesh_nbytes_counts_mesh_arrays(self):
        """
        Test that mesh byte estimation sums known mesh array storage.
        """
        mesh = DummyMesh(12)
        mesh.triangles = np.zeros(8, dtype=np.uint8)

        self.assertEqual(estimate_mesh_nbytes(mesh), 20)

    def test_static_meshes_stay_cached_when_dynamic_entries_are_evicted(self):
        """
        Test that static meshes remain resident and dynamic entries honor max_bytes.
        """
        static_item = DummyItem("static", static=True, nbytes=80)
        dynamic_item = DummyItem("dynamic", static=False, nbytes=30)
        cache = MeshFrameCache(max_bytes=100)

        static_mesh = cache.get(static_item, 0)
        cache.get(dynamic_item, 0)

        self.assertIs(cache.get(static_item, 0), static_mesh)
        self.assertEqual(static_item.builds, 1)
        self.assertFalse(cache.contains(dynamic_item, 0))
        self.assertEqual(cache.static_bytes, 80)
        self.assertEqual(cache.dynamic_bytes, 0)

    def test_dynamic_entries_are_lru_evicted_by_max_bytes(self):
        """
        Test dynamic LRU eviction when the shared byte budget is exceeded.
        """
        item = DummyItem("dynamic", static=False, nbytes=40)
        cache = MeshFrameCache(max_bytes=100)

        cache.get(item, 0)
        cache.get(item, 1)
        cache.get(item, 2)

        self.assertFalse(cache.contains(item, 0))
        self.assertTrue(cache.contains(item, 1))
        self.assertTrue(cache.contains(item, 2))
        self.assertEqual(cache.dynamic_bytes, 80)

    def test_get_with_status_reports_eviction(self):
        """
        Test that preloading can detect when a cache insert caused eviction.
        """
        item = DummyItem("dynamic", static=False, nbytes=40)
        cache = MeshFrameCache(max_bytes=100)

        first = cache.get_with_status(item, 0)
        second = cache.get_with_status(item, 1)
        third = cache.get_with_status(item, 2)

        self.assertTrue(first.created)
        self.assertFalse(first.evicted)
        self.assertTrue(second.created)
        self.assertFalse(second.evicted)
        self.assertTrue(third.created)
        self.assertTrue(third.evicted)

    def test_get_with_status_can_stop_before_eviction(self):
        """
        Test no-eviction insertion reports an unstored mesh before evicting entries.
        """
        item = DummyItem("dynamic", static=False, nbytes=40)
        cache = MeshFrameCache(max_bytes=100)

        cache.get_with_status(item, 0, allow_eviction=False)
        cache.get_with_status(item, 1, allow_eviction=False)
        result = cache.get_with_status(item, 2, allow_eviction=False)

        self.assertFalse(result.created)
        self.assertFalse(result.stored)
        self.assertTrue(cache.contains(item, 0))
        self.assertTrue(cache.contains(item, 1))
        self.assertFalse(cache.contains(item, 2))
        self.assertEqual(cache.dynamic_bytes, 80)

    def test_evict_dynamic_frame_removes_only_that_frame(self):
        """
        Test exact frame eviction removes all dynamic entries for one frame only.
        """
        item = DummyItem("dynamic", static=False, nbytes=10)
        cache = MeshFrameCache(max_bytes=100)

        for frame_index in range(4):
            cache.get(item, frame_index)

        evicted = cache.evict_dynamic_frame(2)

        self.assertEqual(evicted, {2})
        self.assertTrue(cache.contains(item, 0))
        self.assertTrue(cache.contains(item, 1))
        self.assertFalse(cache.contains(item, 2))
        self.assertTrue(cache.contains(item, 3))
        self.assertEqual(cache.dynamic_bytes, 30)

    def test_evict_dynamic_frames_not_in_retained_set(self):
        """
        Test explicit frame-window eviction for dynamic cache entries.
        """
        item = DummyItem("dynamic", static=False, nbytes=10)
        cache = MeshFrameCache(max_bytes=100)

        for frame_index in range(4):
            cache.get(item, frame_index)

        evicted = cache.evict_dynamic_frames_not_in({1, 3})

        self.assertEqual(evicted, {0, 2})
        self.assertFalse(cache.contains(item, 0))
        self.assertTrue(cache.contains(item, 1))
        self.assertFalse(cache.contains(item, 2))
        self.assertTrue(cache.contains(item, 3))
        self.assertEqual(cache.dynamic_bytes, 20)


if __name__ == "__main__":
    unittest.main()
