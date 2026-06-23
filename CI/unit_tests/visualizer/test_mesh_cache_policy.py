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
Tests for pure lazy mesh cache policy math.
"""

import unittest

from znvis.mesh_cache.mesh_cache_policy import MeshCachePolicy


class TestMeshCachePolicy(unittest.TestCase):
    """
    Test policy math independently from worker/thread management.
    """

    def test_startup_balanced_fill_wraps_into_past_at_zero(self):
        """
        Test startup window order wraps into the trajectory tail.
        """
        policy = MeshCachePolicy(2 / 3)

        target = policy.get_target_window(
            current_frame=0,
            do_rewind=False,
            frame_step=1,
            number_of_steps=10,
            initial_cached_frame_count=10,
        )

        self.assertEqual(target.build_priority, list(range(10)))

    def test_rolling_window_uses_startup_depth_and_future_fraction(self):
        """
        Test forward rolling policy derives its window from startup depth.
        """
        policy = MeshCachePolicy(2 / 3)

        target = policy.get_target_window(
            current_frame=50,
            do_rewind=False,
            frame_step=1,
            number_of_steps=400,
            initial_cached_frame_count=100,
        )

        self.assertEqual(min(target.retained_frames), 17)
        self.assertEqual(max(target.retained_frames), 116)
        self.assertEqual(len(target.retained_frames), 100)
        self.assertEqual(target.build_priority[:5], [50, 51, 52, 53, 54])

    def test_rewind_window_flips_future_toward_lower_frames(self):
        """
        Test reverse rolling policy points future toward lower frame indices.
        """
        policy = MeshCachePolicy(2 / 3)

        target = policy.get_target_window(
            current_frame=150,
            do_rewind=True,
            frame_step=1,
            number_of_steps=400,
            initial_cached_frame_count=100,
        )

        self.assertEqual(min(target.retained_frames), 117)
        self.assertEqual(max(target.retained_frames), 216)
        self.assertEqual(len(target.retained_frames), 100)
        self.assertEqual(target.build_priority[:5], [150, 149, 148, 147, 146])

    def test_speed_scaled_priority_uses_playback_stride(self):
        """
        Test fast playback prioritizes the stride implied by playback speed.
        """
        policy = MeshCachePolicy(2 / 3)

        target = policy.get_target_window(
            current_frame=50,
            do_rewind=False,
            frame_step=8,
            number_of_steps=200,
            initial_cached_frame_count=100,
        )

        self.assertEqual(target.build_priority[:5], [50, 58, 66, 74, 82])


if __name__ == "__main__":
    unittest.main()
