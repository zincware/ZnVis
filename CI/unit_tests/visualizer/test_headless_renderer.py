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

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from znvis.material import Material
from znvis.mesh.arrow import Arrow
from znvis.mesh.sphere import Sphere
from znvis.particle.particle import Particle
from znvis.particle.vector_field import VectorField
from znvis.visualizer.headless_visualizer import HeadlessVisualizer


class TestHeadlessVisualizer(unittest.TestCase):
    """
    A test class for the Particle class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set up the class for testing.
        """
        cls.material = Material(colour=np.array([0, 0, 255]) / 255)
        project_root = Path(__file__).resolve().parents[2]

        position = np.random.uniform(-5, 5, (10, 2, 3))
        particle1 = Particle(
            name="my_particle", position=position.copy(), mesh=Sphere()
        )
        static_pos = np.random.uniform(-5, 5, (2, 3))
        particle2 = Particle(
            name="static_particle",
            position=static_pos.copy(),
            mesh=Sphere(),
            static=True,
        )
        x_dim = 5
        y_dim = 5
        n_frames = 10
        x_grid = np.arange(0, x_dim, 1)
        y_grid = np.arange(0, y_dim, 1)
        pos_grid = np.array(np.meshgrid(x_grid, y_grid)).T
        pos = np.zeros((x_dim, y_dim, 3))
        pos[:, :, 0] = pos_grid[:, :, 0]
        pos[:, :, 1] = pos_grid[:, :, 1]
        pos[:, :, 2] = -5

        pos = pos.reshape(-1, pos.shape[-1])
        static_directors = np.full_like(pos, np.array([0, 0, 1]))

        static_vector_field = VectorField(
            name="static_field",
            position=pos,
            mesh=Arrow(scale=10, material=cls.material),
            direction=static_directors,
            static=True,
        )

        dynamic_pos = np.zeros((n_frames, pos.shape[0], pos.shape[1]))
        for i in range(dynamic_pos.shape[0]):
            dynamic_pos[i, :, :] = pos[np.newaxis, :] - np.array([10, 10, 0])
        dynamic_directors = np.zeros((n_frames, pos.shape[0], pos.shape[1]))
        dynamic_directors[0, :, :] = static_directors[np.newaxis, :]
        dynamic_directors[0, :, :] = np.array([1, 1, 1]) / np.linalg.norm(
            np.array([1, 1, 1])
        )

        dynamic_vector_field = VectorField(
            name="dynamic_field",
            position=dynamic_pos,
            mesh=Arrow(scale=10, material=cls.material),
            direction=dynamic_directors,
        )

        save_path = project_root / "test_files" / "headless_visualizer" / "basic_test"

        cls.visualizer = HeadlessVisualizer(
            particles=[particle1, particle2],
            vector_field=[static_vector_field, dynamic_vector_field],
            frame_rate=10,
            renderer_resolution=[192, 108],
            output_folder=save_path,
            video_format="mp4",
            renderer_spp=64,
            keep_frames=True,
        )

        particle1 = Particle(name="my_particle", position=position, mesh=Sphere())
        particle2 = Particle(
            name="static_particle", position=static_pos, mesh=Sphere(), static=True
        )
        static_vector_field = VectorField(
            name="static_field",
            position=pos,
            mesh=Arrow(scale=10, material=cls.material),
            direction=static_directors,
            static=True,
        )
        dynamic_vector_field = VectorField(
            name="dynamic_field",
            position=dynamic_pos,
            mesh=Arrow(scale=10, material=cls.material),
            direction=dynamic_directors,
        )

        save_path = (
            project_root / "test_files" / "headless_visualizer" / "delete_frames_test"
        )
        cls.visualizer_delete_frames = HeadlessVisualizer(
            particles=[particle1, particle2],
            vector_field=[static_vector_field, dynamic_vector_field],
            frame_rate=10,
            renderer_resolution=[192, 108],
            output_folder=save_path,
            video_format="mp4",
            renderer_spp=64,
            keep_frames=False,
        )

        particle1 = Particle(
            name="my_particle", position=static_pos, mesh=Sphere(), static=True
        )

        save_path = project_root / "test_files" / "headless_visualizer" / "empty_test"
        cls.visualizer_empty = HeadlessVisualizer(
            particles=[particle1],
            frame_rate=10,
            renderer_resolution=[192, 108],
            output_folder=save_path,
            video_format="mp4",
            renderer_spp=64,
            keep_frames=True,
            do_create_video=False,
        )

    def test_instantiation(self):
        """
        Test the class instantiation.

        Returns
        -------
        Check the class was built correctly.
        """
        self.assertEqual(self.visualizer.number_of_steps, 10)
        self.assertEqual(self.visualizer.counter, 0)
        self.assertEqual(self.visualizer.renderer_resolution, [192, 108])
        self.assertEqual(
            self.visualizer.output_folder,
            Path(__file__).resolve().parents[2]
            / "test_files"
            / "headless_visualizer"
            / "basic_test",
        )
        self.assertEqual(self.visualizer.video_format, "mp4")
        self.assertEqual(self.visualizer.renderer_spp, 64)
        self.assertEqual(self.visualizer.keep_frames, True)
        self.assertEqual(self.visualizer.parallel_render_workers, 2)
        self.assertFalse(self.visualizer.parallel_render_enabled)

    def test_render_dispatch_uses_serial_by_default(self):
        """
        Test that serial rendering is used when parallel rendering is disabled.
        """
        with (
            patch.object(self.visualizer, "_render_frames_serial") as serial_mock,
            patch.object(self.visualizer, "_render_frames_parallel") as parallel_mock,
        ):
            self.visualizer.do_create_video = False
            self.visualizer.parallel_render_enabled = False
            self.visualizer._record_trajectory()

        serial_mock.assert_called_once()
        parallel_mock.assert_not_called()

    def test_render_dispatch_uses_parallel_when_enabled(self):
        """
        Test that parallel rendering is used when explicitly enabled.
        """
        with (
            patch.object(self.visualizer, "_render_frames_serial") as serial_mock,
            patch.object(self.visualizer, "_render_frames_parallel") as parallel_mock,
        ):
            self.visualizer.do_create_video = False
            self.visualizer.parallel_render_enabled = True
            self.visualizer._record_trajectory()

        parallel_mock.assert_called_once()
        serial_mock.assert_not_called()

    def test_headless_rendering(self):
        """
        Test the headless rendering process.
        """
        self.visualizer.render_visualization()

        self.assertTrue(
            (
                self.visualizer.output_folder / "video_frames" / "frame_000000.png"
            ).exists()
        )
        self.assertTrue(
            (
                self.visualizer.output_folder / "video_frames" / "frame_000009.png"
            ).exists()
        )

        # Delete all content of the output folder
        for file in self.visualizer.output_folder.glob("video_frames/*"):
            file.unlink()
        for file in self.visualizer.output_folder.glob("video_frames"):
            file.rmdir()

    def test_keep_frames(self):
        """
        Test the keep frames option.
        """
        self.visualizer_delete_frames.render_visualization()

        self.assertFalse(
            (
                self.visualizer_delete_frames.output_folder
                / "video_frames"
                / "frame_000000.png"
            ).exists()
        )
        self.assertFalse(
            (
                self.visualizer_delete_frames.output_folder
                / "video_frames"
                / "frame_000009.png"
            ).exists()
        )

    def test_empty_visualizer(self):
        """
        Test the empty visualizer.
        """
        self.visualizer_empty.render_visualization()

        self.assertTrue(
            (
                self.visualizer_empty.output_folder
                / "video_frames"
                / "frame_000000.png"
            ).exists()
        )
        self.assertFalse(
            (
                self.visualizer_empty.output_folder
                / "video_frames"
                / "frame_000001.png"
            ).exists()
        )

        for file in self.visualizer_empty.output_folder.glob("video_frames/*"):
            file.unlink()
        for file in self.visualizer_empty.output_folder.glob("video_frames"):
            file.rmdir()

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Clean up after all tests have run.
        """
        import shutil

        project_root = Path(__file__).resolve().parents[2]
        test_output_dir = project_root / "test_files" / "headless_visualizer"

        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
