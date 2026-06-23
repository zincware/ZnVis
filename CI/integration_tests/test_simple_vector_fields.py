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
Test the visualization of simple spheres.
"""

import time

import numpy as np

import znvis as vis
from znvis import Material
from znvis.testing.znvis_process import Process


class TestSimpleVectorFields:
    """
    A test class for the Particle class.
    """

    def test_run(self):
        """
        Run the simple vector fields tutorial and ensure it does not throw
        exceptions.

        Returns
        -------

        """
        process = Process(target=self.run_process)
        process.start()
        time.sleep(10)  # give it 10 seconds to run.
        process.terminate()
        if process.exception:
            error, traceback = process.exception
            print(traceback)
        assert process.exception is None

    @staticmethod
    def run_process():
        """
        Execute the run on the given process.
        """

        material = Material(colour=np.array([0, 0, 255]) / 255)
        dyn_material = Material(colour=np.array([255, 0, 0]) / 255)
        x_dim = 5
        y_dim = 5
        n_frames = 36
        x_grid = np.arange(0, x_dim, 1)
        y_grid = np.arange(0, y_dim, 1)
        pos_grid = np.array(np.meshgrid(x_grid, y_grid)).T
        pos = np.zeros((x_dim, y_dim, 3))
        pos[:, :, 0] = pos_grid[:, :, 0]
        pos[:, :, 1] = pos_grid[:, :, 1]
        pos[:, :, 2] = -5

        pos = pos.reshape(-1, pos.shape[-1])
        rotation_angle = np.radians(10)
        R_z = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1],
            ]
        )

        dynamic_pos = np.zeros((n_frames, pos.shape[0], pos.shape[1]))
        for i in range(dynamic_pos.shape[0]):
            dynamic_pos[i, :, :] = pos[np.newaxis, :] - np.array([10, 10, 0])
        dynamic_directors = np.zeros((n_frames, pos.shape[0], pos.shape[1]))
        static_directors = np.full_like(pos, np.array([0, 0, 1]))
        dynamic_directors[0, :, :] = static_directors[np.newaxis, :]
        dynamic_directors[0, :, :] = np.array([1, 1, 1]) / np.linalg.norm(
            np.array([1, 1, 1])
        )
        for i in range(1, n_frames):
            for j in range(dynamic_directors.shape[1]):
                last_vec = dynamic_directors[i - 1, j, :]
                new_vec = np.dot(R_z, last_vec)
                new_vec = new_vec / np.linalg.norm(new_vec)
                dynamic_directors[i, j, :] = new_vec

        v_mesh = vis.Arrow(scale=1, material=material)
        dyn_mesh = vis.Arrow(scale=1, material=dyn_material)

        static_field = vis.VectorField(
            "Static", mesh=v_mesh, position=pos, direction=static_directors, static=True
        )
        dynamic_field = vis.VectorField(
            "Dynamic", mesh=dyn_mesh, position=dynamic_pos, direction=dynamic_directors
        )

        # Create a bounding box
        bounding_box = vis.BoundingBox(
            center=np.array([0, 0, 0]),
            box_size=np.array([20, 20, 20]),
            colour=np.array([0.7, 0.3, 0.1]),
        )

        # Construct the visualizer and run
        visualizer = vis.Visualizer(
            particles=[],
            vector_field=[
                static_field,
                dynamic_field,
            ],
            frame_rate=20,
            bounding_box=bounding_box,
        )
        visualizer.run_visualization()
