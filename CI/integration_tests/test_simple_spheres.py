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
from znvis.testing.znvis_process import Process


class TestSimpleSpheres:
    """
    A test class for the Particle class.
    """

    def test_run(self):
        """
        Run the simple spheres tutorial and ensure it does not throw
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
        # Define the first particle.
        trajectory = np.random.uniform(-10, 10, (10, 10, 3))
        material_1 = vis.Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)

        mesh = vis.Sphere(radius=2.0, material=material_1, resolution=3)
        particle = vis.Particle(name="Blue", mesh=mesh, position=trajectory)

        # Define the second particle.
        material_2 = vis.Material(colour=np.array([255, 140, 0]) / 255, alpha=1.0)
        trajectory_2 = np.random.uniform(-10, 10, (10, 10, 3))
        mesh_2 = vis.Sphere(radius=1.0, material=material_2, resolution=3)
        particle_2 = vis.Particle(name="Orange", mesh=mesh_2, position=trajectory_2)

        # Create a bounding box
        bounding_box = vis.BoundingBox(
            center=np.array([0, 0, 0]),
            box_size=np.array([20, 20, 20]),
            colour=np.array([0.7, 0.3, 0.1]),
        )

        # Construct the visualizer and run
        visualizer = vis.Visualizer(
            particles=[particle, particle_2],
            frame_rate=20,
            bounding_box=bounding_box,
        )
        visualizer.run_visualization()
