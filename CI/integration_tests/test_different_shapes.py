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


class TestDifferentShapes:
    """
    A test class for the Particle class.
    """

    def test_run(self):
        """
        Run the different_shapes tutorial and ensure it does not throw
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
        # Define several meshes
        names = [
            "Box",
            "Cone",
            "Cylinder",
            "Icosahedron",
            "MobiusLoop",
            "Octahedron",
            "Sphere",
            "Tetrahedron",
            "Torus",
        ]
        meshs = [
            vis.Box,
            vis.Cone,
            vis.Cylinder,
            vis.Icosahedron,
            vis.MobiusLoop,
            vis.Octahedron,
            vis.Sphere,
            vis.Tetrahedron,
            vis.Torus,
        ]

        particle_list = []
        for name, mesh in zip(names, meshs):

            trajectory = np.random.uniform(-10, 10, (10, 10, 3))
            material = vis.Material(
                colour=np.random.uniform(0, 255, (3)) / 255, alpha=0.9
            )
            mesh = mesh(material=material)
            if name == "Octahedron":
                mesh = vis.Octahedron(material=material)
                particle_list.append(
                    vis.Particle(
                        name=name,
                        mesh=mesh,
                        position=np.random.uniform(-20, 20, (10, 3)),
                        static=True,
                    )
                )
            else:
                particle_list.append(
                    vis.Particle(name=name, mesh=mesh, position=trajectory)
                )

        trajectory = np.zeros((10, 3))
        material = vis.Material(colour=np.array([136, 8, 8]) / 255, alpha=1)
        mesh = vis.CustomMesh(file="../test_files/red_blood_particle.obj", scale=0.7)
        particle_list.append(
            vis.Particle(name="Custom", mesh=mesh, position=trajectory, static=True)
        )

        # Create a bounding box
        bounding_box = vis.BoundingBox(
            center=np.array([0, 0, 0]),
            box_size=np.array([20, 20, 20]),
            colour=np.array([0.7, 0.3, 0.1]),
        )

        # Construct the visualizer and run
        visualizer = vis.Visualizer(
            particles=particle_list,
            frame_rate=20,
            bounding_box=bounding_box,
        )
        visualizer.run_visualization()
