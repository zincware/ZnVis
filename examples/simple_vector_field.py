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
Tutorial script to visualize simple spheres over a random trajectory.
"""

import numpy as np

import znvis as vis

if __name__ == "__main__":
    """
    Run the vector field example.
    """

    # Build a grid
    x_values = np.linspace(-10, 10, 21)
    y_values = np.linspace(-10, 10, 21)
    z_values = np.linspace(0, 0, 1)

    grid = np.meshgrid(x_values, y_values, z_values)
    grid = np.array(grid).T.reshape(-1, 3)
    grid = np.tile(grid, (100, 1, 1))

    # Define arrow mesh and insert in vector field
    material = vis.Material(colour=np.array([30, 144, 255]) / 255, alpha=0.6)
    mesh = vis.Arrow(scale=0.5, resolution=20, material=material)

    directions = np.random.uniform(-1, 1, (100, 441, 3))
    # confine the directions to be in the z = 0 plane
    directions[:,:,2] = 0

    vector_field = vis.VectorField(name="VectorField",
                                   mesh=mesh,
                                   position=grid,
                                   direction=directions)

    # Define particles
    material_2 = vis.Material(colour=np.array([255, 140, 0]) / 255, alpha=1.0)
    mesh_2 = vis.Sphere(radius=1.0, resolution=20, material=material_2)

    trajectory_2 = np.random.uniform(-10, 10, (100, 1, 3))
    # confine the particles to be in the z = 0 plane
    trajectory_2[:,:,2] = 0

    particle = vis.Particle(name="Spheres",
                            mesh=mesh_2,
                            position=trajectory_2,
                            smoothing=False)

    # Construct the visualizer and run
    visualizer = vis.Visualizer(particles=[particle],
                                vector_field=[vector_field],
                                frame_rate=20)
    visualizer.run_visualization()