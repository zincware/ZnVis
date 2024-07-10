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
    Run the dynamic color example.
    """

    # Create a color list (N_frames, N_particles, 3 (RGB))
    # Basically give each particle a specified color for each frame
    colours = np.tile([30, 144, 255], (100, 5, 1))
    # Change the color of the first particle to red
    colours[:,0,0] = np.linspace(30, 255, 100)
    # Change the color of the second particle to green
    colours[:,1,1] = np.linspace(144, 255, 100)
    colours[:,1,2] = np.linspace(255, 30, 100)
    # Change the color of the third particle to blue
    colours[:,2,0] = np.linspace(30, 10, 100)
    colours[:,2,1] = np.linspace(140, 90, 100)
    # Change the color of the fourth particle to white
    colours[:,3,0] = np.linspace(30, 255, 100)
    colours[:,3,1] = np.linspace(144, 255, 100)
    # Change the color of the fifth particle to black
    colours[:,4,0] = np.linspace(30, 0, 100)
    colours[:,4,1] = np.linspace(144, 0, 100)
    colours[:,4,2] = np.linspace(255, 0, 100)

    material_1 = vis.Material(colour=colours / 255, alpha=1.0)
    # Define the first particle.
    trajectory = np.random.uniform(-5, 5, (1, 5, 3))
    trajectory = np.tile(trajectory, (100, 1, 1))
    # Turn on dynamic coloring for the mesh
    mesh = vis.Sphere(radius=2.0,
                      resolution=20,
                      material=material_1,
                      dynamic_color=True)
    particle = vis.Particle(
        name="Spheres", mesh=mesh, position=trajectory, smoothing=False
    )

    # Construct the visualizer and run
    visualizer = vis.Visualizer(particles=[particle], frame_rate=20)
    visualizer.run_visualization()