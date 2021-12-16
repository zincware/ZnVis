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

if __name__ == '__main__':
    """
    Run the simple spheres example.
    """
    trajectory = np.random.uniform(-10, 10, (1000, 10, 3))
    trajectory_2 = np.random.uniform(10, 20, (1000, 10, 3))
    mesh = vis.Sphere(radius=2.0, colour=np.array([30, 144, 255]) / 255, resolution=10)
    mesh_2 = vis.Sphere(
        radius=2.0, colour=np.array([255, 140, 0]) / 255, resolution=10
    )
    particle = vis.Particle(name="Ball", mesh=mesh, position=trajectory)
    particle_2 = vis.Particle(name="Ball_O", mesh=mesh_2, position=trajectory_2)

    visualizer = vis.Visualizer(particles=[particle, particle_2])
    visualizer.run_visualization()
