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
    Run the simple spheres example.
    """
    material_1 = vis.Material(colour=np.array([30, 144, 255]) / 255, alpha=0.6)
    # Define the first particle.
    trajectory = np.random.uniform(-100, 100, (100, 1000, 3))
    trajectory = []
    p_number = 1
    for t in range(1000):
        trajectory.append(np.random.uniform(-100, 100, (p_number, 3)))

        p_number = np.random.randint(p_number, 5 * p_number)

        if p_number > 20000:
            p_number = 20000

    mesh = vis.Sphere(radius=2.0, resolution=3, material=material_1)
    particle = vis.Particle(
        name="Blue", mesh=mesh, position=trajectory, smoothing=False
    )

    # Create a bounding box
    bounding_box = vis.BoundingBox(
        center=np.array([0, 0, 0]), box_size=np.array([100, 100, 100])
    )

    # Construct the visualizer and run
    visualizer = vis.Visualizer(
        particles=[particle], frame_rate=20, bounding_box=bounding_box
    )
    visualizer.run_visualization()
