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
Tutorial on reading custom mesh objects.
"""
import numpy as np

import znvis as vis

if __name__ == "__main__":
    """
    Run the custom mesh example.
    """
    # Define the particle type.
    virus_trajectory = np.random.uniform(-10, 10, (10, 1, 3))
    virus_orientation = np.random.uniform(0, 1, (10, 1, 3))

    material = vis.Material(colour=np.array([30, 144, 255]) / 255, alpha=0.7)

    virus_mesh = vis.CustomMesh(file="example_data/virus.stl", material=material)
    virus = vis.Particle(
        name="Virus",
        mesh=virus_mesh,
        position=virus_trajectory,
        director=virus_orientation,
    )

    # Construct the visualizer and run
    visualizer = vis.Visualizer(particles=[virus], frame_rate=10)
    visualizer.run_visualization()
