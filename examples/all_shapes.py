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
    Run the all shapes example.
    """

    material_1 = vis.Material(colour=np.array([30, 144, 255]) / 255, alpha=0.9)
    # Define the sphere.
    trajectory = np.random.uniform(-10, 10, (10, 1, 3))
    mesh = vis.Sphere(radius=2.0, material=material_1, resolution=30)
    particle = vis.Particle(name="Sphere", mesh=mesh, position=trajectory)

    material_2 = vis.Material(colour=np.array([255, 140, 0]) / 255, alpha=1.0)
    # Define the cylinder.
    trajectory_2 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_2 = vis.Cylinder(
        radius=1.0, height=2.0, split=1, material=material_2, resolution=30
    )
    particle_2 = vis.Particle(name="Cylinder", mesh=mesh_2, position=trajectory_2)

    material_3 = vis.Material(colour=np.array([100, 255, 130]) / 255, alpha=1.0)
    # Define the icosahedron.
    trajectory_3 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_3 = vis.Icosahedron(radius=2.0, material=material_3)
    particle_3 = vis.Particle(name="Icosahedron", mesh=mesh_3, position=trajectory_3)

    material_4 = vis.Material(colour=np.array([255, 200, 50]) / 255, alpha=1.0)
    # Define the torus.
    trajectory_4 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_4 = vis.Torus(
        torus_radius=1.0,
        tube_radius=0.5,
        tubular_resolution=30,
        radial_resolution=30,
        material=material_4,
    )
    particle_4 = vis.Particle(name="Torus", mesh=mesh_4, position=trajectory_4)

    material_5 = vis.Material(colour=np.array([250, 50, 20]) / 255, alpha=1.0)
    # Define the mobius loop.
    trajectory_5 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_5 = vis.MobiusLoop(
        twists=3,
        radius=2,
        flatness=1,
        width=2,
        scale=1,
        length_split=200,
        width_split=200,
        material=material_5,
    )
    particle_5 = vis.Particle(name="MobiusLoop", mesh=mesh_5, position=trajectory_5)

    material_6 = vis.Material(colour=np.array([255, 90, 255]) / 255, alpha=1.0)
    # Define the octahedron.
    trajectory_6 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_6 = vis.Octahedron(radius=2.0, material=material_6)
    particle_6 = vis.Particle(name="Octahedron", mesh=mesh_6, position=trajectory_6)

    material_7 = vis.Material(colour=np.array([255, 220, 100]) / 255, alpha=1.0)
    # Define the tetrahedron.
    trajectory_7 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_7 = vis.Tetrahedron(radius=2.0, material=material_7)
    particle_7 = vis.Particle(name="Tetrahedron", mesh=mesh_7, position=trajectory_7)

    material_8 = vis.Material(colour=np.array([255, 200, 240]) / 255, alpha=1.0)
    # Define the arrow.
    trajectory_8 = np.random.uniform(-10, 10, (10, 1, 3))
    direction_8 = np.random.uniform(-1, 1, (10, 1, 3))
    mesh_8 = vis.Arrow(scale=2, material=material_8, resolution=30)
    particle_8 = vis.Particle(
        name="Arrow", mesh=mesh_8, position=trajectory_8, director=direction_8
    )

    material_9 = vis.Material(colour=np.array([150, 255, 230]) / 255, alpha=1.0)
    # Define the box.
    trajectory_9 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_9 = vis.Box(width=1, height=3, depth=0.5, material=material_9)
    particle_9 = vis.Particle(name="BoxMesh", mesh=mesh_9, position=trajectory_9)

    material_10 = vis.Material(colour=np.array([255, 10, 100]) / 255, alpha=1.0)
    # Define the cone.
    trajectory_10 = np.random.uniform(-10, 10, (10, 1, 3))
    mesh_10 = vis.Cone(radius=1.0, height=2.0, material=material_10, resolution=30)
    particle_10 = vis.Particle(name="Cone", mesh=mesh_10, position=trajectory_10)

    particle_list = [
        particle,
        particle_2,
        particle_3,
        particle_4,
        particle_5,
        particle_6,
        particle_7,
        particle_8,
        particle_9,
        particle_10,
    ]

    # Create a bounding box
    bounding_box = vis.BoundingBox(
        center=np.array([0, 0, 0]), box_size=np.array([20, 20, 20])
    )

    # Construct the visualizer and run
    visualizer = vis.Visualizer(
        particles=particle_list, frame_rate=20, bounding_box=bounding_box
    )
    visualizer.run_visualization()
