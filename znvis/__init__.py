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
init file for the main ZnVis package.
"""

from znvis import rendering
from znvis.bounding_objects.bounding_box import BoundingBox
from znvis.cameras.camera import Camera
from znvis.cameras.keyframe_camera import KeyframeCamera
from znvis.cameras.particle_following_camera import ParticleFollowingCamera
from znvis.cameras.trajectories.base_trajectory import BaseTrajectory
from znvis.cameras.trajectories.circular_trajectory import CircularTrajectory
from znvis.cameras.trajectories.zooming_trajectory import ZoomingTrajectory
from znvis.cameras.trajectory_camera import TrajectoryCamera
from znvis.material.material import Material
from znvis.mesh.arrow import Arrow
from znvis.mesh.box import Box
from znvis.mesh.cone import Cone
from znvis.mesh.custom import CustomMesh
from znvis.mesh.cylinder import Cylinder
from znvis.mesh.icosahedron import Icosahedron
from znvis.mesh.mobius_loop import MobiusLoop
from znvis.mesh.octahedron import Octahedron
from znvis.mesh.sphere import Sphere
from znvis.mesh.tetrahedron import Tetrahedron
from znvis.mesh.torus import Torus
from znvis.particle.particle import Particle
from znvis.particle.vector_field import VectorField
from znvis.visualizer.headless_visualizer import Headless_Visualizer
from znvis.visualizer.visualizer import Visualizer

__all__ = [
    Particle.__name__,
    Sphere.__name__,
    Arrow.__name__,
    Box.__name__,
    Cone.__name__,
    Tetrahedron.__name__,
    Octahedron.__name__,
    Icosahedron.__name__,
    Torus.__name__,
    MobiusLoop.__name__,
    VectorField.__name__,
    Visualizer.__name__,
    Headless_Visualizer.__name__,
    Cylinder.__name__,
    CustomMesh.__name__,
    BoundingBox.__name__,
    Material.__name__,
    rendering.__name__,
    Camera.__name__,
    KeyframeCamera.__name__,
    TrajectoryCamera.__name__,
    ParticleFollowingCamera.__name__,
    BaseTrajectory.__name__,
    ZoomingTrajectory.__name__,
    CircularTrajectory.__name__,
]
