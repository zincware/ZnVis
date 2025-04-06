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
Package for the ZnVis Cameras.
"""

from .camera import Camera
from .keyframe_camera import KeyframeCamera
from .particle_following_camera import ParticleFollowingCamera
from .trajectory_camera import TrajectoryCamera

__all__ = [
    "Camera",
    "KeyframeCamera",
    "TrajectoryCamera",
    "ParticleFollowingCamera",
]
