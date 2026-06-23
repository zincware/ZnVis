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

from znvis.camera_trajectories.base_trajectory import BaseTrajectory
from znvis.camera_trajectories.circular_trajectory import CircularTrajectory
from znvis.camera_trajectories.zooming_trajectory import ZoomingTrajectory

__all__ = [
    BaseTrajectory.__name__,
    CircularTrajectory.__name__,
    ZoomingTrajectory.__name__,
]
