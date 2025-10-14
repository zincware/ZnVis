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
Module for the TrajectoryCamera class.
"""

import numpy as np

from znvis.cameras.base_camera import BaseCamera
from znvis.cameras.trajectories.base_trajectory import BaseTrajectory


class TrajectoryCamera(BaseCamera):
    """
    A class to produce camera trajectories.
    """

    def __init__(self, trajectory: BaseTrajectory):
        """
        Initializes the TrajectoryCamera object.
        Parameters
        ----------
        trajectory : BaseCameraTrajectory
            The camera trajectory object.
        total_frames : int
            The total number of frames in the trajectory.
        """
        self.trajectory = trajectory
        self.view_matrix = self.get_view_matrix(0)

    def get_view_matrix(self, frame_index: int) -> np.ndarray:
        center, eye, up = self.trajectory.get_center_eye_up(frame_index)
        self.view_matrix = self.look_at(center, eye, up)
        return self.view_matrix
