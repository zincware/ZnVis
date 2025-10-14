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
Module for the zoom-in camera trajectory.
"""

import numpy as np

from znvis.cameras.trajectories.base_trajectory import BaseTrajectory


class ZoomingTrajectory(BaseTrajectory):
    """
    Class to generate a zooming camera trajectory.
    The camera zooms from a certain distance to another.
    """

    def __init__(
        self,
        total_frames: int,
        center: np.ndarray,
        initial_eye: np.ndarray,
        zoom_distance: float,
        up: np.ndarray = None,
        zoom_fraction: float = 1.0,
    ):
        """
        Initialize the ZoomingCamera object.

        Parameters
        ----------
        total_frames : int
            The number of frames in which the trajectory should be recorded.
        center : np.ndarray
            The center of the camera zoom.
        initial_eye : np.ndarray
            The eye of the camera, aka its position.
        zoom_distance : float
            The distance travelled in the zoom process from
            the initial eye towards the center.
        up : np.ndarray
            The eye vector of the camera. Defines where up is. Gets set to
            [0, 1, 0] if None is passed.
        zoom_fraction : float
            The fraction of the total frames that the camera
            will spend zooming in or out. By default 1.0.
        """
        self.total_frames = total_frames
        self.center = np.array(center)
        self.initial_eye = np.array(initial_eye)
        self.zoom_distance = zoom_distance
        self.up = np.array([0, 1, 0]) if up is None else np.array(up)
        direction = self.center - self.initial_eye
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            raise ValueError("center and initial_eye must not be identical")
        self.direction = direction / direction_norm

        self.end_eye = self.initial_eye + self.direction * self.zoom_distance

        if zoom_fraction is not None:
            self.zoom_fraction = zoom_fraction
            self.number_of_zoom_frames = int(self.zoom_fraction * self.total_frames)
            if self.number_of_zoom_frames < 1:
                raise ValueError(
                    "The number of frames while zooming must be at least 1."
                )
        else:
            self.number_of_zoom_frames = self.total_frames

        distance = np.linalg.norm(self.initial_eye - self.end_eye)
        denom = max(1, self.number_of_zoom_frames - 1)
        self.step_size = distance / denom
        if denom > 1 and self.step_size == 0:
            raise ValueError('"zoom_distance" too small for the number of frames.')

    def get_center_eye_up(self, frame_index: int) -> tuple:
        """
        Provides the view matrix for a given frame index to
        zoom in or out.
        Parameters
        ----------
        frame_index : int
            The index of the frame for which to generate the view matrix.

        Returns
        -------
        (center, eye, up): tuple
            The center, eye, and up vectors of the camera.
        """

        if frame_index >= self.number_of_zoom_frames:
            frame_index = self.number_of_zoom_frames - 1
        if frame_index < 0:
            frame_index = 0

        eye = self.initial_eye + self.step_size * frame_index * self.direction
        center = self.center

        return center, eye, self.up
