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
Module for the circular camera trajectory.
"""

import numpy as np

from .base_trajectory import BaseTrajectory


class CircularTrajectory(BaseTrajectory):
    """
    Class to generate a circular camera trajectory.
    The camera rotates around a center point in a circular motion in the desired plane.
    The camera's up vector is fixed.
    """

    def __init__(
        self,
        total_frames: int,
        center: np.ndarray,
        radius: float,
        frames_per_rotation: int,
        angle_range: tuple = (0, 2 * np.pi),
        loop: bool = False,
        ping_pong: bool = False,
        axis: str = "y",
        clockwise: bool = True,
        smoothing: bool = True,
    ) -> None:
        """
        Initialize the RotatingCamera object.
        Parameters
        ----------
        total_frames : int
            The number of frames in the trajectory.
        center: np.ndarray
            The center of the circular trajectory.
        radius: float
            The radius of the circular trajectory.
        frames_per_rotation: int
            The number of frames per full rotation.
        angle_range: tuple (float, float)
            The range of angles for the trajectory, by default (0, 2 * np.pi).
        loop: bool, optional
            Whether to loop the trajectory, by default False.
        ping_pong: bool, optional
            Whether to ping pong the trajectory, by default False.
        axis: str
            The axis around which the camera rotates. Options are 'x', 'y', or 'z'.
        clockwise: bool, optional
            Whether the rotation of the camera is clockwise, by default True.
            Note that the scene then will rotate counter-clockwise and vice versa.
        smoothing: bool
            Whether to apply smoothing to the trajectory, by default True.
        """
        self.total_frames = total_frames
        self.center = center
        self.radius = radius
        self.angle_range = angle_range
        self.loop = loop
        self.ping_pong = ping_pong
        self.axis = axis
        self.clockwise = clockwise
        self.smoothing = smoothing

        if (
            frames_per_rotation == 0
            or frames_per_rotation > self.total_frames
            or frames_per_rotation is None
        ):
            self.frames_per_rotation = self.total_frames
        else:
            self.frames_per_rotation = frames_per_rotation

        if self.axis not in ["x", "y", "z"]:
            raise ValueError("Axis must be one of 'x', 'y', or 'z'.")
        if self.frames_per_rotation < 1:
            raise ValueError("The number of frames per rotation must be at least 1.")

    def get_loop_and_ping_pong_frame_index(self, frame_index: int) -> int:
        """
        Adjust the frame index based on the loop and ping pong settings.

        Parameters
        ----------
        frame_index : int
            The index of the frame for which to adjust the frame index.

        Returns
        -------
        frame_index: int
            The adjusted frame index.
        """

        if self.loop and not self.ping_pong:
            return frame_index % self.frames_per_rotation

        elif self.loop and self.ping_pong:

            total_ping_pong_frames = 2 * self.frames_per_rotation
            frame_index = frame_index % total_ping_pong_frames
            if frame_index >= self.frames_per_rotation:
                frame_index = total_ping_pong_frames - frame_index
                return frame_index
            else:
                return frame_index

        elif not self.loop and not self.ping_pong:
            return min(frame_index, self.frames_per_rotation - 1)

        elif not self.loop and self.ping_pong:
            total_ping_pong_frames = 2 * self.frames_per_rotation
            if not frame_index // total_ping_pong_frames >= 1:
                frame_index = frame_index % total_ping_pong_frames
                if frame_index >= self.frames_per_rotation:
                    return total_ping_pong_frames - frame_index
                else:
                    return frame_index
            else:
                frame_index = 0
                return frame_index

    def get_center_eye_up(self, frame_index: int = None) -> tuple:
        """
        Provides the view matrix for the given frame index.

        Parameters
        ----------
        frame_index : int, optional
            The index of the frame for which to get the view matrix, by default None.

        Returns
        -------
        (center, eye, up): tuple
            The center, eye, and up vectors of the camera.

        """
        frame_index = self.get_loop_and_ping_pong_frame_index(frame_index)
        progress = frame_index / (self.frames_per_rotation - 1)

        start_theta = self.angle_range[0]
        end_theta = self.angle_range[1]
        if self.smoothing:
            progress = 0.5 - 0.5 * np.cos(np.pi * progress)

        theta = start_theta + (end_theta - start_theta) * frame_index / (
            self.frames_per_rotation - 1
        )

        if not self.clockwise:
            theta = -theta

        if self.axis == "z":
            eye = np.array(
                [self.radius * np.sin(theta), self.radius * np.cos(theta), 0]
            )
            up = np.array([0, 0, 1])
        elif self.axis == "y":
            eye = np.array(
                [self.radius * np.sin(theta), 0, self.radius * np.cos(theta)]
            )
            up = np.array([0, 1, 0])
        elif self.axis == "x":
            eye = np.array([0, self.radius * np.sin(theta), np.cos(theta)])
            up = np.array([1, 0, 0])

        if not self.clockwise:
            eye = eye[::-1]
        eye += self.center

        return self.center, eye, up
