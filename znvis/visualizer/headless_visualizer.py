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
Main visualizer class.
"""

import os

import znvis.cameras

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import pathlib
import time
import typing

import numpy as np
from rich.progress import Progress

import znvis
from znvis import cameras
from znvis.rendering import Mitsuba
from znvis.visualizer.base_visualizer import BaseVisualizer


class Headless_Visualizer(BaseVisualizer):
    """
    Main class to perform visualization.

    Attributes
    ----------
    particles : list[znvis.Particle]
            A list of particle objects to add to the visualizer.
    counter : int
            Internally stored counter to track which configuration is currently
            being viewed.
    """

    def __init__(
        self,
        particles: typing.List[znvis.Particle],
        vector_field: typing.List[znvis.VectorField] | None = None,
        output_folder: typing.Union[str, pathlib.Path] = "./",
        frame_rate: int = 24,
        number_of_steps: int | None = None,
        keep_frames: bool = True,
        bounding_box: znvis.BoundingBox | None = None,
        video_format: str = "mp4",
        video_title: str = "ZnVis-Video",
        renderer_resolution: typing.Sequence[int] | None = None,
        renderer_spp: int = 64,
        renderer: Mitsuba | None = None,
        do_create_video: bool = True,
        camera: cameras.BaseCamera | None = None,
    ):
        """
        Constructor for the visualizer.

        Parameters
        ----------
        particles : list[znvis.Particle]
                List of particles to add to the visualizer.
        vector_field : list[znvis.VectorField]
                List of vector fields to add to the visualizer.
        output_folder: typing.Union[str, pathlib.Path]
                Path to the output folder where the video and frames will be saved.
        frame_rate : int
                Frame rate for the visualizer measured in frames per second (fps)
        number_of_steps : int
                Number of steps in the visualization. If None, the zeroth order of one
                particle is taken. This is left as an option in case the user wishes
                to overlay two particle trajectories of different length.
        keep_frames : bool
                If True, the visualizer will keep all frames
                after combining them into a video.
        video_format : str
                The format of the video to be generated. Default is mp4.
        video_title : str
                The name of the video file.
        renderer_resolution : list
                List containing the resolution of the rendered videos and screenshots
        renderer_spp : int
                Samples per pixel for the rendered videos and screenshots.
        renderer : Mitsuba
                The renderer engine to use for rendering.
        do_create_video: bool
                If True, the visualizer will create a video from the frames.
        camera : znvis.cameras.Camera object
                Camera object to use for the visualization. If None, a default camera
                will be used.
        """
        # Call parent constructor
        super().__init__(
            particles=particles,
            vector_field=vector_field,
            output_folder=output_folder,
            frame_rate=frame_rate,
            number_of_steps=number_of_steps,
            keep_frames=keep_frames,
            bounding_box=bounding_box,
            video_format=video_format,
            video_title=video_title,
            renderer_resolution=renderer_resolution,
            renderer_spp=renderer_spp,
            renderer=renderer,
        )

        # Headless-specific attributes
        self.do_create_video = do_create_video
        self.app = None
        self.vis = None

        # Set the camera
        self.camera = camera
        if self.camera is not None:
            self.camera.verify_camera_setup_for_rendering()
        else:
            print(
                "No camera provided, using a default view matrix."
                "Consider using the StaticCamera class to set the "
                "view matrix based on your particle positions or box size."
            )
            self.view_matrix = np.array(
                [[1, 0, 0, -100], [0, 1, 0, -90], [0, 0, 1, -230], [0, 0, 0, 1]]
            )

    def _record_trajectory(self):
        """
        Record the trajectory.
        """
        self.update_thread_finished = True
        self.save_thread_finished = True

        def update_callable():
            """
            Function to be called on thread to update positions.
            """
            self._update_particles()
            self.update_thread_finished = True

        def save_callable():
            """
            Function to be called on thread to save image.
            """
            mesh_dict = self.get_mesh_dict()
            # Check if the camera should be used for deciding on the view matrix
            if self.camera is not None:
                self.view_matrix = self.camera.get_view_matrix(self.counter)
            self.renderer.render_mesh_objects(
                mesh_dict,
                self.view_matrix,
                save_dir=self.frame_folder,
                save_name=f"frame_{self.counter:0>6}.png",
                resolution=self.renderer_resolution,
                samples_per_pixel=self.renderer_spp,
            )
            self.save_thread_finished = True

        with Progress() as progress:
            task = progress.add_task("Saving scenes...", total=self.number_of_steps)
            while not progress.finished:
                time.sleep(1 / self.frame_rate)

                if self.save_thread_finished and self.update_thread_finished:
                    self.save_thread_finished = False
                    save_callable()
                    progress.update(task, advance=1)

                if self.update_thread_finished:
                    self.update_thread_finished = False
                    update_callable()

        time.sleep(1)  # Ensure the last image is saved
        if self.do_create_video:
            self._create_movie()

    def render_visualization(self):
        """
        Run the visualization.

        Returns
        -------
        Launches the visualization.
        """
        self.frame_folder.mkdir(parents=True, exist_ok=True)
        self._record_trajectory()
