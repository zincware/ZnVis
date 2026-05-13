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
Base visualizer class with shared functionality.
"""

import pathlib
import typing

import znvis
from znvis.rendering import Mitsuba
from znvis.video import VideoManager


class BaseVisualizer:
    """
    Base class for visualizers containing shared functionality.

    This class provides common methods and initialization logic used by both
    the interactive Visualizer and the Headless_Visualizer.

    Attributes
    ----------
    particles : list[znvis.Particle]
            A list of particle objects to add to the visualizer.
    vector_field : list[znvis.VectorField]
            A list of vector field objects to add to the visualizer.
    frame_rate : int
            Frame rate for the visualizer measured in frames per second (fps).
    number_of_steps : int
            Number of steps in the visualization.
    output_folder : pathlib.Path
            Path to the output folder where the video and frames will be saved.
    frame_folder : pathlib.Path
            Path to the folder where individual frames will be saved.
    video_format : str
            The format of the video to be generated.
    video_title : str
            The name of the video file.
    renderer : Mitsuba
            The renderer engine to use for rendering.
    video_manager : VideoManager
            Manager for video creation and format handling.
    counter : int
            Internally stored counter to track which configuration is currently
            being viewed.
    """

    def __init__(
        self,
        particles: typing.List[znvis.Particle],
        vector_field: typing.List[znvis.VectorField] | None = None,
        output_folder: typing.Union[str, pathlib.Path] = "./",
        frame_rate: int = 60,
        number_of_steps: int | None = None,
        keep_frames: bool = True,
        bounding_box: znvis.BoundingBox | None = None,
        video_format: str = "mp4",
        video_title: str = "ZnVis-Video",
        renderer_resolution: typing.Sequence[int] | None = None,
        renderer_spp: int = 64,
        renderer: Mitsuba | None = None,
    ):
        """
        Initialize the base visualizer.

        Parameters
        ----------
        particles : list[znvis.Particle]
                List of particles to add to the visualizer.
        vector_field : list[znvis.VectorField], optional
                List of vector fields to add to the visualizer.
        output_folder : typing.Union[str, pathlib.Path], optional
                Path to the output folder where the video and frames will be saved.
        frame_rate : int, optional
                Frame rate for the visualizer measured in frames per second (fps).
        number_of_steps : int, optional
                Number of steps in the visualization. If None, the zeroth order of one
                particle is taken.
        keep_frames : bool, optional
                If True, the visualizer will keep all frames
                after combining them into a video.
        bounding_box : znvis.BoundingBox, optional
                Bounding box for the visualization.
        video_format : str, optional
                The format of the video to be generated. Default is mp4.
        video_title : str, optional
                The name of the video file.
        renderer_resolution : typing.Sequence[int], optional
                Sequence containing the resolution of the rendered videos and images.
        renderer_spp : int, optional
                Samples per pixel for the rendered videos and screenshots.
        renderer : Mitsuba, optional
                The renderer engine to use for rendering.
        """
        self.particles = particles
        self.vector_field = vector_field
        self.frame_rate = frame_rate
        self.bounding_box = bounding_box() if bounding_box else None

        # Determine number of steps
        if number_of_steps is None:
            len_list = [len(p.position) for p in particles if not p.static]
            if vector_field is not None:
                len_list += [len(v.position) for v in vector_field if not v.static]
            self.number_of_steps = min(len_list) if len_list else 1
        else:
            self.number_of_steps = number_of_steps

        # Setup paths
        self.output_folder = pathlib.Path(output_folder).resolve()
        self.frame_folder = self.output_folder / "video_frames"

        # Video settings
        self.video_title = video_title
        self.keep_frames = keep_frames

        # Renderer settings
        self.renderer_resolution = (
            list(renderer_resolution) if renderer_resolution else [4096, 2160]
        )
        self.renderer_spp = renderer_spp
        self.renderer = renderer or Mitsuba()

        # Initialize video manager
        self.video_manager = VideoManager(
            output_folder=self.output_folder, frame_rate=self.frame_rate
        )
        # Validate video format
        self.video_format = self.video_manager.validate_video_format(video_format)

        # Counter for tracking current frame
        self.counter = 0

    def _create_movie(self):
        """
        Concatenate images into a movie using VideoManager.

        This needs to be a separate method so that the
        image storing thread can run to completion before
        this one is called. (GIL stuff)
        """
        try:
            self.video_manager.create_video_from_frames(
                frame_folder=self.frame_folder,
                video_name=self.video_title,
                video_format=self.video_format,
                keep_frames=self.keep_frames,
                frame_pattern="*.png",
            )
        except RuntimeError as e:
            print(f"Error creating video: {e}")
            raise

    def _initialize_particles(self):
        """
        Initialize the particles in the simulation.

        This method will construct the particle dictionaries in each Particle class.
        """
        # Build the mesh dict for each particle
        for item in self.particles:
            item.construct_mesh_list()

    def _initialize_vector_field(self):
        """
        Initialize the vector field in the simulation.

        This method will construct the mesh dictionaries in each VectorField class.
        """
        if self.vector_field is not None:
            for item in self.vector_field:
                item.construct_mesh_list()

    def _update_particles(self, visualizer=None, step: int = None):
        """
        Update the positions of the particles.

        Parameters
        ----------
        visualizer : object, optional
                Visualizer instance (if applicable).
        step : int, optional
                Step to update to. If None, advances the counter.

        Returns
        -------
        Updates the positions of the particles in the box.
        """
        if step is None:
            if self.counter == self.number_of_steps - 1:
                self.counter = 0
            else:
                self.counter += 1
            step = self.counter

    def get_mesh_dict(self, counter=None):
        """
        Creates the mesh dict for a given scene.
        """
        if counter is None:
            counter = self.counter

        mesh_dict = {}
        if self.vector_field is not None:
            for item in self.vector_field:
                idx = 0 if item.static else counter
                mesh = (
                    item.mesh_list[idx]
                    if item.mesh_list is not None
                    else item.get_mesh_for_frame(counter)
                )
                mesh_dict[item.name] = {
                    "mesh": mesh,
                    "bsdf": item.mesh.material.mitsuba_bsdf,
                    "material": item.mesh.o3d_material,
                }
        for item in self.particles:
            idx = 0 if item.static else counter
            mesh = (
                item.mesh_list[idx]
                if item.mesh_list is not None
                else item.get_mesh_for_frame(counter)
            )
            mesh_dict[item.name] = {
                "mesh": mesh,
                "bsdf": item.mesh.material.mitsuba_bsdf,
                "material": item.mesh.o3d_material,
            }

        return mesh_dict
