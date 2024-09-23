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

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import pathlib
import re
import shutil
import time
import typing
import numpy as np
import cv2
import open3d as o3d
from rich.progress import Progress, track

import znvis
from znvis.rendering import Mitsuba


class Headless_Renderer:
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
        vector_field: typing.List[znvis.VectorField] = None,
        output_folder: typing.Union[str, pathlib.Path] = "./",
        frame_rate: int = 24,
        number_of_steps: int = None,
        keep_frames: bool = True,
        bounding_box: znvis.BoundingBox = None,
        video_format: str = "mp4",
        renderer_resolution: list = [4096, 2160],
        renderer_spp: int = 64,
        renderer: Mitsuba = Mitsuba(),
        view_matrix: np.ndarray = np.array([[ 1,  0,  0, -100],
                                            [ 0,  1,  0, -90],
                                            [ 0,  0,  1, -230],
                                            [ 0,  0,  0,  1]]),
        ):
        """
        Constructor for the visualizer.

        Parameters
        ----------
        particles : list[znvis.Particle]
                List of particles to add to the visualizer.
        vector_field : list[znvis.VectorField]
                List of vector fields to add to the visualizer.
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
                The format of the video to be generated.
        renderer_resolution : list
                List containing the resolution of the rendered videos and screenshots
        renderer_spp : int
                Samples per pixel for the rendered videos and screenshots.
        view_matrix : np.array
                The view matrix for the camera. Default is a view matrix exported from a 200x200x1 system.
        """
        self.particles = particles
        self.vector_field = vector_field
        self.frame_rate = frame_rate
        self.bounding_box = bounding_box() if bounding_box else None
        self.view_matrix = view_matrix

        if number_of_steps is None:
            len_list = []
            for particle in particles:
                if not particle.static:
                    len_list.append(len(particle.position))
            
            if len_list == []:
                self.number_of_steps = 1
            else:
                self.number_of_steps = min(len_list)
    
        self.output_folder = pathlib.Path(output_folder).resolve()
        self.frame_folder = self.output_folder / "video_frames"
        self.video_format = video_format
        self.renderer_resolution = renderer_resolution
        self.renderer_spp = renderer_spp
        self.keep_frames = keep_frames
        self.renderer = renderer
        self.app = None
        self.vis = None
        self.counter = 0

    def _create_movie(self):
        """
        Concatenate images into a movie.

        This needs to be a seperate method so that the
        image storing thread can run to completion before
        this one is called. (GIL stuff)
        """
        images = [f.as_posix() for f in self.frame_folder.glob("*.png")]

        # Sort images by number
        images = sorted(images, key=lambda s: int(re.search(r"\d+", s).group()))

        single_frame = cv2.imread(images[0])
        height, width, layers = single_frame.shape

        video = cv2.VideoWriter(
            (self.output_folder / f"ZnVis-Video.{self.video_format}").as_posix(),
            0,
            self.frame_rate,
            (width, height),
        )
        for image in track(images, description="Exporting Video..."):
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()

        # Delete temporary directory if not storing run files
        if not self.keep_frames:
            shutil.rmtree(self.frame_folder, ignore_errors=False)

    def _initialize_particles(self):
        """
        Initialize the particles in the simulation.

        This method will construct the particle dictionaries in each Particle class.
        """
        # Build the mesh dict for each particle
        for item in self.particles:
            item.construct_mesh_list()

    def _initialize_vector_field(self):
        for item in self.vector_field:
            item.construct_mesh_list()

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
            mesh_dict = {}

            if self.vector_field is not None:
                for item in self.vector_field:
                    if item.static:
                        mesh_dict[item.name] = {
                            "mesh": item.mesh_list[0],
                            "bsdf": item.mesh.material.mitsuba_bsdf,
                            "material": item.mesh.o3d_material,
                        }
                    else:
                        mesh_dict[item.name] = {
                            "mesh": item.mesh_list[self.counter],
                            "bsdf": item.mesh.material.mitsuba_bsdf,
                            "material": item.mesh.o3d_material,
                        }

            for item in self.particles:
                if item.static:
                    mesh_dict[item.name] = {
                        "mesh": item.mesh_list[0],
                        "bsdf": item.mesh.material.mitsuba_bsdf,
                        "material": item.mesh.o3d_material,
                    }
                else:
                    mesh_dict[item.name] = {
                        "mesh": item.mesh_list[self.counter],
                        "bsdf": item.mesh.material.mitsuba_bsdf,
                        "material": item.mesh.o3d_material,
                    }
            self.output_folder.mkdir(parents=True, exist_ok=True)   
            self.frame_folder.mkdir(parents=True, exist_ok=True)

            self.renderer.render_mesh_objects(
                mesh_dict,
                self.view_matrix,
                save_dir=self.frame_folder,
                save_name=f"frame_{self.counter:0>6}.png", 
                resolution=self.renderer_resolution,
                samples_per_pixel=self.renderer_spp
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
                    update_callable()

        time.sleep(1)  # Ensure the last image is saved
        self._create_movie()

    def _update_particles(self, visualizer=None, step: int = None):
        """
        Update the positions of the particles.

        Parameters
        ----------
        step : int
                Step to update to.

        Returns
        -------
        Updates the positions of the particles in the box.
        """
        if visualizer is None:
            visualizer = self.vis
        if step is None:
            if self.counter == self.number_of_steps - 1:
                self.counter = 0
            else:
                self.counter += 1
            step = self.counter

    def render_visualization(self):
        """
        Run the visualization.

        Returns
        -------
        Launches the visualization.
        """
        self._initialize_particles()
        if self.vector_field is not None:
            self._initialize_vector_field()
        self._record_trajectory()
