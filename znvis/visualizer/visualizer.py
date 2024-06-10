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
import threading
import time
import typing

import cv2
import open3d as o3d
import open3d.visualization.gui as gui
from rich.progress import Progress, track

import znvis
from znvis.rendering import Mitsuba


class Visualizer:
    """
    Main class to perform visualization.

    Attributes
    ----------
    particles : list[znvis.Particle]
            A list of particle objects to add to the visualizer.
    app : o3d.gui.Application.instance
            An open3d application.
    vis : o3d.visualization.O3DVisualizer
            An open3d visualizer window
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
        renderer: Mitsuba = Mitsuba(),
    ):
        """
        Constructor for the visualizer.

        Parameters
        ----------
        particles : list[znvis.Particle]
                List of particles to add to the visualizer.
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
        """
        self.particles = particles
        self.vector_field = vector_field
        self.frame_rate = frame_rate
        self.bounding_box = bounding_box() if bounding_box else None

        if number_of_steps is None:
            len_list = []
            for particle in particles:
                if not particle.static:
                    len_list.append(len(particle.position))
            number_of_steps = min(len_list)
        self.number_of_steps = number_of_steps

        self.output_folder = pathlib.Path(output_folder).resolve()
        self.frame_folder = self.output_folder / "video_frames"
        self.video_format = video_format
        self.keep_frames = keep_frames
        self.renderer = renderer
        self.play_speed = 1

        self.obj_folder = self.output_folder / "obj_files"

        # Added later during run
        self.app = None
        self.vis = None

        self.counter = 0

    def _initialize_app(self):
        """
        Initialize the app object.

        Returns
        -------
        Updates the app and vis class attributes.
        """
        self.app = gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer("ZnVis Visualizer", 1920, 1080)
        self.vis.show_settings = True
        self.vis.reset_camera_to_default()

        # Add actions to the visualizer.
        self.vis.add_action("<<", self._update_particles_back)
        self.vis.add_action("Play", self._continuous_trajectory)
        self.vis.add_action(">>", self._update_particles)
        self.vis.add_action(">>>", self._toggle_play_speed)
        self.vis.add_action("Slow", self._toggle_slowmotion)
        self.vis.add_action("Restart", self._restart_trajectory)
        self.vis.add_action("Export Scene", self._export_scene)
        self.vis.add_action("Screenshot", self._take_screenshot)
        self.vis.add_action("Export Video", self._export_video)
        self.vis.add_action("Export Mesh Trajectory", self._export_mesh_trajectory)


        # Add the visualizer to the app.
        self.app.add_window(self.vis)

        self.interrupt: int = 0  # 0 = Not running, 1 = running

    def _pause_run(self, vis):
        """
        Pause a live visualization run.

        Returns
        -------
        Set self.interrupt = 1
        """
        self.interrupt = 0

    def _export_video(self, vis):
        """
        Export a video of the simulation.

        Parameters
        ----------
        vis : Visualizer
                The active visualizer.

        Returns
        -------
        Saves a video locally.
        """
        self.interrupt = 0  # stop live feed if running.

        # Create temporary directory
        self.frame_folder.mkdir(parents=True, exist_ok=True)

        # Write all PNG files to directory
        t = threading.Thread(target=self._record_trajectory)
        t.start()

    def _export_mesh_trajectory(self, vis):
        """
        Export a video of the simulation.

        Parameters
        ----------
        vis : Visualizer
                The active visualizer.

        Returns
        -------
        Saves a video locally.
        """
        self.interrupt = 0  # stop live feed if running.

        # Create temporary directory
        self.obj_folder.mkdir(parents=True, exist_ok=True)

        # Write all PNG files to directory
        t = threading.Thread(target=self._record_mesh_trajectory)
        t.start()

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

    def _export_scene(self, vis):
        """
        Export the current visualization scene.

        Parametersor texture in ("albedo", "normal", "ao", "metallic", "roughness"):
        ----------
        vis : Visualizer
                The active visualizer.

        Returns
        -------
        Stores a .ply model locally.
        """
        old_state = self.interrupt  # get old state
        self.interrupt = 0  # stop live feed if running.
        for i, item in enumerate(self.particles):
            if i == 0:
                mesh = item.mesh_list[self.counter]
            else:
                mesh += item.mesh_list[self.counter]

        o3d.io.write_triangle_mesh(
            (self.output_folder / f"My_mesh_{self.counter}.obj").as_posix(), mesh
        )

        # Restart live feed if it was running before the export.
        if old_state == 1:
            self._continuous_trajectory(vis)

    def _take_screenshot(self, vis):
        """
        Take a screenshot

        Parameters
        ----------
        vis : Visualizer
                The activate visualizer.

        Returns
        -------
        Takes a screenshot and dumps it
        """
        old_state = self.interrupt  # get old state
        self.interrupt = 0  # stop live feed if running.
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

        view_matrix = vis.scene.camera.get_view_matrix()

        self.renderer.render_mesh_objects(
            mesh_dict, view_matrix, save_name=f"frame_{self.counter}.png"
        )

        # Restart live feed if it was running before the export.
        if old_state == 1:
            self._continuous_trajectory(vis)

    def _initialize_particles(self):
        """
        Initialize the particles in the simulation.

        This method will construct the particle dictionaries in each Particle class and
        then add the first location of each particle to the visualizer window.
        """
        # Build the mesh dict for each particle and add them to the window.
        for item in self.particles:
            item.construct_mesh_list()

        self._draw_particles(initial=True)

    def _initialize_vector_field(self):
        for item in self.vector_field:
            item.construct_mesh_list()
        self._draw_vector_field(initial=True)

    def _draw_particles(self, visualizer=None, initial: bool = False):
        """
        Draw the particles on the visualizer.

        Parameters
        ----------
        initial : bool (default = True)
                If true, no particles are removed.

        Returns
        -------
        updates the information in the visualizer.

        Notes
        -----
        TODO: Use of initial is a dirty fix. It can be removed when support for
              transforming multiple geometry objects is added to open3d.
        """
        # Check if a visualizer was passed.
        if visualizer is None:
            visualizer = self.vis

        # Add the particles to the visualizer.
        if initial:
            for i, item in enumerate(self.particles):
                visualizer.add_geometry(
                    item.name, item.mesh_list[self.counter], item.mesh.o3d_material
                )

            # check for bounding box
            if self.bounding_box is not None:
                visualizer.add_geometry("Box", self.bounding_box)
        else:
            for i, item in enumerate(self.particles):
                if not item.static:
                    visualizer.remove_geometry(item.name)
                    visualizer.add_geometry(
                        item.name, item.mesh_list[self.counter], item.mesh.o3d_material
                    )


    def _draw_vector_field(self, visualizer=None, initial: bool = False):
        """
        Draw the vector field on the visualizer.

        Parameters
        ----------
        initial : bool (default = True)
                If true, no particles are removed.

        Returns
        -------
        updates the information in the visualizer.
        -----
        """
        if visualizer is None:
            visualizer = self.vis
        
        if initial:
            for i, item in enumerate(self.vector_field):
                visualizer.add_geometry(
                    item.name, item.mesh_list[self.counter], item.mesh.o3d_material
                )
        else:
            for i, item in enumerate(self.vector_field):
                if not item.static:
                    visualizer.remove_geometry(item.name)
                    visualizer.add_geometry(
                        item.name, item.mesh_list[self.counter], item.mesh.o3d_material
                    )

    def _continuous_trajectory(self, vis):
        """
        Button command for running the simulation in the visualizer.

        Parameters
        ----------
        vis : visualizer
                Object passed during the callback.
        """
        if self.interrupt == 1:
            self._pause_run(vis)
        else:
            threading.Thread(target=self._run_trajectory).start()

    def _continuous_trajectory_backwards(self, vis):
        """
        Button command for running the simulation in the visualizer backwards.

        Parameters
        ----------
        vis : visualizer
                Object passed during the callback.
        """
        if self.interrupt == 1:
            self._pause_run(vis)
        else:
            threading.Thread(target=self._run_trajectory_backwards).start()

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

            view_matrix = self.vis.scene.camera.get_view_matrix()
            self.renderer.render_mesh_objects(
                mesh_dict,
                view_matrix,
                save_dir=self.frame_folder,
                save_name=f"frame_{self.counter:0>6}.png",
            )

            self.save_thread_finished = True

        with Progress() as progress:
            task = progress.add_task("Saving scenes...", total=self.number_of_steps)
            while not progress.finished:
                time.sleep(1 / self.frame_rate)

                if self.save_thread_finished and self.update_thread_finished:
                    self.save_thread_finished = False
                    o3d.visualization.gui.Application.instance.post_to_main_thread(
                        self.vis, save_callable
                    )
                    progress.update(task, advance=1)

                if self.update_thread_finished:
                    self.update_thread_finished = False
                    o3d.visualization.gui.Application.instance.post_to_main_thread(
                        self.vis, update_callable
                    )

        time.sleep(1)  # Ensure the last image is saved
        self._create_movie()

    def _record_mesh_trajectory(self):
        """
        Export the trajectory as mesh files.
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
            for i, item in enumerate(self.particles):
                if item.static:
                    if i == 0:
                        mesh = item.mesh_list[0]
                    else:
                        mesh += item.mesh_list[0]
                else:
                    if i == 0:
                        mesh = item.mesh_list[self.counter]
                    else:
                        mesh += item.mesh_list[self.counter]

            o3d.io.write_triangle_mesh(
                (self.obj_folder / f"export_mesh_{self.counter}.ply").as_posix(), mesh
            )
            self.save_thread_finished = True

        with Progress() as progress:
            task = progress.add_task("Saving scenes...", total=self.number_of_steps)
            # while self.counter < (self.number_of_steps - 1):
            while not progress.finished:
                time.sleep(1 / self.frame_rate)

                if self.save_thread_finished and self.update_thread_finished:
                    self.save_thread_finished = False
                    o3d.visualization.gui.Application.instance.post_to_main_thread(
                        self.vis, save_callable
                    )
                    progress.update(task, advance=1)

                if self.update_thread_finished:
                    self.update_thread_finished = False
                    o3d.visualization.gui.Application.instance.post_to_main_thread(
                        self.vis, update_callable
                    )

        time.sleep(1)  # Ensure the last image is saved

    def _run_trajectory(self):
        """
        Callback method for running the trajectory smoothly.

        Returns
        -------
        Runs through the trajectory.
        """
        self.interrupt = 1  # set global run state.
        while self.counter < self.number_of_steps:
            time.sleep(1 / (self.frame_rate * self.play_speed))
            o3d.visualization.gui.Application.instance.post_to_main_thread(
                self.vis, self._update_particles
            )
            # Break if interrupted.
            if self.interrupt == 0:
                break

        self.interrupt = 0  # reset global state.

    
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

        self._draw_particles(visualizer=visualizer)  # draw the particles.

        # draw the vector field if it exists.
        if self.vector_field is not None:
            self._draw_vector_field(visualizer=visualizer) 

        visualizer.post_redraw()  # re-draw the window.

    def _update_particles_back(self, visualizer=None, step: int = None):
        """
        Update the positions of the particles one step back (rewind-feature)

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
            if self.counter == 0:
                self.counter = self.number_of_steps - 1
            else:
                self.counter -= 1
            step = self.counter
        self._draw_particles(visualizer=visualizer)  # draw the particles.

        # draw the vector field if it exists.
        if self.vector_field is not None:
            self._draw_vector_field(visualizer=visualizer) 

        visualizer.post_redraw()  # re-draw the window.


    def _restart_trajectory(self, visualizer=None):
        if visualizer is None:
            visualizer = self.vis
        self.counter = 0

        self._draw_particles(visualizer=visualizer)  # draw the particles.

        # draw the vector field if it exists.
        if self.vector_field is not None:
            self._draw_vector_field(visualizer=visualizer) 

        visualizer.post_redraw()  # re-draw the window.

    def _toggle_play_speed(self, visualizer=None):
        """ 
        Toggle the play speed from 1 to 2 to 4 to 8 and back to 1.
        
        """
        if self.play_speed == 1:
            self.play_speed = 2
        elif self.play_speed == 2:
            self.play_speed = 4
        elif self.play_speed == 4:
            self.play_speed = 8
        else:
            self.play_speed = 1
            
    def _toggle_play_speed_back(self, visualizer=None):
        """ 
        Toggle the play speed from 1 to 2 to 4 to 8 and back to 1.
        """
 
        self._run_trajectory_backwards()
        if self.play_speed == 1:
            self.play_speed = 2
        elif self.play_speed == 2:
            self.play_speed = 4
        elif self.play_speed == 4:
            self.play_speed = 8
        else:
            self.play_speed = 1
        
    def _toggle_slowmotion(self, visualizer=None):
        """ 
        Toggle the play speed from 1 to 1/2 to 1/4 to 1/8 and back to 1
        """
        if self.play_speed >= 1:
            self.play_speed = 1/2
        elif self.play_speed == 1/2:
            self.play_speed = 1/4
        elif self.play_speed == 1/4:
            self.play_speed = 1/8
        else:
            self.play_speed = 1

        
    def run_visualization(self):
        """
        Run the visualization.

        Returns
        -------
        Launches the visualization.
        """
        self._initialize_app()
        self._initialize_particles()
        if self.vector_field is not None:
            self._initialize_vector_field()

        self.vis.reset_camera_to_default()
        self.app.run()
