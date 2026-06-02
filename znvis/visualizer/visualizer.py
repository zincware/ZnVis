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
import threading
import time
import typing

import open3d as o3d
import open3d.visualization.gui as gui
from rich.progress import Progress

import znvis
from znvis.cameras import KeyframeCamera
from znvis.rendering import Mitsuba
from znvis.visualizer.base_visualizer import BaseVisualizer
from znvis.visualizer.cache.mesh_cache_manager import MeshCacheManager
from znvis.visualizer.cache.mesh_frame_cache import MeshFrameCache


class Visualizer(BaseVisualizer):
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
        keyframe_camera: KeyframeCamera = None,
        lazy_mesh_loading: bool = False,
        mesh_cache_max_gb: float | None = None,
        mesh_cache_future_fraction: float = 2 / 3,
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
        keyframe_camera : KeyframeCamera
                The camera to use for interpolation
        lazy_mesh_loading : bool
                Default: False
                Decide whether eagerly preloading all meshes into memory before a
                visualization starts or lazily load only a subset of frames into cache.
        mesh_cache_max_gb : float | None
                Default: None
                Maximum estimated mesh memory in GiB to keep in the cache. Static meshes
                stay cached and dynamic meshes are evicted to keep the total below
                this limit when possible.
        mesh_cache_future_fraction : float
                Default: 2 / 3
                Fraction of the rolling cache window to place in the current playback
                direction. The rest is placed behind the current frame.

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

        # Visualizer-specific attributes
        self.play_speed = 1
        self.do_rewind = False
        self.obj_folder = self.output_folder / "obj_files"
        self.scene_folder = self.output_folder / "scenes"
        self.screenshot_folder = self.output_folder / "screenshots"
        self.app = None
        self.vis = None
        self.interrupt = 0
        self._headless_export_hint_shown = False
        self.mesh_cache = None
        self.mesh_cache_manager = None
        self.lazy_mesh_loading = lazy_mesh_loading
        self.mesh_cache_max_gb = mesh_cache_max_gb
        self.mesh_cache_future_fraction = min(
            1.0,
            max(0.0, float(mesh_cache_future_fraction)),
        )
        self._trajectory_update_pending = False

        # Camera Handling
        if keyframe_camera is not None:
            if not isinstance(keyframe_camera, KeyframeCamera):
                raise TypeError("You can only use a KeyframeCamera in the Visualizer")
            self.keyframe_camera = keyframe_camera

            if self.keyframe_camera.view_matrices_path is None:
                print(
                    "Setting the visualizer output folder as "
                    "output folder of the KeyframeCamera.\n"
                )
                self.keyframe_camera.view_matrices_path = self.output_folder

            self.keyframe_camera.number_of_frames = self.number_of_steps
            self.activate_view_matrix_interface = True
        else:
            self.activate_view_matrix_interface = False

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
        self.vis.add_action("<<<", self._toggle_play_speed_back)
        self.vis.add_action("<<", self._update_particles_back)
        self.vis.add_action("Play", self._continuous_trajectory)
        self.vis.add_action(">>", self._update_particles)
        self.vis.add_action(">>>", self._toggle_play_speed)
        self.vis.add_action("Toggle Direction", self._toogle_play_direction)
        self.vis.add_action("Slow", self._toggle_slowmotion)
        self.vis.add_action("Restart", self._restart_trajectory)
        self.vis.add_action("Export Scene", self._export_scene)
        self.vis.add_action("Screenshot", self._take_screenshot)
        self.vis.add_action("Export Video", self._export_video)
        self.vis.add_action("Export Mesh Trajectory", self._export_mesh_trajectory)
        self.vis.add_action("Print current frame", self._output_current_counter)

        if self.activate_view_matrix_interface:
            # Add actions to the visualizer for the keyframe camera.
            # The lambdas are necessary, as vis.add_action does not allow
            # passing arguments to the function.
            self.vis.add_action(
                "Add current View Matrix",
                lambda _: self.keyframe_camera.add_view_matrix(
                    self.counter, self.vis.scene.camera.get_view_matrix()
                ),
            )
            self.vis.add_action(
                "Remove current View Matrix",
                lambda _: self.keyframe_camera.remove_view_matrix(self.counter),
            )
            self.vis.add_action(
                "Interpolate and export view matrices",
                lambda _: self.keyframe_camera.interpolate_and_export_view_matrices(),
            )
            self.vis.add_action(
                "Reset View Matrix Progress",
                lambda _: self.keyframe_camera.reset_view_matrix_progress(),
            )

        # Add the visualizer to the app.
        self.app.add_window(self.vis)

        self.interrupt = 0  # 0 = Not running, 1 = running

    def _pause_run(self, vis):
        """
        Pause a live visualization run.

        Returns
        -------
        Set self.interrupt = 0
        """
        self.interrupt = 0
        if self.lazy_mesh_loading:
            self.mesh_cache_manager.submit_pause_refill(
                self.counter,
                self.do_rewind,
                self._get_playback_frame_step(),
            )

    def _print_headless_export_hint_once(self):
        """
        Print the GUI export hint once per visualizer instance.
        """
        if not self._headless_export_hint_shown:
            print(
                "Info: GUI video export renders through the "
                "interactive visualizer and may block the window. "
                "For long or high-resolution renders, consider using "
                "HeadlessVisualizer, which can render without the GUI "
                "and supports parallel rendering."
            )
            self._headless_export_hint_shown = True

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
        self._print_headless_export_hint_once()
        self.do_rewind = False
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

    def _export_scene(self, vis):
        """
        Export the current visualization scene.

        Parameters or texture in ("albedo", "normal", "ao", "metallic", "roughness"):
        ----------
        vis : Visualizer
                The active visualizer.

        Returns
        -------
        Stores a .ply model locally.
        """
        old_state = self.interrupt  # get old state
        self.interrupt = 0  # stop live feed if running.
        created_mesh = False
        for i, item in enumerate(self.particles):
            if item.static:
                if i == 0 and not created_mesh:
                    mesh = self._get_mesh_for_item(item, 0)
                    created_mesh = True
                elif i == 0 and created_mesh:
                    mesh += self._get_mesh_for_item(item, 0)
                else:
                    continue

            if i == 0 and not created_mesh:
                mesh = self._get_mesh_for_item(item, self.counter)
                created_mesh = True
            else:
                mesh += self._get_mesh_for_item(item, self.counter)

        self.scene_folder.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(
            (self.scene_folder / f"My_mesh_{self.counter}.obj").as_posix(), mesh
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
                        "mesh": self._get_mesh_for_item(item, 0),
                        "bsdf": item.mesh.material.mitsuba_bsdf,
                        "material": item.mesh.o3d_material,
                    }
                else:
                    mesh_dict[item.name] = {
                        "mesh": self._get_mesh_for_item(item, self.counter),
                        "bsdf": item.mesh.material.mitsuba_bsdf,
                        "material": item.mesh.o3d_material,
                    }

        for item in self.particles:
            if item.static:
                mesh_dict[item.name] = {
                    "mesh": self._get_mesh_for_item(item, 0),
                    "bsdf": item.mesh.material.mitsuba_bsdf,
                    "material": item.mesh.o3d_material,
                }
            else:
                mesh_dict[item.name] = {
                    "mesh": self._get_mesh_for_item(item, self.counter),
                    "bsdf": item.mesh.material.mitsuba_bsdf,
                    "material": item.mesh.o3d_material,
                }

        view_matrix = vis.scene.camera.get_view_matrix()
        # Create output folder
        self.screenshot_folder.mkdir(parents=True, exist_ok=True)
        self.renderer.render_mesh_objects(
            mesh_dict,
            view_matrix,
            save_dir=self.screenshot_folder,
            save_name=f"frame_{self.counter}.png",
            resolution=self.renderer_resolution,
            samples_per_pixel=self.renderer_spp,
        )

        # Restart live feed if it was running before the export.
        if old_state == 1:
            self._continuous_trajectory(vis)

    def _get_mesh_for_item(self, item, frame_index):
        """
        Provides the mesh list for an item at a given frame.
        """
        idx = 0 if item.static else frame_index

        if self.lazy_mesh_loading:
            if self.mesh_cache_manager is None:
                raise RuntimeError("Mesh cache is not initialized yet.")
            return self.mesh_cache_manager.get(item, idx)

        return item.mesh_list[idx]

    def _get_mesh_cache_max_bytes(self):
        if self.mesh_cache_max_gb is None:
            return None
        return int(self.mesh_cache_max_gb * 1024**3)

    def _initialize_mesh_cache(self):
        if self.mesh_cache is None:
            self.mesh_cache = MeshFrameCache(
                max_bytes=self._get_mesh_cache_max_bytes(),
            )
        if self.mesh_cache_manager is None:
            self.mesh_cache_manager = MeshCacheManager(
                items=self.particles + (self.vector_field or []),
                number_of_steps=self.number_of_steps,
                cache=self.mesh_cache,
                future_fraction=self.mesh_cache_future_fraction,
            )

        self.mesh_cache_manager.initialize(
            current_frame=self.counter,
            do_rewind=self.do_rewind,
            frame_step=self._get_playback_frame_step(),
            start_worker=self.app is not None,
        )

    def _initialize_particles(self):
        """
        Initialize the particles in the simulation.

        This method will construct the particle dictionaries in each Particle class and
        then add the first location of each particle to the visualizer window.
        """

        if self.lazy_mesh_loading:
            self._initialize_mesh_cache()
        else:
            for item in self.particles:
                item.construct_mesh_list()

        self._draw_particles(initial=True)

    def _initialize_vector_field(self):
        if self.lazy_mesh_loading:
            self._initialize_mesh_cache()
        else:
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

        if initial and self.bounding_box is not None:
            visualizer.add_geometry("Box", self.bounding_box)

        for item in self.particles:
            if not initial:
                if not item.static:
                    visualizer.remove_geometry(item.name)
                else:
                    continue

            mesh = self._get_mesh_for_item(item, self.counter)
            visualizer.add_geometry(item.name, mesh, item.mesh.o3d_material)

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
                    item.name,
                    self._get_mesh_for_item(item, self.counter),
                    item.mesh.o3d_material,
                )
        else:
            for i, item in enumerate(self.vector_field):
                if not item.static:
                    visualizer.remove_geometry(item.name)
                    visualizer.add_geometry(
                        item.name,
                        self._get_mesh_for_item(item, self.counter),
                        item.mesh.o3d_material,
                    )

    def _continuous_trajectory(self, vis):
        """
        Button command for running the simulation in the visualizer.

        Parameters
        ----------
        vis : visualizer
                Object passed during the callback.
        """
        self.do_rewind = False
        if self.interrupt == 1:
            self._pause_run(vis)
        else:
            if self.lazy_mesh_loading:
                self.mesh_cache_manager.cancel_pause_refill()
            threading.Thread(target=self._run_trajectory).start()

    def get_mesh_dict(self, counter: int | None):
        """
        Creates the mesh dict for a given scene.
        """
        if counter is None:
            counter = self.counter

        if not self.lazy_mesh_loading:
            return super().get_mesh_dict(counter)

        mesh_dict = {}
        items = (self.vector_field or []) + self.particles

        for item in items:
            mesh_dict[item.name] = {
                "mesh": self._get_mesh_for_item(item, counter),
                "bsdf": item.mesh.material.mitsuba_bsdf,
                "material": item.mesh.o3d_material,
            }
        return mesh_dict

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

            view_matrix = self.vis.scene.camera.get_view_matrix()
            self.renderer.render_mesh_objects(
                mesh_dict,
                view_matrix,
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
                        mesh = self._get_mesh_for_item(item, 0)
                    else:
                        mesh += self._get_mesh_for_item(item, 0)
                else:
                    if i == 0:
                        mesh = self._get_mesh_for_item(item, self.counter)
                    else:
                        mesh += self._get_mesh_for_item(item, self.counter)

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
            if not self._trajectory_update_pending:
                self._trajectory_update_pending = True

                def update_callable():
                    try:
                        self._update_particles(
                            block_on_cache_miss=False,
                            frame_step=self._get_playback_frame_step(),
                        )
                    finally:
                        self._trajectory_update_pending = False

                o3d.visualization.gui.Application.instance.post_to_main_thread(
                    self.vis, update_callable
                )
            # Break if interrupted.
            if self.interrupt == 0:
                break

        self.interrupt = 0  # reset global state.

    def _get_playback_frame_step(self) -> int:
        """
        Return how many frames autoplay should advance per drawn update.
        """
        return max(1, int(self.play_speed))

    def _update_particles(
        self,
        visualizer=None,
        step: int = None,
        block_on_cache_miss: bool = True,
        frame_step: int = 1,
    ):
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
        old_counter = self.counter
        if step is None:
            frame_step = max(1, int(frame_step))
            delta = -frame_step if self.do_rewind else frame_step
            self.counter = (self.counter + delta) % self.number_of_steps
            step = self.counter

        if self.lazy_mesh_loading:
            if (
                not block_on_cache_miss
                and not self.mesh_cache_manager.has_meshes_for_frame(self.counter)
            ):
                prefetch_frame = self.counter
                self.counter = old_counter
                self.mesh_cache_manager.submit_prefetch(
                    self.counter,
                    self.do_rewind,
                    self._get_playback_frame_step(),
                    urgent_frame=prefetch_frame,
                )
                return False
            self.mesh_cache_manager.ensure_current_frame(self.counter)

        self._draw_particles(visualizer=visualizer)  # draw the particles.

        # draw the vector field if it exists.
        if self.vector_field is not None:
            self._draw_vector_field(visualizer=visualizer)

        if self.lazy_mesh_loading:
            self.mesh_cache_manager.submit_prefetch(
                self.counter,
                self.do_rewind,
                self._get_playback_frame_step(),
            )

        visualizer.post_redraw()  # re-draw the window.
        return True

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

        if self.lazy_mesh_loading:
            self.mesh_cache_manager.ensure_current_frame(self.counter)

        self._draw_particles(visualizer=visualizer)  # draw the particles.

        # draw the vector field if it exists.
        if self.vector_field is not None:
            self._draw_vector_field(visualizer=visualizer)

        if self.lazy_mesh_loading:
            self.mesh_cache_manager.submit_prefetch(
                self.counter,
                self.do_rewind,
                self._get_playback_frame_step(),
            )

        visualizer.post_redraw()  # re-draw the window.

    def _toogle_play_direction(self, visualizer=None):
        """
        Reverts the direction of play.

        Returns
        -------
        Rewinds the trajectory.
        """
        self.do_rewind = not self.do_rewind

    def _restart_trajectory(self, visualizer=None):
        if visualizer is None:
            visualizer = self.vis
        self.counter = 0

        if self.lazy_mesh_loading:
            self.mesh_cache_manager.ensure_current_frame(self.counter)

        self._draw_particles(visualizer=visualizer)  # draw the particles.

        # draw the vector field if it exists.
        if self.vector_field is not None:
            self._draw_vector_field(visualizer=visualizer)

        if self.lazy_mesh_loading:
            self.mesh_cache_manager.submit_prefetch(
                self.counter,
                self.do_rewind,
                self._get_playback_frame_step(),
            )

        visualizer.post_redraw()  # re-draw the window.

    def _toggle_play_speed(self, visualizer=None):
        """
        Toggle the play speed from 1 to 2 to 4 to 8 and back to 1.

        """
        if self.do_rewind is True:
            self.play_speed = 1

        self.do_rewind = False

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
        if self.do_rewind is False:
            self.play_speed = 1

        self.do_rewind = True

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
            self.play_speed = 1 / 2
        elif self.play_speed == 1 / 2:
            self.play_speed = 1 / 4
        elif self.play_speed == 1 / 4:
            self.play_speed = 1 / 8
        else:
            self.play_speed = 1

    def _output_current_counter(self, visualizer=None):
        """
        Output the current counter value.
        """
        print(self.counter)

    def _shutdown_cache_manager(self):
        """
        Stop the background mesh cache manager.
        """
        if self.mesh_cache_manager is not None:
            self.mesh_cache_manager.shutdown()

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
        try:
            self.app.run()
        finally:
            self._shutdown_cache_manager()
