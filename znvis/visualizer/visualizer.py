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
import threading
import time
from typing import List

import open3d as o3d
import open3d.visualization.gui as gui

import znvis


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
        particles: List[znvis.Particle],
        frame_rate: int = 24,
        number_of_steps: int = None,
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
        """
        self.particles = particles
        self.frame_rate = frame_rate

        if number_of_steps is None:
            number_of_steps = particles[0].position.shape[0]
        self.number_of_steps = number_of_steps

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

        self.vis = o3d.visualization.O3DVisualizer("ZnVis Visualizer", 1024, 768)
        self.vis.show_settings = True
        self.vis.reset_camera_to_default()
        # Add actions to the visualizer.
        self.vis.add_action("Step", self._update_particles)
        self.vis.add_action("Play", self._continuous_trajectory)
        self.vis.add_action("Export Scene", self._export_scene)
        self.vis.add_action("Screenshot", self._take_screenshot)

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

    def _export_scene(self, vis):
        """
        Export the current visualization scene.

        Parameters
        ----------
        vis : Visualizer
                The activte visualizer.

        Returns
        -------
        Stores a .gltf model locally.
        """
        old_state = self.interrupt  # get old state
        self.interrupt = 0  # stop live feed if running.
        for i, item in enumerate(self.particles):
            if i == 0:
                mesh = item
            else:
                mesh += item

        o3d.io.write_triangle_mesh(f"My_mesh_{self.counter}.ply", mesh)

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
        vis.export_current_image(f"screenshot_{self.counter}.png")

    def _initialize_particles(self):
        """
        Initialize the particles in the simulation.

        This method will construct the particle dictionaries in each Particle class and
        then add the first location of each particle to the visualizer window.

        Returns
        -------

        """
        # Build the mesh dict for each particle and add them to the window.
        for item in self.particles:
            item.construct_mesh_dict()

        self._draw_particles(initial=True)

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
        if visualizer is None:
            visualizer = self.vis
        if initial:
            for i, item in enumerate(self.particles):
                visualizer.add_geometry(item.name, item.mesh_dict[self.counter])
        else:
            for i, item in enumerate(self.particles):
                visualizer.remove_geometry(item.name)
                visualizer.add_geometry(item.name, item.mesh_dict[self.counter])

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

    def _run_trajectory(self):
        """
        Callback method for running the trajectory smoothly.

        Returns
        -------
        Runs through the trajectory.
        """
        self.interrupt = 1  # set global run state.
        while self.counter < self.number_of_steps:
            time.sleep(1 / self.frame_rate)
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
        visualizer.post_redraw()  # re-draw the window.

    def run_visualization(self):
        """
        Run the visualization.

        Returns
        -------
        Launches the visualization.
        """
        self._initialize_app()
        self._initialize_particles()

        self.vis.reset_camera_to_default()
        self.app.run()
