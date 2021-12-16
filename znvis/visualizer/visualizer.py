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
import open3d.visualization.gui as gui
import open3d as o3d
import time

from typing import List

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
    def __init__(self, particles: List[znvis.Particle]):
        """
        Constructor for the visualizer.

        Parameters
        ----------
        particles : list[znvis.Particle]
                List of particles to add to the visualizer.
        """
        self.particles = particles

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

        # Add additional control list
        vis_controls = gui.CollapsableVert("Visualizer Controls", 0.25 * 12,
                                         gui.Margins(12, 0, 0, 0))
        self.vis.add_action("Step forward in time", self._update_particles)
        self.vis.add_action("Run Simulation", self._continuous_trajectory)
        self.app.add_window(self.vis)

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
            for item in self.particles:
                for particle in item.mesh_dict:
                    visualizer.add_geometry(particle, item.mesh_dict[particle])
        else:
            for item in self.particles:
                for particle in item.mesh_dict:
                    visualizer.remove_geometry(particle)
                    visualizer.add_geometry(particle, item.mesh_dict[particle])

    def _continuous_trajectory(self, vis):
        """
        Callback method for running the trajectory smoothly.

        Returns
        -------
        Runs through the trajectory.
        """
        for step in range(90):
            self._update_particles(visualizer=vis)

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
            step = self.counter
        for particle in self.particles:
            particle.update_position_data(step)

        self._draw_particles(visualizer=visualizer)  # draw the particles.
        visualizer.post_redraw()  # re-draw the window.
        self.counter += 1

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
