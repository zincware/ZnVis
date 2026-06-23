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

import os
import pathlib
import typing
from numbers import Integral

import znvis
from znvis.rendering import Mitsuba
from znvis.video import VideoManager


def build_mesh_dict_for_frame(
    particles: typing.List[znvis.Particle],
    vector_field: typing.List[znvis.VectorField] | None,
    frame_index: int,
) -> dict:
    """
    Build the renderer mesh dictionary for a specific frame.
    """
    mesh_dict = {}
    items = (vector_field or []) + particles
    for item in items:
        idx = 0 if item.static else frame_index
        mesh = (
            item.mesh_list[idx]
            if item.mesh_list is not None
            else item.get_mesh_for_frame(frame_index)
        )
        mesh_dict[item.name] = {
            "mesh": mesh,
            "bsdf": item.mesh.material.mitsuba_bsdf,
            "material": item.mesh.o3d_material,
        }
    return mesh_dict


def _detect_available_gpu_devices() -> int:
    """
    Detect number of visible CUDA GPU devices.
    """
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        devices = [d.strip() for d in cuda_visible_devices.split(",")]
        devices = [d for d in devices if d and d != "-1"]
        return len(devices)

    try:
        import drjit as dr

        jit_backend = getattr(dr, "JitBackend", None)
        has_backend = getattr(dr, "has_backend", None)
        if (
            jit_backend is not None
            and has_backend is not None
            and has_backend(jit_backend.CUDA)
        ):
            cuda_mod = getattr(dr, "cuda", None)
            if cuda_mod is not None:
                device_count_fn = getattr(cuda_mod, "device_count", None)
                if callable(device_count_fn):
                    return int(device_count_fn())
    except Exception:
        pass

    return 0


class BaseVisualizer:
    """
    Base class for visualizers containing shared functionality.

    This class provides common methods and initialization logic used by both
    the interactive Visualizer and the HeadlessVisualizer.

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
        parallel_render_workers: int | None = None,
        parallel_render: bool = False,
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
        parallel_render_workers : int, optional
                Number of worker processes to use for headless parallel
                rendering. If ``None``, ZnVis uses one worker per visible GPU.
                Only add more workers than visible GPUs if you exactly know what you're
                doing. Only implemented as experimental feature!
        parallel_render : bool, optional
                If ``True``, enables headless parallel rendering. If no visible
                CUDA GPU is available, ZnVis warns and falls back to serial
                rendering.
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
        self.available_gpu_devices = _detect_available_gpu_devices()
        if parallel_render_workers is None:
            parallel_render_workers = max(1, self.available_gpu_devices)
        elif not isinstance(parallel_render_workers, Integral) or isinstance(
            parallel_render_workers, bool
        ):
            raise ValueError(
                "parallel_render_workers must be an integer (not a boolean) "
                "greater than or equal to 1, or None for automatic selection."
            )
        if parallel_render_workers < 1:
            raise ValueError(
                "parallel_render_workers must be greater than or equal to 1."
            )
        self.parallel_render_workers = int(parallel_render_workers)
        self.parallel_render = parallel_render

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

        return build_mesh_dict_for_frame(
            particles=self.particles,
            vector_field=self.vector_field,
            frame_index=counter,
        )
