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
Module for the KeyframeCamera class.
"""

import pathlib

import numpy as np

from znvis.cameras.base_camera import BaseCamera


class KeyframeCamera(BaseCamera):
    """
    A class to produce an smooth camera trajectory based on manually
    picked frames and camera settings pairs.
    This is done by interpolating the view matrices of the manually
    picked frames. At the moment, only linear interpolation with
    a smoothing via SVD is currently implemented.
    This can lead to unexpected results if the camera
    positions are chosen too far apart. Be aware of this
    and pick the camera positions accordingly. Especially take
    care if you want to zoom in or out. However, this way you have
    total control over the camera trajectory and can create
    very nice custom animations.

    """

    def __init__(self, view_matrices_path: pathlib.Path = None) -> None:
        """
        Initialize the Keyframe Camera object.

        Parameters
        ----------
        view_matrices_path : pathlib.Path
                The path to the view matrices.
        """
        self.view_matrices_path = view_matrices_path
        self.view_matrices_dictionary = {}
        if self.view_matrices_path is not None:
            self.load_view_matrices()

    def add_view_matrix(self, frame_index: int, view_matrix: np.ndarray) -> None:
        """
        Add the current view matrix to
        the view matrix dictionary. This function is called in the
        visualizer.py to add the current view matrix to the
        view matrix dictionary.
        """
        self.view_matrices_dictionary[frame_index] = view_matrix.copy()

    def reset_view_matrix_progress(self) -> None:
        """
        Reset the view matrix progress.
        This function is called in the visualizer.py to reset the
        view matrix recording progress.
        """
        self.view_matrices_dictionary = {}
        print("View matrix progress reset.")

    def remove_view_matrix(self, frame_index: int) -> None:
        """
        Remove the current view matrix from the view matrix dictionary.
        This function is called in the visualizer.py to remove the
        current view matrix from the view matrix dictionary.
        """
        print(self.view_matrices_dictionary.keys())
        if frame_index in self.view_matrices_dictionary.keys():

            self.view_matrices_dictionary.pop(frame_index)
            print(f"Removed view matrix for frame {frame_index}")
        else:
            print(f"No view matrix found for frame {frame_index}")

    def sort_dictionary(self) -> None:
        """
        Create the view matrix dictionary.
        This function is called in the visualizer.py to create the
        view matrix dictionary.
        """
        min_key = min(self.view_matrices_dictionary.keys())
        if min_key != 0:
            print(
                "No view matrix found for the first frame."
                "Adding the first saved view matrix of the trajectory",
                " as entry for frame 0.",
            )
            self.view_matrices_dictionary[0] = self.view_matrices_dictionary[min_key]

        # Sort the dictionary by key
        self.view_matrices_dictionary = dict(
            sorted(self.view_matrices_dictionary.items())
        )
        if self.number_of_frames - 1 not in self.view_matrices_dictionary:
            print(
                "No view matrix found for the last frame."
                "Adding the last saved view matrix of the trajectory",
                f" as entry for frame {self.number_of_frames - 1}.",
            )
            max_key = max(self.view_matrices_dictionary.keys())
            self.view_matrices_dictionary[self.number_of_frames - 1] = (
                self.view_matrices_dictionary[max_key]
            )
        return self.view_matrices_dictionary

    def interpolate_and_export_view_matrices(self) -> None:
        """
        Starts the interpolation between the keyframed view matrices
        in the given dictionary.
        The dictionary is expected to have the first key 0 and last
        key the last frame index.
        This is constructed in the visualizer.py interactively by the user.
        The interpolated view matrices are then saved to the given path.

        Parameters
        ----------
        None

        Returns
        -------
        interpolated_view_matrices : np.ndarray shape=(n_frames, 4, 4)
                The interpolated view matrices
        """
        if len(self.view_matrices_dictionary) == 0:
            print("No view matrices found. Please add view matrices first.")
        else:
            if not self.view_matrices_path.exists():
                self.view_matrices_path.mkdir(parents=True, exist_ok=True)

            sorted_view_matrices_dictionnary = self.sort_dictionary()
            interpolated_view_matrices_dictionnary = self.interpolate_view_matrices(
                sorted_view_matrices_dictionnary
            )
            self.export_interpolated_view_matrices(
                interpolated_view_matrices_dictionnary
            )

    def load_view_matrices(self) -> dict:
        """
        Loads the view matrices from the given path.

        Parameters
        ----------
        None

        Returns
        -------
        view_matrices_dictionary : dict
                A dictionary containing the view matrices of the manually
                picked
        """
        self.view_matrices_dictionary = np.load(
            self.view_matrices_path, allow_pickle=True
        )

    def interpolate_view_matrices(self, view_matrices_dictionary: dict) -> dict:
        """
        Interpolates the view matrices in the given dictionary.
        The dictionary is expected to have the first key 0 and last
        key the last frame index.
        This is constructed in the visualizer.py interactively by the user.

        Parameters
        ----------
        view_matrices_dictionary : dict
                A dictionary containing the view matrices of the manually
                picked frames.

        Returns
        -------
        interpolated_view_matrices : np.ndarray shape=(n_frames, 4, 4)
                The interpolated view matrices
        """
        frame_indexes, view_matrices = zip(*view_matrices_dictionary.items())

        interpolated_view_matrices = []
        # Catch the case where the first view matrix is not at frame 0.
        if frame_indexes[0] != 0:
            start_frame = frame_indexes[0]
            start_matrix = view_matrices[0]
            for i in range(start_frame):
                interpolated_view_matrices.append(start_matrix)

        interpolated_view_matrices.append(view_matrices[0])
        for i in range(1, len(view_matrices)):
            interpolation_steps = frame_indexes[i] - frame_indexes[i - 1]
            matrix_1 = view_matrices[i - 1]
            matrix_2 = view_matrices[i]

            # Perform a linear interpolation between the two view matrices
            for j in range(interpolation_steps):
                new_matrix = matrix_1 + j * (matrix_2 - matrix_1) / interpolation_steps

                new_matrix = self._apply_svd_smoothing(new_matrix)
                interpolated_view_matrices.append(new_matrix)
        interpolated_view_matrices = np.array(interpolated_view_matrices)

        # Make sure that the interpolated view matrices have the correct shape
        # aka a view matrix for each frame
        assert interpolated_view_matrices.shape[0] == frame_indexes[-1] + 1
        self.interpolated_view_matrices = interpolated_view_matrices

        return interpolated_view_matrices

    def export_interpolated_view_matrices(
        self, interpolated_view_matrices_dictionnary: dict
    ) -> None:
        """
        Exports the interpolated view matrices to the specified path.
        This would not be necessary if the visualizer could be run twice
        from the same script. Then the Keyframe Camera could just be passed to
        the renderer. However, Open3D makes this quite difficult so here we choose
        this approach.
        """

        export_path = self.view_matrices_path / "interpolated_view_matrices.npy"
        np.save(export_path, interpolated_view_matrices_dictionnary)
        print(f"Interpolated view matrices saved to {export_path}")
        self.export_path = export_path

    def get_view_matrix(self, frame_index: int = None) -> np.ndarray:
        """
        Get the correct view matrix for the camera for a given frame index.

        Parameters
        ----------
        frame_index : int
                The frame index of the view matrix that should be returned.

        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        """
        current_view_matrix = self.view_matrices_dictionary[frame_index]

        return current_view_matrix

    def _apply_svd_smoothing(self, view_matrix: np.ndarray) -> np.ndarray:
        """
        Applies a singular value decomposition to the view matrix to smooth the
        rotation.

        Parameters
        ----------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.

        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The smoothed view matrix.
        """

        rotation_scaling = view_matrix[:3, :3]

        # norm the column vectors
        u, _, vh = np.linalg.svd(rotation_scaling, full_matrices=False)
        rotation_matrix = np.dot(u, vh)

        # upate the View-Matrix
        view_matrix[:3, :3] = rotation_matrix
        return view_matrix
