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
Module for the StaticCamera class.
"""

import numpy as np

from znvis.cameras.base_camera import BaseCamera


class StaticCamera(BaseCamera):
    def __init__(
        self,
        center: np.ndarray = None,
        eye: np.ndarray = None,
        up: np.ndarray = None,
        view_matrix: np.ndarray = None,
    ) -> None:
        """
        Static camera with a fixed view matrix.

        Parameters
        ----------
        center : np.ndarray shape=(3,)
                The spatial point the camera looks at.
        eye : np.ndarray shape=(3,)
                The spatial position of the camera.
        up : np.ndarray shape=(3,)
                The up vector of the camera.
        ↑ (up)
        |
        |       camera looks from the eye to the center
        |
        ●------>●
        eye   center
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera. Only needed,
                if the camera is not defined by the center, eye and up vectors.

        Returns
        -------
        None
        """
        if view_matrix is not None and any(v is not None for v in (center, eye, up)):
            raise ValueError(
                "Cannot specify both view_matrix and center/eye/up vectors."
            )
        if view_matrix is not None:
            if view_matrix.shape != (4, 4):
                raise ValueError("view_matrix must have shape (4, 4).")
            self.view_matrix = view_matrix

        elif all(v is not None for v in (center, eye, up)):
            for name, vec in [("center", center), ("eye", eye), ("up", up)]:
                if not isinstance(vec, np.ndarray) or vec.shape != (3,):
                    raise ValueError(f"{name} must be a numpy array with shape (3,).")
            self.view_matrix = self.look_at(center, eye, up)

        else:
            raise ValueError(
                "Either a view_matrix or center, eye and up must be provided."
            )

    def get_view_matrix(self, frame_index: int) -> np.ndarray:
        """
        Get the current view matrix of the camera.

        Parameters
        ----------
        frame_index : int
                The frame index of the view matrix that should be returned.

        Returns
        -------
        view_matrix : np.ndarray shape=(4, 4)
                The view matrix of the camera.
        """
        return self.view_matrix
