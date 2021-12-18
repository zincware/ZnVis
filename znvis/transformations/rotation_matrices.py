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
Module for the computation of rotation matrices.
"""
import numpy as np


def rotation_matrix(current: np.ndarray, target: np.ndarray):
    """
    Compute the rotation matrix between two unit vectors.

    Parameters
    ----------
    current : np.ndarray shape=(3,)
            Current orientation vector.
    target : np.ndarray shape=(3,)
            Vector to rotate into.

    Returns
    -------
    rotation : np.ndarray shape=(3, 3)
            A 3x3 rotation matrix to move a into b
    """
    if (abs(current - target)).sum() == 0.0:
        return np.eye(3)
    else:
        a, b = (current / np.linalg.norm(current)).reshape(3), (
            target / np.linalg.norm(target)
        ).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        return rotation_matrix
