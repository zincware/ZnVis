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
Module for the particle parent class
"""
from dataclasses import dataclass
from znvis.mesh import Mesh
import numpy as np


@dataclass
class Particle:
    """
    Parent class for a ZnVis particle.

    Parameters
    ----------
    name : str
            Name of the particle
    mesh : Mesh
            Mesh to use e.g. sphere
    position : np.ndarray
            Position tensor of the shape (n_confs, n_particles, n_dims)
    velocity : np.ndarray
            Velocity tensor of the shape (n_confs, n_particles, n_dims)
    force : np.ndarray
            Force tensor of the shape (n_confs, n_particles, n_dims)
    director: np.ndarray
            Director tensor of the shape (n_confs, n_particles, n_dims)
    """
    name: str
    mesh: Mesh = None
    position: np.ndarray = None
    velocity: np.ndarray = None
    force: np.ndarray = None
    director: np.ndarray = None
