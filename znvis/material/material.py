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
Material parent class.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    """
    Parent class for the ZnVis materials.

    Attributes
    ----------
    colour : np.ndarray
            Colour of the mesh. RGB array in the range 0, 1.
    alpha : float
            Transparancy of the mesh.
    roughness : float
            Roughness of the material.
    metallica : float
            How metallic the material looks. 0. not and 1.0 is metallic.
    reflectance: float
            How reflective the material is.
    anisotropy: float
            How anisotopic the material is.
    mitsuba_bsdf: mitsuba.bsdf (default: None)
            Mitsuba bsdf object.
    """

    colour: np.ndarray = np.array([59.0, 53.0, 97.0]) / 255
    alpha: float = 1.0
    roughness: float = 0.5
    metallic: float = 0.0
    reflectance: float = 0.4
    anisotropy: float = 0.4

    mitsuba_bsdf = None
