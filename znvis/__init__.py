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
init file for the main ZnVis package.
"""
from znvis import rendering
from znvis.bounding_objects.bounding_box import BoundingBox
from znvis.material.material import Material
from znvis.mesh.custom import CustomMesh
from znvis.mesh.cylinder import Cylinder
from znvis.mesh.sphere import Sphere
from znvis.particle.particle import Particle
from znvis.visualizer.visualizer import Visualizer

__all__ = [
    Particle.__name__,
    Sphere.__name__,
    Visualizer.__name__,
    Cylinder.__name__,
    CustomMesh.__name__,
    BoundingBox.__name__,
    Material.__name__,
    rendering.__name__,
]
