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
Tutorial script to visualize simple spheres.
"""
import numpy as np
import znvis as zn

if __name__ == '__main__':
    """
    Run the simple spheres example.
    """
    trajectory = np.random.uniform(-10, 10, (100, 10, 3))
    particle = zn.Particle(name="Ball")


