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
"""
from os import path

import setuptools

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "requirements.txt")) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [
        line
        for line in requirements_file.read().splitlines()
        if not line.startswith("#")
    ]

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ZnVis",
    version="0.0.1",
    author="zincwarecode",
    author_email="tovey.samuel@gmail.com",
    description="Visualization of particle trajectories.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zincware/ZnVis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=requirements,
)
