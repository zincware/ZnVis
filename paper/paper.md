---
title: 'ZnVis: A Python Package for the Visualisation of Particle Simulation Data'
tags:
  - Python
  - visualisation
  - particle simulations
  - molecular dynamics
  - Open3D
authors:
  - name: Samuel Tovey
    orcid: 0000-0001-9537-8361
    affiliation: 1
affiliations:
  - name: Institute for Computational Physics, University of Stuttgart, Germany
    index: 1
date: 3 April 2026
bibliography: paper.bib
---

# Summary

`ZnVis` is a Python package for the interactive three-dimensional visualisation
of particle simulation data. It provides a high-level interface for rendering
particle trajectories and vector fields directly from NumPy arrays, removing the
need for intermediate file formats or external visualisation software. Built on
the Open3D rendering engine [@Zhou2018], `ZnVis` supports a range of geometric
primitives including spheres, cylinders, arrows, boxes, and custom mesh objects,
with optional photorealistic rendering via the Mitsuba renderer [@Jakob2022].

# Statement of Need

Researchers working with particle-based simulations, such as molecular dynamics,
coarse-grained models, or agent-based systems, frequently need to visualise
trajectories to verify simulation correctness, communicate results, and gain
physical insight. Existing tools often require exporting data to specialised file
formats (e.g., XYZ, PDB, GSD) and loading them into external applications such
as VMD [@Humphrey1996], OVITO [@Stukowski2010], or PyMOL. This workflow
introduces friction, particularly during rapid prototyping or when working in
Jupyter notebooks.

`ZnVis` addresses this gap by enabling visualisation directly from Python scripts
and Jupyter notebooks using NumPy arrays as input. Users define particles with
positions, orientations, and visual properties, and the package handles mesh
construction, scene management, and interactive playback. Key features include:

- Interactive 3D visualisation with playback controls (play, pause, rewind,
  speed adjustment).
- Support for time-varying particle counts, allowing particles to be added or
  removed during a simulation.
- Export of individual frames as PNG images, full trajectories as videos, and
  scenes as OBJ mesh files.
- Photorealistic rendering via Mitsuba with configurable materials (roughness,
  metallicity, reflectance).
- A library of built-in mesh primitives (sphere, cylinder, cone, box, arrow,
  torus, tetrahedron, octahedron, icosahedron, Mobius loop) alongside support
  for custom mesh objects loaded from file.

`ZnVis` has been used in research on reinforcement learning for active matter
systems and is designed to integrate seamlessly with simulation frameworks such
as SwarmRL.

# References
