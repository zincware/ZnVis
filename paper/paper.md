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
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Jannik Drotleff
    equal-contrib: true
    affiliation: 1
  - name: Christoph Lohrmann
    affiliation: 1
  - name: David Zimmer
    affiliation: 1
affiliations:
  - name: Institute for Computational Physics, University of Stuttgart, Germany
    index: 1
date: 13 July 2026
bibliography: paper.bib
---

# Summary

`ZnVis` is a Python package for the three-dimensional visualisation of particle
simulation data. It provides a high-level interface for rendering particle
trajectories and vector fields directly from NumPy arrays, removing the need for
intermediate file formats or external visualisation software. Built on the
Open3D rendering engine [@Zhou2018], `ZnVis` supports a range of geometric
primitives including spheres, cylinders, arrows, boxes, and custom mesh objects,
with optional photorealistic rendering via the Mitsuba renderer [@Jakob2022].
Trajectories can be explored interactively in a live window or rendered
offscreen in parallel for the scripted, reproducible generation of images and
animations on machines without a display, such as compute clusters.

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
- Headless, parallel offscreen rendering for producing frames and animations
  without an interactive display, together with programmable camera paths
  (circular and zooming trajectories, keyframe and particle-following cameras)
  for scripted fly-throughs.
- Memory-efficient lazy construction and caching of per-frame meshes, enabling
  the visualisation of long trajectories that would not fit in memory if
  rendered eagerly.
- A library of built-in mesh primitives (sphere, cylinder, cone, box, arrow,
  torus, tetrahedron, octahedron, icosahedron, Mobius loop) alongside support
  for custom mesh objects loaded from file.

`ZnVis` has been used to visualise reinforcement-learning experiments on active
matter systems and integrates with simulation frameworks such as SwarmRL
[@Tovey2025].

# References
