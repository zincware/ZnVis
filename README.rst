|madewithpython| |build| |license|

ZnVis
-----
A visualisation engine for particle simulations.

ZnVis is a Python only visualization engine aimed at the live visualization of particle
simulation.
Simply define the particles in the simulation with details including their positions,
colour, direction, and shape, and the visualization engine will display the system
using the Open3D engine.

Installation
^^^^^^^^^^^^
ZnVis is a purely Python package hosted on PyPi.
It can therefore be installed using pip with:

.. code-block:: bash

   pip install znvis

If you prefer to access the source code, run the following from a terminal:

.. code-block:: bash

   git clone https://github.com/zincware/ZnVis.git
   cd ZnVis
   pip install .

Once complete, you will be able to start using the visualizer by importing it as:

.. code-block:: python

   import znvis

Known limitations
^^^^^^^^^^^^^^^^^
Currently it seems that for medium sized systems (400 particles) running the visualizer
can result in a memory failure after a while.
We are working on how to extend ZnVis to arbitrary sized systems.


.. badges

.. |madewithpython| image:: https://img.shields.io/badge/Made%20With-Python-blue.svg?style=flat
    :alt: Made with Python

.. |build| image:: https://github.com/zincware/ZnVis/actions/workflows/pytest.yaml/badge.svg
    :alt: Build tests passing
    :target: https://github.com/zincware/ZnVis/blob/readme_badges/

.. |license| image:: https://img.shields.io/badge/License-EPLv2.0-purple.svg?style=flat
    :alt: Project license

.. |coverage| image:: https://coveralls.io/repos/github/zincware/ZnVis/badge.svg?branch=main
    :alt: Coverage Report
    :target: https://coveralls.io/github/zincware/ZnVis?branch=main
