ZnVis Documentation
-------------------

Welcome to the documentation of ZnVis.
ZnVis is a Python packaged built on the
`Open3D <http://www.open3d.org/docs/release/index.html>`_ framework for the
visualization of particle data.
Currently ZnVis is capable of rendering spheres, cylinders, and custom .stl files but
more shapes are expected soon.

.. toctree::
   :maxdepth: 2
   :caption: Welcome Guide:

   _welcome_guide/getting_started


.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   _user_guide/user_guide


.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

   _developer_docs/developer_guide

Known limitations
^^^^^^^^^^^^^^^^^
Currently it seems that for medium sized systems (400 particles) running the visualizer
can result in a memory failure after a while.
We are working on how to extend ZnVis to arbitrary sized systems.
