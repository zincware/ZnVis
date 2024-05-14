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
Mitsuba rendering module.
"""

import os

import mitsuba as mi
import numpy as np
import open3d as o3d

try:
    mi.set_variant("cuda_ad_rgb")
except AttributeError:
    mi.set_variant("scalar_rgb")
    # mi.set_variant("llvm_ad_rgb")


# Default scene dict.
default_scene_dict = {
    "type": "scene",
    "integrator": {"type": "path"},
    "light": {"type": "constant", "radiance": {"type": "rgb", "value": 1.0}},
    "sensor": {
        "type": "perspective",
        "fov": 90,
        "thefilm": {
            "type": "hdrfilm",
            "width": 4096,
            "height": 2160,
        },
        "thesampler": {
            "type": "multijitter",
            "sample_count": 64,
        },
    },
}


class Mitsuba:
    """
    Class for Mitsuba rendering.
    """

    def __init__(self, scene_dict: dict = None, update_camera: bool = True) -> None:
        """
        Initialize the Mitsuba renderer.

        Parameters
        ----------
        scene_dict : dict (default = None)
            Dictionary containing the scene information for Mitsuba.
            The mesh objects will be added to this dictionary. Do not
            include mesh material information here unless an additional
            mesh is being added to the scene from outside of ZnVis.
            If no dict is provided, a templated dict will be used.
        update_camera : bool (default = True)
            If True, the camera will be updated to look at the mesh center.
            Set this to False if you want to use a custom camera which is
            defined in the scene_dict.
        """
        if scene_dict is None:
            scene_dict = default_scene_dict
        self.scene_dict = scene_dict
        self.update_camera = update_camera

    def _update_camera(self, view_matrix: np.ndarray) -> None:
        """
        Update the camera to look at the mesh center.

        Parameters
        ----------
        view_matrix : np.ndarray
            View matrix for the camera from open3d.

        Notes
        -----
        This function updates the camera in the scene_dict.
        It should be called before rendering.
        """
        to_world_matrix = np.linalg.inv(view_matrix)

        # Step 2: Adjust the coordinate system by flipping the Z-axis
        z_flip_matrix = np.array(
            [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        adjusted_to_world_matrix = np.dot(to_world_matrix, z_flip_matrix)

        self.scene_dict["sensor"]["to_world"] = mi.ScalarTransform4f(
            adjusted_to_world_matrix
        )

    def render_mesh_objects(
        self,
        mesh_objects: dict,
        view_matrix: np.ndarray,
        save_dir: str = "./",
        save_name: str = "znvis_render.exr",
    ):
        """
        Render mesh objects.

        Parameters
        ----------
        mesh_objects : list
            List of mesh objects to render.
        view_matrix : np.ndarray
            View matrix for the camera from open3d.
        save_dir : str (default = "./")
            Directory to save the rendered image.
        save_name : str (default = "znvis_render.exr")
            Name of the rendered image.

        Returns
        -------
        Saves an image to disk.
        """
        # Update camera.
        if self.update_camera:
            self._update_camera(view_matrix)

        # Add mesh objects to scene dict.
        for mesh_name in mesh_objects:
            bsdf = mesh_objects[mesh_name]["bsdf"]
            material = mesh_objects[mesh_name]["material"]

            # Convert to a tensor mesh.
            # print(mesh_objects[mesh_name]["mesh"].material)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(
                mesh_objects[mesh_name]["mesh"]
            )

            if bsdf is None:
                mesh.material.set_default_properties()
                mesh.material.material_name = "defaultLit"
                mesh.material.vector_properties["base_color"] = material.base_color
                mesh.material.scalar_properties["roughness"] = material.base_roughness
                mesh.material.scalar_properties["metallic"] = material.base_metallic
                mesh.material.scalar_properties["reflectance"] = (
                    material.base_reflectance
                )
                mesh.material.scalar_properties["anisotropy"] = material.base_anisotropy

            # Convert to Mitsuba mesh
            mitsuba_mesh = mesh.to_mitsuba(mesh_name, bsdf=bsdf)
            # Add to scene dict.
            self.scene_dict[mesh_name] = mitsuba_mesh

        # Render the scene.
        scene = mi.load_dict(self.scene_dict)

        img = mi.render(scene)
        bmp = mi.Bitmap(img)

        bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
        bmp.write(os.path.join(save_dir, save_name))
