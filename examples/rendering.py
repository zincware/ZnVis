import open3d as o3d
import mitsuba as mi


def render_mesh(mesh, mesh_center):
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {
            'type': 'path'
        },
        'light': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': 1.0
            }
            # NOTE: For better results comment out the constant emitter above
            # and uncomment out the lines below changing the filename to an HDRI
            # envmap you have.
            # 'type': 'envmap',
            # 'filename': '/home/renes/Downloads/solitude_interior_4k.exr'
        },
        'sensor': {
            'type':
                'perspective',
            'focal_length':
                '50mm',
            'to_world':
                mi.ScalarTransform4f.look_at(origin=[0, 0, 5],
                                             target=mesh_center,
                                             up=[0, 1, 0]),
            'thefilm': {
                'type': 'hdrfilm',
                'width': 1024,
                'height': 768,
            },
            'thesampler': {
                'type': 'multijitter',
                'sample_count': 64,
            },
        },
        'themesh': mesh,
    })

    img = mi.render(scene, spp=256)
    return img


# Default to LLVM variant which should be available on all
# platforms. If you have a system with a CUDA device then comment out LLVM
# variant and uncomment cuda variant
mi.set_variant('llvm_ad_rgb')
# mi.set_variant('cuda_ad_rgb')


mesh = o3d.t.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh.compute_vertex_normals()
mesh_center = mesh.get_axis_aligned_bounding_box().get_center()


# mesh.material.set_default_properties()
print('Render mesh with material converted to Mitsuba principled BSDF')
# mi_mesh = mesh.to_mitsuba('sphere')
# img = render_mesh(mi_mesh, mesh_center.numpy())
# mi.Bitmap(img).write('test.exr')


# Render with Open3D
o3d.visualization.draw(mesh)