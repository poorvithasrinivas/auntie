"""Run inside Blender: blender --background --python scripts/blender_render.py -- 80 200.0"""
import bpy
import math
import sys


args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ns = float(args[0]) if args else 64.0
output_path = args[1] if len(args) > 1 else "outputs/blender_highlight.png"

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 64
scene.render.resolution_x = 900
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

bpy.ops.mesh.primitive_uv_sphere_add(segments=128, ring_count=64, radius=1.0, location=(0, 0, 0))
sphere = bpy.context.object
sphere.data.materials.clear()
material = bpy.data.materials.new("ForecastSharpness")
material.use_nodes = True
bsdf = material.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.10, 0.22, 0.35, 1)
bsdf.inputs["Metallic"].default_value = 0.0
bsdf.inputs["Roughness"].default_value = 0.08
bsdf.inputs["Specular IOR Level"].default_value = 0.75
material.diffuse_color = (0.10, 0.22, 0.35, 1)
sphere.data.materials.append(material)

bpy.ops.object.light_add(type="AREA", location=(3, -4, 6))
light = bpy.context.object
light.data.energy = 900
light.data.size = 2.5

bpy.ops.object.camera_add(location=(0, 0, 7))
camera = bpy.context.object
camera.rotation_euler = (0, 0, 0)
scene.camera = camera

world = bpy.data.worlds.new("World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.07, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
scene.world = world

# ns_min/ns_max are intentionally represented directly by the Principled BSDF control.
# Blender roughness offers the visual counterpart; the exact ns is written into material metadata.
material["forecast_ns"] = ns
bsdf.inputs["Roughness"].default_value = float(1.0 / math.sqrt(ns))

scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)
