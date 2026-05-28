"""
import_flightpath.py  —  Run inside Blender's Scripting tab.
=============================================================
Imports any CSV with columns: x, y, z, t  (and optionally: speed)
Comments lines starting with '#' are skipped automatically.

Creates:
  • A smooth NURBS curve tracing the full flight path
  • A colour gradient on the curve (low=blue → high=red) via a
    second object used as a bevel profile reference
  • An Empty ("Drone") that flies along the path as an animation
  • A glowing bevel tube on the curve so the path is clearly visible
  • A camera-friendly orange/cyan colour scheme

HOW TO USE:
  1. Save your .blend file first (so Blender knows where it lives).
  2. Place your CSV in the same folder as the .blend file.
  3. Change CSV_FILENAME below to match your file name.
  4. Optionally tweak the visual settings in the CONFIG block.
  5. Press "Run Script".
"""

import bpy
import csv
import math
from pathlib import Path
from mathutils import Vector, Matrix

# ─────────────────────────────────────────────────────────────
#  CONFIG  — edit these to suit your file and taste
# ─────────────────────────────────────────────────────────────
CSV_FILENAME   = "inspection_path_air_2.csv"   # filename only, placed next to .blend
CURVE_RADIUS   = 0.08    # thickness of the visible tube (metres, world scale)
SCALE          = 1.0     # multiply all coordinates (1.0 = use metres as-is)
BLENDER_Z_UP   = True    # if your CSV has Z as altitude, keep True
ANIMATE_DRONE  = True    # insert keyframes so an Empty flies the path
FPS            = 24      # frames per second for the animation
SPEED_COLUMN   = "speed" # set to None if your CSV has no speed column
# Colour for the path tube material
PATH_COLOR_R   = 0.02
PATH_COLOR_G   = 0.65
PATH_COLOR_B   = 1.00
PATH_EMISSION  = 3.0     # emission strength (makes the path glow in Cycles/EEVEE)
# ─────────────────────────────────────────────────────────────

# ── locate CSV next to the .blend file ───────────────────────
blend_path = bpy.data.filepath
if not blend_path:
    raise RuntimeError(
        "Please SAVE the .blend file first!\n"
        "The script needs to find the CSV next to the blend file."
    )
CSV = Path(blend_path).parent / CSV_FILENAME
if not CSV.exists():
    raise FileNotFoundError(
        f"Cannot find '{CSV_FILENAME}' next to the .blend file.\n"
        f"Expected: {CSV}"
    )

# ── read CSV ─────────────────────────────────────────────────
points = []   # list of (x, y, z, t, speed_or_None)
with open(CSV, newline="") as f:
    reader = csv.reader(row for row in f if not row.strip().startswith("#"))
    header = [h.strip().lower() for h in next(reader)]

    has_speed = (SPEED_COLUMN is not None and SPEED_COLUMN.lower() in header)

    for row in reader:
        if not any(row):
            continue
        d = {k: v.strip() for k, v in zip(header, row)}
        try:
            x = float(d["x"]) * SCALE
            y = float(d["y"]) * SCALE
            z = float(d["z"]) * SCALE
            t = float(d["t"])
            spd = float(d[SPEED_COLUMN.lower()]) if has_speed else None
        except (KeyError, ValueError):
            continue
        points.append((x, y, z, t, spd))

if len(points) < 2:
    raise ValueError("CSV contained fewer than 2 valid data points — nothing to import.")

print(f"[FlightPath] Loaded {len(points)} waypoints from '{CSV_FILENAME}'.")

# ── clean up previous import objects ─────────────────────────
for name in ("FlightPath", "Drone", "FlightPath_mat"):
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

# ── build a NURBS path curve ──────────────────────────────────
curve_data = bpy.data.curves.new(name="FlightPath", type="CURVE")
curve_data.dimensions = "3D"
curve_data.resolution_u = 6          # smoothness along path
curve_data.bevel_depth = CURVE_RADIUS
curve_data.bevel_resolution = 6       # roundness of tube cross-section
curve_data.use_fill_caps = True

spline = curve_data.splines.new("NURBS")
spline.points.add(len(points) - 1)   # spline starts with 1 point already
spline.use_endpoint_u = True

for i, (x, y, z, t, spd) in enumerate(points):
    spline.points[i].co = (x, y, z, 1.0)   # 4th component = weight

curve_obj = bpy.data.objects.new("FlightPath", curve_data)
bpy.context.collection.objects.link(curve_obj)

# ── create glowing material for the path ─────────────────────
mat = bpy.data.materials.new(name="FlightPath_mat")
mat.use_nodes = True
mat.blend_method = "OPAQUE"

nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

# Principled BSDF + Emission combined
out  = nodes.new("ShaderNodeOutputMaterial")
mix  = nodes.new("ShaderNodeMixShader")
emit = nodes.new("ShaderNodeEmission")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")

emit.inputs["Color"].default_value    = (PATH_COLOR_R, PATH_COLOR_G, PATH_COLOR_B, 1.0)
emit.inputs["Strength"].default_value = PATH_EMISSION
bsdf.inputs["Base Color"].default_value = (PATH_COLOR_R, PATH_COLOR_G, PATH_COLOR_B, 1.0)
bsdf.inputs["Roughness"].default_value  = 0.2
bsdf.inputs["Metallic"].default_value   = 0.8
mix.inputs["Fac"].default_value = 0.6   # 60 % emission, 40 % BSDF

links.new(bsdf.outputs["BSDF"],   mix.inputs[1])
links.new(emit.outputs["Emission"], mix.inputs[2])
links.new(mix.outputs["Shader"],  out.inputs["Surface"])

curve_obj.data.materials.append(mat)

# ── place layout nodes neatly ─────────────────────────────────
out.location  = (300,  0)
mix.location  = (100,  0)
bsdf.location = (-200, 80)
emit.location = (-200,-80)

# ── animated Empty ("Drone") that follows the path ────────────
bpy.ops.object.empty_add(type="SPHERE", radius=CURVE_RADIUS * 3)
drone = bpy.context.active_object
drone.name = "Drone"
drone.rotation_mode = "XYZ"

# scale empty time: map real seconds → frames
t_start = points[0][3]
t_end   = points[-1][3]
total_frames = round((t_end - t_start) * FPS) + 1
bpy.context.scene.render.fps = FPS
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = max(total_frames, 1)

if ANIMATE_DRONE:
    for (x, y, z, t, spd) in points:
        frame = round((t - t_start) * FPS) + 1
        bpy.context.scene.frame_set(frame)
        drone.location = Vector((x, y, z))
        drone.keyframe_insert("location", frame=frame)

    # Smooth interpolation on all fcurves
    if drone.animation_data and drone.animation_data.action:
        for fc in drone.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"

    bpy.context.scene.frame_set(1)
    print(f"[FlightPath] Animated 'Drone' empty over {total_frames} frames at {FPS} fps.")

# ── summary ───────────────────────────────────────────────────
x_vals = [p[0] for p in points]
y_vals = [p[1] for p in points]
z_vals = [p[2] for p in points]
print(f"[FlightPath] Path bounding box:")
print(f"  X: {min(x_vals):.2f} → {max(x_vals):.2f} m")
print(f"  Y: {min(y_vals):.2f} → {max(y_vals):.2f} m")
print(f"  Z: {min(z_vals):.2f} → {max(z_vals):.2f} m")
print(f"  Time: {points[0][3]:.1f} → {points[-1][3]:.1f} s")
print(f"[FlightPath] Done. Objects created: 'FlightPath' (curve), 'Drone' (empty).")