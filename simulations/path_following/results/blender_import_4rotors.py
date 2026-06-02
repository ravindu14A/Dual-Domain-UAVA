"""
blender_import_4rotors.py  — run inside Blender's scripting tab.
Requires: kinematics_4rotors.csv in the same directory as this script.

Creates an Empty object and inserts location + rotation keyframes
for every recorded timestep.  Set your scene FPS to match the export.
"""
import bpy, csv, math
from pathlib import Path

CSV = Path(__file__).parent / "kinematics_4rotors.csv"
FPS = round(1.0 / 0.0200)   # match simulation dt

# Remove existing "Drone" object if present
if "Drone" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["Drone"], do_unlink=True)

bpy.ops.object.empty_add(type="ARROWS")
obj = bpy.context.active_object
obj.name = "Drone"
obj.rotation_mode = "QUATERNION"
bpy.context.scene.render.fps = FPS

with open(CSV, newline="") as f:
    reader = csv.reader(row for row in f if not row.startswith("#"))
    header = next(reader)
    for row in reader:
        d = dict(zip(header, row))
        frame = round(float(d["time_s"]) * FPS) + 1
        bpy.context.scene.frame_set(frame)
        obj.location = (float(d["x_m"]), float(d["y_m"]), float(d["z_m"]))
        obj.rotation_quaternion = (
            float(d["qw"]), float(d["qx"]), float(d["qy"]), float(d["qz"])
        )
        obj.keyframe_insert("location", frame=frame)
        obj.keyframe_insert("rotation_quaternion", frame=frame)

print(f"Imported {frame} frames into Blender object \'Drone\'.")
