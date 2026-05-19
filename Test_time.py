# ==========================================
# DRONE INSPECTION ROUTING - MAIN INTERFACE
# ==========================================
import math
import numpy as np
import matplotlib.pyplot as plt

# Import the core math functions from the separate file
from Test_time_functions import (
    get_distance_lawnmower,
    get_distance_spiral,
    get_h_frame,
    get_required_pixels,
    get_v_frame,
    get_w_arc,
    get_velocity_rgb_lawnmower,
    get_velocity_rgb_spiral,
    get_velocity_event,
    get_velocity_event_spiral,
    get_velocity_hyper_lawnmower,
    time_lawnmower,
    time_spiral,
    plot_inspection_route
)

# ==========================================================
# A. TOWER & TURBINE GEOMETRY CONFIGURATION
# ==========================================================
R_base = 4        # Monopile radius (constant from seafloor to tower top base)
R_top  = 2.85     # Nacelle interface radius at the top of the cone

H_water    = 60   # Underwater portion of the monopile
H_air_cyl  = 30   # Above-water cylindrical section (before cone starts)
H_air_cone = 135  # Height of the tapered tower cone

# Turbine (modelled as 3 equivalent cylinders)
R_blade = 3.5     # Abstract radius of blade cylinder
H_blade = 110     # Length of each blade

# ==========================================================
# B. CAMERA CONFIGURATIONS
# ==========================================================
# h_fov, v_fov: full-angle field of view in degrees
# D: standoff distance from structure surface (m)

rgb_camera = {
    "gsd": 0.002, "max_blur": 2.0, "shutter": 0.001,
    "h_fov": 63.0, "v_fov": 46.0, "D": 2.0,
    "h_overlap": 0.75, "v_overlap": 0.75, "fps": 10,
}
event_camera = {
    "h_fov": 60.0, "v_fov": 45.0, "D":2.0,
    "h_overlap": 0.2, "v_overlap": 0.2,
}
hyperspectral_camera = {
    "gsd": 0.004, "max_blur": 2.0, "h_fov": 38.0,
    "D": 2.0, "line_rate": 330, "integration": 0.003, "h_overlap": 0.2,
}  # [Specimen AFX10]

cameras = {"RGB": rgb_camera, "EVENT": event_camera, "HYPERSPECTRAL": hyperspectral_camera}

# ==========================================================
# C. PHASE CONFIGURATIONS
# ==========================================================

# WATER PROFILE
water_config = {
    "camera_type": "RGB",
    "flight_mode": "lawnmower",
    "v_max":   5,    # [m/s] max vertical scan speed (camera limits may reduce further)
    "v_horiz": 0.2,    # [m/s] horizontal step speed between strips (no camera constraint)
}

# AIR PROFILE  ← quadcopterSC.py imports this
air_config = {
    "camera_type": "RGB",
    "flight_mode": "lawnmower",
    "v_max":   5,    # [m/s] max vertical scan speed (camera limits may reduce further)
    "v_horiz": 0.2,    # [m/s] horizontal step speed between strips (no camera constraint)
}

# TURBINE PROFILE
turbine_config = {
    "camera_type": "RGB",
    "flight_mode": "lawnmower",
    "v_max":   5,    # [m/s] max vertical scan speed (camera limits may reduce further)
    "v_horiz": 0.2,    # [m/s] horizontal step speed between strips (no camera constraint)
}

transition_penalty_seconds = 45.0

# ---  MAIN EXECUTION LOGIC ---
if __name__ == "__main__":
    # DONT CHANGE CODE AFTER THIS POINT
    # ---------------------------------------------------------
    # D. EXECUTE PHASE 1: UNDERWATER
    # ---------------------------------------------------------
    t_water = 0.0
    cw = cameras[water_config["camera_type"]]
    v_frame_w = get_v_frame(cw["D"], cw["v_fov"]) if "v_fov" in cw else None
    w_arc_w   = get_w_arc(R_base, cw["D"], cw["h_fov"], label="monopile")

    if water_config["flight_mode"] == "lawnmower":
        dist_w = get_distance_lawnmower(R_base, R_base, H_water, w_arc_w * (1 - cw["h_overlap"]))

        if water_config["camera_type"] == "RGB":
            v_vert, limit = get_velocity_rgb_lawnmower(water_config["v_max"], cw["gsd"], cw["max_blur"], cw["shutter"], v_frame_w, cw["v_overlap"], cw["fps"])
        elif water_config["camera_type"] == "EVENT":
            v_vert, limit = get_velocity_event(water_config["v_max"])
        elif water_config["camera_type"] == "HYPERSPECTRAL":
            v_vert, limit = get_velocity_hyper_lawnmower(water_config["v_max"], cw["gsd"], cw["line_rate"], cw["integration"], cw["max_blur"])

        nstrips_w  = math.ceil((2 * math.pi * R_base) / (w_arc_w * (1 - cw["h_overlap"])))
        spacing_w  = (2 * math.pi * R_base) / nstrips_w
        t_water = time_lawnmower(dist_w["vertical"], dist_w["horizontal"], v_vert, water_config["v_horiz"])
        water_vel_str  = f"v_scan={v_vert:.3f} m/s [{limit}], v_horiz={water_config['v_horiz']:.3f} m/s"
        water_dist_str = f"vert={dist_w['vertical']:.1f}m, horiz={dist_w['horizontal']:.1f}m, spacing={spacing_w:.3f}m, n_strips={nstrips_w}"

    elif water_config["flight_mode"] == "spiral":
        pitch_w = v_frame_w * (1 - cw["v_overlap"])
        dist_w = get_distance_spiral(R_base, R_base, H_water, pitch_w)

        if water_config["camera_type"] == "RGB":
            v_path, limit = get_velocity_rgb_spiral(water_config["v_max"], R_base, pitch_w, w_arc_w, cw["gsd"], cw["max_blur"], cw["shutter"], cw["h_overlap"], cw["fps"])
        elif water_config["camera_type"] == "EVENT":
            v_path, limit = get_velocity_event_spiral(water_config["v_max"])

        t_water = time_spiral(dist_w, v_path)
        water_vel_str  = f"v_path={v_path:.3f} m/s [{limit}]"
        water_dist_str = f"path={dist_w:.1f}m, pitch={pitch_w:.3f}m"

    # ---------------------------------------------------------
    # E. EXECUTE PHASE 2: AIR
    # ---------------------------------------------------------
    t_air = 0.0
    ca = cameras[air_config["camera_type"]]
    v_frame_a = get_v_frame(ca["D"], ca["v_fov"]) if "v_fov" in ca else None
    w_arc_a   = get_w_arc(R_base, ca["D"], ca["h_fov"], label="tower")

    if air_config["flight_mode"] == "lawnmower":
        dist_air_cyl = get_distance_lawnmower(R_base, R_base, H_air_cyl, w_arc_a * (1 - ca["h_overlap"]))
        dist_air_cone = get_distance_lawnmower(R_base, R_top, H_air_cone, w_arc_a * (1 - ca["h_overlap"]))

        total_vert_air = dist_air_cyl["vertical"] + dist_air_cone["vertical"]
        total_horiz_air = dist_air_cyl["horizontal"]

        if air_config["camera_type"] == "RGB":
            v_vert, limit = get_velocity_rgb_lawnmower(air_config["v_max"], ca["gsd"], ca["max_blur"], ca["shutter"], v_frame_a, ca["v_overlap"], ca["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_vert, limit = get_velocity_event(air_config["v_max"])
        elif air_config["camera_type"] == "HYPERSPECTRAL":
            v_vert, limit = get_velocity_hyper_lawnmower(air_config["v_max"], ca["gsd"], ca["line_rate"], ca["integration"], ca["max_blur"])

        nstrips_a  = math.ceil((2 * math.pi * R_base) / (w_arc_a * (1 - ca["h_overlap"])))
        spacing_a  = (2 * math.pi * R_base) / nstrips_a
        t_air = time_lawnmower(total_vert_air, total_horiz_air, v_vert, air_config["v_horiz"])
        air_vel_str  = f"v_scan={v_vert:.3f} m/s [{limit}], v_horiz={air_config['v_horiz']:.3f} m/s"
        air_dist_str = f"vert={total_vert_air:.1f}m, horiz={total_horiz_air:.1f}m, spacing={spacing_a:.3f}m, n_strips={nstrips_a}"

    elif air_config["flight_mode"] == "spiral":
        pitch_a = v_frame_a * (1 - ca["v_overlap"])
        dist_air_cyl = get_distance_spiral(R_base, R_base, H_air_cyl, pitch_a)
        dist_air_cone = get_distance_spiral(R_base, R_top, H_air_cone, pitch_a)
        total_dist_air = dist_air_cyl + dist_air_cone

        if air_config["camera_type"] == "RGB":
            v_path, limit = get_velocity_rgb_spiral(air_config["v_max"], R_base, pitch_a, w_arc_a, ca["gsd"], ca["max_blur"], ca["shutter"], ca["h_overlap"], ca["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_path, limit = get_velocity_event_spiral(air_config["v_max"])

        t_air = time_spiral(total_dist_air, v_path)
        air_vel_str  = f"v_path={v_path:.3f} m/s [{limit}]"
        air_dist_str = f"path={total_dist_air:.1f}m, pitch={pitch_a:.3f}m"

    # ---------------------------------------------------------
    # G. EXECUTE PHASE 3: TURBINE (3 Independent Blades)
    # ---------------------------------------------------------
    t_turbine = 0.0
    ct = cameras[turbine_config["camera_type"]]
    v_frame_t = get_v_frame(ct["D"], ct["v_fov"]) if "v_fov" in ct else None
    w_arc_t   = get_w_arc(R_blade, ct["D"], ct["h_fov"], label="turbine blade")

    if turbine_config["flight_mode"] == "lawnmower":
        dist_t = get_distance_lawnmower(R_blade, R_blade, H_blade, w_arc_t * (1 - ct["h_overlap"]))

        if turbine_config["camera_type"] == "RGB":
            v_vert, limit = get_velocity_rgb_lawnmower(turbine_config["v_max"], ct["gsd"], ct["max_blur"], ct["shutter"], v_frame_t, ct["v_overlap"], ct["fps"])
        elif turbine_config["camera_type"] == "EVENT":
            v_vert, limit = get_velocity_event(turbine_config["v_max"])
        elif turbine_config["camera_type"] == "HYPERSPECTRAL":
            v_vert, limit = get_velocity_hyper_lawnmower(turbine_config["v_max"], ct["gsd"], ct["line_rate"], ct["integration"], ct["max_blur"])

        nstrips_t  = math.ceil((2 * math.pi * R_blade) / (w_arc_t * (1 - ct["h_overlap"])))
        spacing_t  = (2 * math.pi * R_blade) / nstrips_t
        t_turbine = 3 * time_lawnmower(dist_t["vertical"], dist_t["horizontal"], v_vert, turbine_config["v_horiz"])
        turbine_vel_str  = f"v_scan={v_vert:.3f} m/s [{limit}], v_horiz={turbine_config['v_horiz']:.3f} m/s"
        turbine_dist_str = f"vert={3*dist_t['vertical']:.1f}m, horiz={3*dist_t['horizontal']:.1f}m, spacing={spacing_t:.3f}m, n_strips={nstrips_t} (3 blades)"

    elif turbine_config["flight_mode"] == "spiral":
        pitch_t = v_frame_t * (1 - ct["v_overlap"])
        dist_t = get_distance_spiral(R_blade, R_blade, H_blade, pitch_t)

        if turbine_config["camera_type"] == "RGB":
            v_path, limit = get_velocity_rgb_spiral(turbine_config["v_max"], R_blade, pitch_t, w_arc_t, ct["gsd"], ct["max_blur"], ct["shutter"], ct["h_overlap"], ct["fps"])
        elif turbine_config["camera_type"] == "EVENT":
            v_path, limit = get_velocity_event_spiral(turbine_config["v_max"])

        t_turbine = 3 * time_spiral(dist_t, v_path)
        turbine_vel_str  = f"v_path={v_path:.3f} m/s [{limit}]"
        turbine_dist_str = f"path={3*dist_t:.1f}m, pitch={pitch_t:.3f}m (3 blades)"

    # ---------------------------------------------------------
    # H. FINAL OUTPUT
    # ---------------------------------------------------------
    total_inspection_time = t_water + transition_penalty_seconds + t_air + t_turbine
    
    print("=== CAMERA SENSOR REQUIREMENTS ===")
    for name, cam in cameras.items():
        if "gsd" not in cam:
            print(f"{name}: N/A (no GSD — event-based sensor)")
        elif "line_rate" in cam:
            n_h = int(get_h_frame(cam["D"], cam["h_fov"]) / cam["gsd"])
            print(f"{name}: {n_h} pixels (pushbroom — vertical resolution set by line_rate)")
        else:
            n_h, n_v = get_required_pixels(cam["D"], cam["h_fov"], cam["v_fov"], cam["gsd"])
            print(f"{name}: {n_h} x {n_v} pixels")
    print()
    col1 = 45
    col2 = 65
    print("=== INSPECTION ROUTE RESULTS ===")
    print(f"{'Water Phase   (' + water_config['camera_type'] + ' / ' + water_config['flight_mode'].capitalize() + '):':<{col1}}{t_water / 60:.2f} min  |  {water_dist_str:<{col2}}|  {water_vel_str}")
    print(f"{'Transition Penalty:':<{col1}}{transition_penalty_seconds / 60:.2f} min")
    print(f"{'Air Phase     (' + air_config['camera_type'] + ' / ' + air_config['flight_mode'].capitalize() + '):':<{col1}}{t_air / 60:.2f} min  |  {air_dist_str:<{col2}}|  {air_vel_str}")
    print(f"{'Turbine Phase (' + turbine_config['camera_type'] + ' / ' + turbine_config['flight_mode'].capitalize() + '):':<{col1}}{t_turbine / 60:.2f} min  |  {turbine_dist_str:<{col2}}|  {turbine_vel_str}")
    print("-" * 50)
    print(f"{'TOTAL ESTIMATED TIME:':<{col1}}{total_inspection_time / 60:.2f} min")
    
    # ---------------------------------------------------------
    # G. VISUALIZE FLIGHT PATH
    # ---------------------------------------------------------
    print("\nGenerating 3D interactive plot...")
    plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, H_blade, R_blade, water_config, air_config, turbine_config, cameras)