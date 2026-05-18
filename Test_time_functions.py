# ==========================================
# DRONE INSPECTION ROUTING - MAIN INTERFACE
# ==========================================
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. CORE GEOMETRY & VELOCITY MODULE (The "Dumb" Calculators) ---

def get_distance_lawnmower(R_base, R_top, H, w_arc):
    """Calculates vertical and horizontal distances for a lawnmower path."""
    if R_base == R_top:
        S = H
    else:
        S = math.sqrt(H**2 + (R_base - R_top)**2)
        
    num_strips = math.ceil((2 * math.pi * R_base) / w_arc)
    vert_dist = num_strips * S
    horiz_dist = 2 * math.pi * R_base
    
    return {"vertical": vert_dist, "horizontal": horiz_dist}

def get_distance_spiral(R_base, R_top, H, v_frame):
    """Calculates the continuous path distance for a spiral path."""
    if R_base == R_top:
        return (H / v_frame) * math.sqrt((2 * math.pi * R_base)**2 + v_frame**2)

    S = math.sqrt(H**2 + (R_base - R_top)**2)
    A_lat = math.pi * (R_base + R_top) * S
    return (A_lat / v_frame) + ((v_frame * S) / (4 * math.pi * (R_base - R_top))) * math.log(R_base / R_top)

def get_h_frame(D, h_fov_deg):
    """Horizontal footprint of the camera frame on a flat surface."""
    return 2 * D * math.tan(math.radians(h_fov_deg) / 2)

def get_v_frame(D, v_fov_deg):
    """Vertical footprint of the camera frame on a flat surface."""
    return 2 * D * math.tan(math.radians(v_fov_deg) / 2)

def get_required_pixels(D, h_fov_deg, v_fov_deg, gsd):
    """Returns required sensor resolution (width x height) in pixels."""
    n_h = int(get_h_frame(D, h_fov_deg) / gsd)
    n_v = int(get_v_frame(D, v_fov_deg) / gsd)
    return n_h, n_v

def get_w_arc(R, D, h_fov_deg, label="structure"):
    """Calculates the true curved arc footprint of a camera on a cylinder.
    Safely handles cases where the camera's FOV is wider than the cylinder."""
    alpha = math.radians(h_fov_deg)
    domain_check = ((R + D) / R) * math.sin(alpha / 2)

    if domain_check <= 1.0:
        gamma = math.asin(domain_check) - (alpha / 2)
        w_arc = 2 * R * gamma
    else:
        w_arc = 2 * R * math.acos(R / (R + D))
        print(f"Warning: Camera FOV exceeds {label} width. Capping footprint at {w_arc:.2f}m.")

    return w_arc

def get_velocity_rgb_lawnmower(v_max, gsd_v, max_blur, shutter, v_frame, v_overlap, fps):
    if shutter <= 0: return v_max, "kinematics"
    v_blur = (gsd_v * max_blur) / shutter
    v_fps  = v_frame * (1.0 - v_overlap) * fps
    candidates = {"kinematics": v_max, "blur": v_blur, "fps/v_overlap": v_fps}
    reason = min(candidates, key=candidates.get)
    return candidates[reason], reason

def get_velocity_rgb_spiral(v_max, R, v_frame, w_arc, gsd, max_blur, shutter, h_overlap, fps):
    theta = math.atan(v_frame / (2 * math.pi * R))
    if shutter <= 0: return v_max, "kinematics"
    v_blur      = (gsd * max_blur) / shutter
    v_fps_horiz = (w_arc * (1.0 - h_overlap) * fps) / math.cos(theta)
    v_fps_vert  = (v_frame * fps) / math.sin(theta)
    candidates = {
        "kinematics": v_max,
        "blur": v_blur, "fps/h_overlap": v_fps_horiz, "fps/v_overlap": v_fps_vert,
    }
    reason = min(candidates, key=candidates.get)
    return candidates[reason], reason

def get_velocity_event(v_max):
    return v_max, "kinematics"

def get_velocity_event_spiral(v_max):
    return v_max, "kinematics"

def get_velocity_hyper_lawnmower(v_kin, gsd_v, line_rate, integration_time, max_blur):
    v_sync = line_rate * gsd_v
    candidates = {"kinematics": v_kin, "line_rate": v_sync}
    if integration_time > 0:
        candidates["blur"] = (gsd_v * max_blur) / integration_time
    reason = min(candidates, key=candidates.get)
    return candidates[reason], reason


# --- 2. SIMPLIFIED TIME FUNCTIONS ---

def time_lawnmower(d_vert, d_horiz, v_vert, v_horiz):
    """Calculates time for a lawnmower path based purely on distances and speeds."""
    return (d_vert / v_vert) + (d_horiz / v_horiz)

def time_spiral(d_total, v_path):
    """Calculates time for a spiral path based purely on distance and speed."""
    return d_total / v_path


# --- 3. 3D VISUALIZATION MODULE ---

def plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, H_blade, R_blade, water_config, air_config, turbine_config, cameras):
    """Generates an interactive 3D matplotlib visualization of the dual-environment flight paths."""
    H_total = H_water + H_air_cyl + H_air_cone

    cw = cameras[water_config["camera_type"]]
    v_frame_w = get_v_frame(cw["D"], cw["v_fov"]) if "v_fov" in cw else None
    w_arc_w   = get_w_arc(R_base, cw["D"], cw["h_fov"], label="monopile")

    ca = cameras[air_config["camera_type"]]
    v_frame_a = get_v_frame(ca["D"], ca["v_fov"]) if "v_fov" in ca else None
    w_arc_a   = get_w_arc(R_base, ca["D"], ca["h_fov"], label="tower")

    ct = cameras[turbine_config["camera_type"]]
    v_frame_t = get_v_frame(ct["D"], ct["v_fov"]) if "v_fov" in ct else None
    w_arc_t   = get_w_arc(R_blade, ct["D"], ct["h_fov"], label="turbine blade")
    
    def get_radius(z):
        if z <= H_water + H_air_cyl:
            return R_base
        else:
            return R_base - ((R_base - R_top) / H_air_cone) * (z - (H_water + H_air_cyl))

    def get_lawnmower_coords(z_start, z_end, R_max, w_arc, theta_start=0.0):
        z_coords, theta_coords = [], []
        num_strips = int(np.ceil((2 * np.pi * R_max) / w_arc))
        d_theta = (2 * np.pi) / num_strips
        current_theta = theta_start

        for _ in range(num_strips):
            # Fly Up
            z_coords.extend(np.linspace(z_start, z_end, 30))
            theta_coords.extend([current_theta] * 30)
            next_theta = current_theta + d_theta
            # Translate Across Top
            z_coords.extend([z_end] * 5)
            theta_coords.extend(np.linspace(current_theta, next_theta, 5))
            current_theta = next_theta
            # Fly Down
            z_coords.extend(np.linspace(z_end, z_start, 30))
            theta_coords.extend([current_theta] * 30)
            # Translate Across Bottom
            next_theta = current_theta + d_theta
            z_coords.extend([z_start] * 5)
            theta_coords.extend(np.linspace(current_theta, next_theta, 5))
            current_theta = next_theta

        return z_coords, theta_coords

    def get_spiral_coords(z_start, z_end, w_flat, theta_start=0.0):
        num_revs = (z_end - z_start) / w_flat
        n_points = max(int(num_revs * 100), 50)
        z_coords = np.linspace(z_start, z_end, n_points)
        theta_coords = theta_start + ((z_coords - z_start) / w_flat) * (2 * np.pi)
        return z_coords.tolist(), theta_coords.tolist()

    def transform_coords(x, y, z, angle_deg, z_offset):
        """Rotates coordinates around the Y-axis to orient the blades, then translates to the tower top."""
        rad = np.radians(angle_deg)
        x_rot = x * np.cos(rad) + z * np.sin(rad)
        y_rot = y
        z_rot = -x * np.sin(rad) + z * np.cos(rad)
        return x_rot, y_rot, z_rot + z_offset

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot Tower Surface
    z_surf = np.linspace(0, H_total, 50)
    theta_surf = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta_surf, z_surf)
    r_grid = np.vectorize(get_radius)(z_grid)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, color='silver', alpha=0.3, edgecolor='none')

    # 2. Plot Sea Level Plane
    x_sea, y_sea = np.meshgrid(np.linspace(-R_base*2, R_base*2, 2), np.linspace(-R_base*2, R_base*2, 2))
    z_sea = np.full(x_sea.shape, H_water)
    ax.plot_surface(x_sea, y_sea, z_sea, color='dodgerblue', alpha=0.2)

    # 2b. Plot Monopile End Plane (top of cylindrical section, where cone begins)
    x_mp, y_mp = np.meshgrid(np.linspace(-R_base*2, R_base*2, 2), np.linspace(-R_base*2, R_base*2, 2))
    z_mp = np.full(x_mp.shape, H_water + H_air_cyl)
    ax.plot_surface(x_mp, y_mp, z_mp, color='red', alpha=0.2)

    # 3. Generate & Plot Water Path
    if water_config["flight_mode"] == "lawnmower":
        zw, tw = get_lawnmower_coords(0, H_water, R_base, w_arc_w * (1 - cw["h_overlap"]))
    else:
        zw, tw = get_spiral_coords(0, H_water, v_frame_w * (1 - cw["v_overlap"]))
    rw = np.array([get_radius(z) for z in zw])
    ax.plot(rw*np.cos(tw), rw*np.sin(tw), zw, color='blue', linewidth=1.5, label=f'Water Phase ({water_config["flight_mode"]})')


    
    theta_at_waterline = tw[-1]

    # 4. Generate & Plot Air Path
    if air_config["flight_mode"] == "lawnmower":
        za, ta = get_lawnmower_coords(H_water, H_total, R_base, w_arc_a * (1 - ca["h_overlap"]), theta_start=theta_at_waterline)
    else:
        za, ta = get_spiral_coords(H_water, H_total, v_frame_a * (1 - ca["v_overlap"]), theta_start=theta_at_waterline)
    ra = np.array([get_radius(z) for z in za])
    ax.plot(ra*np.cos(ta), ra*np.sin(ta), za, color='red', linewidth=1.5, label=f'Air Phase ({air_config["flight_mode"]})')

    air_path = list(zip(ra*np.cos(ta), ra*np.sin(ta), za))


    # with open("inspection_path.csv", "w") as f:
    #     f.write("x,y,z,t\n")
    #     for i, (x, y, z) in enumerate(zip(ra*np.cos(ta), ra*np.sin(ta), za)):
    #         f.write(f"{x:.2f},{y:.2f},{z:.2f},{(i*0.2):.2f}\n")


    # 5. Generate & Plot Turbine (3 Blades) Path & Surfaces
    if turbine_config["flight_mode"] == "lawnmower":
        zb_list, tb_list = get_lawnmower_coords(0, H_blade, R_blade, w_arc_t * (1 - ct["h_overlap"]))
    else:
        zb_list, tb_list = get_spiral_coords(0, H_blade, v_frame_t * (1 - ct["v_overlap"]))
        
    xb_arr = R_blade * np.cos(tb_list)
    yb_arr = R_blade * np.sin(tb_list)
    zb_arr = np.array(zb_list)
    
    
    # Create surface meshes for the blades
    z_b = np.linspace(0, H_blade, 20)
    theta_b = np.linspace(0, 2 * np.pi, 20)
    tb_grid, zb_grid = np.meshgrid(theta_b, z_b)
    xb_grid = R_blade * np.cos(tb_grid)
    yb_grid = R_blade * np.sin(tb_grid)
    
    turbines = []
    for idx, angle in enumerate([0, 120, 240]):
        # Plot Blade Surface
        x_surf, y_surf, z_surf = transform_coords(xb_grid, yb_grid, zb_grid, angle, H_total)
        ax.plot_surface(x_surf, y_surf, z_surf, color='gold', alpha=0.3, edgecolor='none')
        
        # Plot Blade Flight Path
        x_path, y_path, z_path = transform_coords(xb_arr, yb_arr, zb_arr, angle, H_total)
        
        turbines.append((x_path, y_path, z_path))
        # Add label only once to keep legend clean
        lbl = f'Turbine Phase ({turbine_config["flight_mode"]})' if idx == 0 else ""
        ax.plot(x_path, y_path, z_path, color='orange', linewidth=1.5, label=lbl)

    # ---------------------------------------------------------
    # TRUE 1:1 PHYSICAL ASPECT RATIO FIX
    # ---------------------------------------------------------
    ax.set_title("Full Wind Turbine Inspection Path", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude Z (m)")
    
    # Calculate the maximum required span to fit the whole structure
    max_dim = max(R_base * 4, H_total + H_blade)
    
    # Set all axes to have the exact same total span
    ax.set_xlim([-max_dim / 2, max_dim / 2])
    ax.set_ylim([-max_dim / 2, max_dim / 2])
    ax.set_zlim([0, max_dim])
    
    # Force the drawing box itself to be a perfect cube
    ax.set_box_aspect([1, 1, 1])
    
    # Create cleaner legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label['Waterline'] = mpatches.Patch(color='dodgerblue', alpha=0.4)
    by_label['Monopile End'] = mpatches.Patch(color='red', alpha=0.4)
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    # plt.show()
    
    
    return air_path, turbines
    
    


