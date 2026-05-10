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
        
    num_strips = (2 * math.pi * R_base) / w_arc
    vert_dist = num_strips * S
    horiz_dist = 2 * math.pi * R_base
    
    return {"vertical": vert_dist, "horizontal": horiz_dist}

def get_distance_spiral(R_base, R_top, H, w_flat):
    """Calculates the continuous path distance for a spiral path."""
    if R_base == R_top:
        return (H / w_flat) * math.sqrt((2 * math.pi * R_base)**2 + w_flat**2)
        
    S = math.sqrt(H**2 + (R_base - R_top)**2)
    A_lat = math.pi * (R_base + R_top) * S
    return (A_lat / w_flat) + ((w_flat * S) / (4 * math.pi * (R_base - R_top))) * math.log(R_base / R_top)

def get_velocity_rgb_lawnmower(v_kin, gsd_v, max_blur, shutter, h_fov, overlap, fps):
    if shutter <= 0: return v_kin
    return min(v_kin, (gsd_v * max_blur) / shutter, h_fov * (1.0 - overlap) * fps)

def get_velocity_rgb_spiral(v_kin, R, w_flat, w_arc, gsd, max_blur, shutter, overlap, fps):
    if shutter <= 0: return v_kin
    theta = math.atan(w_flat / (2 * math.pi * R))
    v_fps = (w_arc * (1.0 - overlap) * fps) / math.cos(theta)
    return min(v_kin, (gsd * max_blur) / shutter, v_fps)

def get_velocity_event(v_kin):
    return v_kin

def get_velocity_hyper_lawnmower(v_kin, gsd_v, line_rate, integration_time, max_blur):
    v_sync = line_rate * gsd_v
    if integration_time <= 0: return min(v_kin, v_sync)
    return min(v_kin, v_sync, (gsd_v * max_blur) / integration_time)


# --- 2. SIMPLIFIED TIME FUNCTIONS ---

def time_lawnmower(d_vert, d_horiz, v_vert, v_horiz):
    """Calculates time for a lawnmower path based purely on distances and speeds."""
    return (d_vert / v_vert) + (d_horiz / v_horiz)

def time_spiral(d_total, v_path):
    """Calculates time for a spiral path based purely on distance and speed."""
    return d_total / v_path


# --- 3. 3D VISUALIZATION MODULE ---

def plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, H_blade, R_blade, water_config, air_config, turbine_config):
    """Generates an interactive 3D matplotlib visualization of the dual-environment flight paths."""
    H_total = H_water + H_air_cyl + H_air_cone
    
    def get_radius(z):
        if z <= H_water + H_air_cyl:
            return R_base
        else:
            return R_base - ((R_base - R_top) / H_air_cone) * (z - (H_water + H_air_cyl))

    def get_lawnmower_coords(z_start, z_end, R_max, w_arc):
        z_coords, theta_coords = [], []
        num_strips = int(np.ceil((2 * np.pi * R_max) / w_arc))
        d_theta = (2 * np.pi) / num_strips
        current_theta = 0.0
        
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

    def get_spiral_coords(z_start, z_end, w_flat):
        num_revs = (z_end - z_start) / w_flat
        n_points = max(int(num_revs * 100), 50)
        z_coords = np.linspace(z_start, z_end, n_points)
        theta_coords = ((z_coords - z_start) / w_flat) * (2 * np.pi)
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
        zw, tw = get_lawnmower_coords(0, H_water, R_base, water_config["w_arc"])
    else:
        zw, tw = get_spiral_coords(0, H_water, water_config["w_flat"])
    rw = np.array([get_radius(z) for z in zw])
    ax.plot(rw*np.cos(tw), rw*np.sin(tw), zw, color='blue', linewidth=1.5, label=f'Water Phase ({water_config["flight_mode"]})')

    # 4. Generate & Plot Air Path
    if air_config["flight_mode"] == "lawnmower":
        za, ta = get_lawnmower_coords(H_water, H_total, R_base, air_config["w_arc"])
    else:
        za, ta = get_spiral_coords(H_water, H_total, air_config["w_flat"])
    ra = np.array([get_radius(z) for z in za])
    ax.plot(ra*np.cos(ta), ra*np.sin(ta), za, color='red', linewidth=1.5, label=f'Air Phase ({air_config["flight_mode"]})')

    # 5. Generate & Plot Turbine (3 Blades) Path & Surfaces
    if turbine_config["flight_mode"] == "lawnmower":
        zb_list, tb_list = get_lawnmower_coords(0, H_blade, R_blade, turbine_config["w_arc"])
    else:
        zb_list, tb_list = get_spiral_coords(0, H_blade, turbine_config["w_flat"])
        
    xb_arr = R_blade * np.cos(tb_list)
    yb_arr = R_blade * np.sin(tb_list)
    zb_arr = np.array(zb_list)
    
    # Create surface meshes for the blades
    z_b = np.linspace(0, H_blade, 20)
    theta_b = np.linspace(0, 2 * np.pi, 20)
    tb_grid, zb_grid = np.meshgrid(theta_b, z_b)
    xb_grid = R_blade * np.cos(tb_grid)
    yb_grid = R_blade * np.sin(tb_grid)
    
    for idx, angle in enumerate([0, 120, 240]):
        # Plot Blade Surface
        x_surf, y_surf, z_surf = transform_coords(xb_grid, yb_grid, zb_grid, angle, H_total)
        ax.plot_surface(x_surf, y_surf, z_surf, color='gold', alpha=0.3, edgecolor='none')
        
        # Plot Blade Flight Path
        x_path, y_path, z_path = transform_coords(xb_arr, yb_arr, zb_arr, angle, H_total)
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
    plt.show()


# --- 4. MAIN INTERFACE SCRIPT ---

if __name__ == "__main__":
    # ---------------------------------------------------------
    # A. TOWER & TURBINE GEOMETRY CONFIGURATION
    # ---------------------------------------------------------
    R_base = 4.0      # Monopile radius (Constant from seafloor to tower top base)
    R_top = 2.5       # Nacelle interface radius at the top of the cone
    
    H_water = 30.0    # Underwater portion of the monopile
    H_air_cyl = 15.0  # Above-water portion of the monopile (before the cone starts)
    H_air_cone = 70.0 # Height of the tapered tower itself
    
    # Turbine Configuration (Modeled as 3 equivalent cylinders)
    R_blade = 1.2     # Abstract radius of the blade cylinder
    H_blade = 40.0    # Length of each blade
    
    # ---------------------------------------------------------
    # B. ENVIRONMENT & PAYLOAD PARAMETERS 
    # ---------------------------------------------------------
    
    # WATER PROFILE (Harsh lighting, high drag, biofouling)
    water_config = {
        "camera_type": "HYPERSPECTRAL",         # Options: "RGB", "EVENT", "HYPERSPECTRAL"
        "flight_mode": "lawnmower",   # Options: "lawnmower", "spiral"
        "w_arc": 0.8,                 # Small horizontal swath due to refraction/proximity
        "w_flat": 0.5,                # Small vertical pitch for spiral
        "v_kin_vert": 0.5,            # Slow kinematics due to water density
        "v_kin_horiz": 0.5,           
        "camera": {
            "gsd": 0.001, "max_blur": 2.0, "shutter": 0.005, 
            "h_fov": 0.6, "overlap": 0.85, "fps": 10,        
            "line_rate": 100, "integration": 0.01            
        }
    }
    
    # AIR PROFILE (Good lighting, low drag, clean surface)
    air_config = {
        "camera_type": "HYPERSPECTRAL",       
        "flight_mode": "lawnmower",      
        "w_arc": 2.5,                 
        "w_flat": 1.5,                
        "v_kin_vert": 5.0,            
        "v_kin_horiz": 4.0,           
        "camera": {
            "gsd": 0.001, "max_blur": 2.0, "shutter": 0.001, 
            "h_fov": 1.5, "overlap": 0.60, "fps": 10,        
            "line_rate": 200, "integration": 0.004           
        }
    }
    
    # TURBINE PROFILE (Separate selection for the 3 blades)
    turbine_config = {
        "camera_type": "HYPERSPECTRAL",         
        "flight_mode": "lawnmower",   
        "w_arc": 2.0,                 
        "w_flat": 1.0,                
        "v_kin_vert": 3.0,            # Might be slightly slower to navigate around blades
        "v_kin_horiz": 3.0,           
        "camera": {
            "gsd": 0.001, "max_blur": 2.0, "shutter": 0.001, 
            "h_fov": 1.5, "overlap": 0.70, "fps": 10,        
            "line_rate": 200, "integration": 0.004           
        }
    }
    
    transition_penalty_seconds = 45.0  # [s] Time taken to break surface, clear lenses, etc. 

    # ---------------------------------------------------------
    # C. EXECUTE PHASE 1: UNDERWATER (Monopile only)
    # ---------------------------------------------------------
    t_water = 0.0
    c_w = water_config["camera"]
    
    if water_config["flight_mode"] == "lawnmower":
        dist_w = get_distance_lawnmower(R_base, R_base, H_water, water_config["w_arc"])
        
        if water_config["camera_type"] == "RGB":
            v_vert = get_velocity_rgb_lawnmower(water_config["v_kin_vert"], c_w["gsd"], c_w["max_blur"], c_w["shutter"], c_w["h_fov"], c_w["overlap"], c_w["fps"])
        elif water_config["camera_type"] == "EVENT":
            v_vert = get_velocity_event(water_config["v_kin_vert"])
        elif water_config["camera_type"] == "HYPERSPECTRAL":
            v_vert = get_velocity_hyper_lawnmower(water_config["v_kin_vert"], c_w["gsd"], c_w["line_rate"], c_w["integration"], c_w["max_blur"])
            
        t_water = time_lawnmower(dist_w["vertical"], dist_w["horizontal"], v_vert, water_config["v_kin_horiz"])

    elif water_config["flight_mode"] == "spiral":
        dist_w = get_distance_spiral(R_base, R_base, H_water, water_config["w_flat"])
        
        if water_config["camera_type"] == "RGB":
            v_path = get_velocity_rgb_spiral(water_config["v_kin_vert"], R_base, water_config["w_flat"], water_config["w_arc"], c_w["gsd"], c_w["max_blur"], c_w["shutter"], c_w["overlap"], c_w["fps"])
        elif water_config["camera_type"] == "EVENT":
            v_path = get_velocity_event(water_config["v_kin_vert"])
        else:
            raise ValueError("Hyperspectral cannot fly spiral.")
            
        t_water = time_spiral(dist_w, v_path)

    # ---------------------------------------------------------
    # D. EXECUTE PHASE 2: AIR (Composite Monopile + Cone)
    # ---------------------------------------------------------
    t_air = 0.0
    c_a = air_config["camera"]
    
    if air_config["flight_mode"] == "lawnmower":
        # 1. Get distances for both segments
        dist_air_cyl = get_distance_lawnmower(R_base, R_base, H_air_cyl, air_config["w_arc"])
        dist_air_cone = get_distance_lawnmower(R_base, R_top, H_air_cone, air_config["w_arc"])
        
        # 2. COMPOSITE LOGIC: Sum vertical, but take horizontal ONLY ONCE
        total_vert_air = dist_air_cyl["vertical"] + dist_air_cone["vertical"]
        total_horiz_air = dist_air_cyl["horizontal"] 
        
        # 3. Determine velocity
        if air_config["camera_type"] == "RGB":
            v_vert = get_velocity_rgb_lawnmower(air_config["v_kin_vert"], c_a["gsd"], c_a["max_blur"], c_a["shutter"], c_a["h_fov"], c_a["overlap"], c_a["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_vert = get_velocity_event(air_config["v_kin_vert"])
        elif air_config["camera_type"] == "HYPERSPECTRAL":
            v_vert = get_velocity_hyper_lawnmower(air_config["v_kin_vert"], c_a["gsd"], c_a["line_rate"], c_a["integration"], c_a["max_blur"])
            
        t_air = time_lawnmower(total_vert_air, total_horiz_air, v_vert, air_config["v_kin_horiz"])

    elif air_config["flight_mode"] == "spiral":
        dist_air_cyl = get_distance_spiral(R_base, R_base, H_air_cyl, air_config["w_flat"])
        dist_air_cone = get_distance_spiral(R_base, R_top, H_air_cone, air_config["w_flat"])
        total_dist_air = dist_air_cyl + dist_air_cone
        
        if air_config["camera_type"] == "RGB":
            v_path = get_velocity_rgb_spiral(air_config["v_kin_vert"], R_top, air_config["w_flat"], air_config["w_arc"], c_a["gsd"], c_a["max_blur"], c_a["shutter"], c_a["overlap"], c_a["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_path = get_velocity_event(air_config["v_kin_vert"])
        else:
            raise ValueError("Hyperspectral cannot fly spiral.")
            
        t_air = time_spiral(total_dist_air, v_path)

    # ---------------------------------------------------------
    # E. EXECUTE PHASE 3: TURBINE (3 Independent Blades)
    # ---------------------------------------------------------
    t_turbine = 0.0
    c_t = turbine_config["camera"]
    
    if turbine_config["flight_mode"] == "lawnmower":
        # Calculate for one standard cylinder, then multiply by 3
        dist_t = get_distance_lawnmower(R_blade, R_blade, H_blade, turbine_config["w_arc"])
        
        if turbine_config["camera_type"] == "RGB":
            v_vert = get_velocity_rgb_lawnmower(turbine_config["v_kin_vert"], c_t["gsd"], c_t["max_blur"], c_t["shutter"], c_t["h_fov"], c_t["overlap"], c_t["fps"])
        elif turbine_config["camera_type"] == "EVENT":
            v_vert = get_velocity_event(turbine_config["v_kin_vert"])
        elif turbine_config["camera_type"] == "HYPERSPECTRAL":
            v_vert = get_velocity_hyper_lawnmower(turbine_config["v_kin_vert"], c_t["gsd"], c_t["line_rate"], c_t["integration"], c_t["max_blur"])
            
        # Multiply total required time by 3 blades
        t_turbine = 3 * time_lawnmower(dist_t["vertical"], dist_t["horizontal"], v_vert, turbine_config["v_kin_horiz"])

    elif turbine_config["flight_mode"] == "spiral":
        dist_t = get_distance_spiral(R_blade, R_blade, H_blade, turbine_config["w_flat"])
        
        if turbine_config["camera_type"] == "RGB":
            v_path = get_velocity_rgb_spiral(turbine_config["v_kin_vert"], R_blade, turbine_config["w_flat"], turbine_config["w_arc"], c_t["gsd"], c_t["max_blur"], c_t["shutter"], c_t["overlap"], c_t["fps"])
        elif turbine_config["camera_type"] == "EVENT":
            v_path = get_velocity_event(turbine_config["v_kin_vert"])
        else:
            raise ValueError("Hyperspectral cannot fly spiral.")
            
        t_turbine = 3 * time_spiral(dist_t, v_path)

    # ---------------------------------------------------------
    # F. FINAL OUTPUT
    # ---------------------------------------------------------
    total_inspection_time = t_water + transition_penalty_seconds + t_air + t_turbine
    
    print("=== INSPECTION ROUTE RESULTS ===")
    print(f"Water Phase   ({water_config['camera_type']} / {water_config['flight_mode'].capitalize()}): \t{t_water / 60:.2f} min")
    print(f"Transition Penalty: \t\t\t{transition_penalty_seconds / 60:.2f} min")
    print(f"Air Phase     ({air_config['camera_type']} / {air_config['flight_mode'].capitalize()}): \t{t_air / 60:.2f} min")
    print(f"Turbine Phase ({turbine_config['camera_type']} / {turbine_config['flight_mode'].capitalize()}): \t{t_turbine / 60:.2f} min")
    print("-" * 50)
    print(f"TOTAL ESTIMATED TIME: \t\t\t{total_inspection_time / 60:.2f} min")
    
    # ---------------------------------------------------------
    # G. VISUALIZE FLIGHT PATH
    # ---------------------------------------------------------
    print("\nGenerating 3D interactive plot...")
    plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, H_blade, R_blade, water_config, air_config, turbine_config)

# --- 1. 3D VISUALIZATION MODULE ---

def plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, H_blade, R_blade, water_config, air_config, turbine_config):
    """Generates an interactive 3D matplotlib visualization of the dual-environment flight paths."""
    H_total = H_water + H_air_cyl + H_air_cone
    
    def get_radius(z):
        if z <= H_water + H_air_cyl:
            return R_base
        else:
            return R_base - ((R_base - R_top) / H_air_cone) * (z - (H_water + H_air_cyl))

    def get_lawnmower_coords(z_start, z_end, R_max, w_arc):
        z_coords, theta_coords = [], []
        num_strips = int(np.ceil((2 * np.pi * R_max) / w_arc))
        d_theta = (2 * np.pi) / num_strips
        current_theta = 0.0
        
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

    def get_spiral_coords(z_start, z_end, w_flat):
        num_revs = (z_end - z_start) / w_flat
        n_points = max(int(num_revs * 100), 50)
        z_coords = np.linspace(z_start, z_end, n_points)
        theta_coords = ((z_coords - z_start) / w_flat) * (2 * np.pi)
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
        zw, tw = get_lawnmower_coords(0, H_water, R_base, water_config["w_arc"])
    else:
        zw, tw = get_spiral_coords(0, H_water, water_config["w_flat"])
    rw = np.array([get_radius(z) for z in zw])
    ax.plot(rw*np.cos(tw), rw*np.sin(tw), zw, color='blue', linewidth=1.5, label=f'Water Phase ({water_config["flight_mode"]})')

    # 4. Generate & Plot Air Path
    if air_config["flight_mode"] == "lawnmower":
        za, ta = get_lawnmower_coords(H_water, H_total, R_base, air_config["w_arc"])
    else:
        za, ta = get_spiral_coords(H_water, H_total, air_config["w_flat"])
    ra = np.array([get_radius(z) for z in za])
    ax.plot(ra*np.cos(ta), ra*np.sin(ta), za, color='red', linewidth=1.5, label=f'Air Phase ({air_config["flight_mode"]})')

    # 5. Generate & Plot Turbine (3 Blades) Path & Surfaces
    if turbine_config["flight_mode"] == "lawnmower":
        zb_list, tb_list = get_lawnmower_coords(0, H_blade, R_blade, turbine_config["w_arc"])
    else:
        zb_list, tb_list = get_spiral_coords(0, H_blade, turbine_config["w_flat"])
        
    xb_arr = R_blade * np.cos(tb_list)
    yb_arr = R_blade * np.sin(tb_list)
    zb_arr = np.array(zb_list)
    
    # Create surface meshes for the blades
    z_b = np.linspace(0, H_blade, 20)
    theta_b = np.linspace(0, 2 * np.pi, 20)
    tb_grid, zb_grid = np.meshgrid(theta_b, z_b)
    xb_grid = R_blade * np.cos(tb_grid)
    yb_grid = R_blade * np.sin(tb_grid)
    
    for idx, angle in enumerate([0, 120, 240]):
        # Plot Blade Surface
        x_surf, y_surf, z_surf = transform_coords(xb_grid, yb_grid, zb_grid, angle, H_total)
        ax.plot_surface(x_surf, y_surf, z_surf, color='gold', alpha=0.3, edgecolor='none')
        
        # Plot Blade Flight Path
        x_path, y_path, z_path = transform_coords(xb_arr, yb_arr, zb_arr, angle, H_total)
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
    plt.show()

