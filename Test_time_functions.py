# drone inspection routing - helper functions
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_distance_lawnmower(R_base, R_top, H, w_arc):
    """Vertical and horizontal distances for a lawnmower path."""
    if R_base == R_top:
        S = H
    else:
        S = math.sqrt(H**2 + (R_base - R_top)**2)
        
    num_strips = math.ceil((2 * math.pi * R_base) / w_arc)
    vert_dist = num_strips * S
    horiz_dist = 2 * math.pi * R_base
    
    return {"vertical": vert_dist, "horizontal": horiz_dist}

def get_distance_spiral(R_base, R_top, H, v_frame):
    """Continuous path distance for a spiral."""
    if R_base == R_top:
        return (H / v_frame) * math.sqrt((2 * math.pi * R_base)**2 + v_frame**2)

    S = math.sqrt(H**2 + (R_base - R_top)**2)
    A_lat = math.pi * (R_base + R_top) * S
    return (A_lat / v_frame) + ((v_frame * S) / (4 * math.pi * (R_base - R_top))) * math.log(R_base / R_top)

def get_h_frame(D, h_fov_deg):
    """Horizontal footprint on a flat surface."""
    return 2 * D * math.tan(math.radians(h_fov_deg) / 2)

def get_v_frame(D, v_fov_deg):
    """Vertical footprint on a flat surface."""
    return 2 * D * math.tan(math.radians(v_fov_deg) / 2)

def get_required_pixels(D, h_fov_deg, v_fov_deg, gsd):
    """Required sensor resolution (width x height) in pixels."""
    n_h = int(get_h_frame(D, h_fov_deg) / gsd)
    n_v = int(get_v_frame(D, v_fov_deg) / gsd)
    return n_h, n_v

def get_w_arc(R, D, h_fov_deg, label="structure"):
    """True curved arc footprint of camera on a cylinder.
    Handles FOV wider than the cylinder gracefully."""
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


def time_lawnmower(d_vert, d_horiz, v_vert, v_horiz):
    """Total time for a lawnmower path."""
    return (d_vert / v_vert) + (d_horiz / v_horiz)

def time_spiral(d_total, v_path):
    """Total time for a spiral path."""
    return d_total / v_path


def get_lawnmower_coords(z_start, z_end, R_max, w_arc, theta_start=0.0):
    """Generate (z, theta) sample points for a lawnmower path (one full revolution).
    Returns z_coords, theta_coords as lists."""
    z_coords, theta_coords = [], []
    num_strips = int(np.ceil((2 * np.pi * R_max) / w_arc))
    d_theta = (2 * np.pi) / num_strips
    current_theta = theta_start

    for i in range(num_strips):
        going_up = (i % 2 == 0)
        z_from = z_start if going_up else z_end
        z_to   = z_end   if going_up else z_start
        # vertical strip
        z_coords.extend(np.linspace(z_from, z_to, 30))
        theta_coords.extend([current_theta] * 30)
        # horizontal step (skip after final strip)
        if i < num_strips - 1:
            next_theta = current_theta + d_theta
            z_coords.extend([z_to] * 30)
            theta_coords.extend(np.linspace(current_theta, next_theta, 30))
            current_theta = next_theta

    return z_coords, theta_coords


def get_spiral_coords(z_start, z_end, w_flat, theta_start=0.0):
    """Generate (z, theta) sample points for a spiral path.
    Returns z_coords, theta_coords as lists."""
    num_revs = (z_end - z_start) / w_flat
    n_points = max(int(num_revs * 100), 50)
    z_coords = np.linspace(z_start, z_end, n_points)
    theta_coords = theta_start + ((z_coords - z_start) / w_flat) * (2 * np.pi)
    return z_coords.tolist(), theta_coords.tolist()


def plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, blade_span, blade_chord, water_config, air_config, turbine_config, cameras):
    """3D matplotlib plot of the dual-environment flight paths."""
    H_total = H_water + H_air_cyl + H_air_cone

    cw = cameras[water_config["camera_type"]]
    v_frame_w = get_v_frame(cw["D"], cw["v_fov"]) if "v_fov" in cw else None
    w_arc_w   = get_w_arc(R_base, cw["D"], cw["h_fov"], label="monopile")

    ca = cameras[air_config["camera_type"]]
    v_frame_a = get_v_frame(ca["D"], ca["v_fov"]) if "v_fov" in ca else None
    w_arc_a   = get_w_arc(R_base, ca["D"], ca["h_fov"], label="tower")

    ct = cameras[turbine_config["camera_type"]]

    def get_radius(z):
        if z <= H_water + H_air_cyl:
            return R_base
        else:
            return R_base - ((R_base - R_top) / H_air_cone) * (z - (H_water + H_air_cyl))

    def transform_coords(x, y, z, angle_deg, z_offset):
        """Rotate around Y-axis then translate to tower top."""
        rad = np.radians(angle_deg)
        x_rot = x * np.cos(rad) + z * np.sin(rad)
        y_rot = y
        z_rot = -x * np.sin(rad) + z * np.cos(rad)
        return x_rot, y_rot, z_rot + z_offset

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # tower surface: cylinder + cone
    theta_surf = np.linspace(0, 2 * np.pi, 60)
    _THc, _Zc = np.meshgrid(theta_surf, np.linspace(0, H_water + H_air_cyl, 30))
    ax.plot_surface(R_base * np.cos(_THc), R_base * np.sin(_THc), _Zc,
                    color='silver', alpha=0.3, edgecolor='none')
    _THt, _Zt = np.meshgrid(theta_surf, np.linspace(H_water + H_air_cyl, H_total, 50))
    _Rt = R_base + (R_top - R_base) * (_Zt - (H_water + H_air_cyl)) / H_air_cone
    ax.plot_surface(_Rt * np.cos(_THt), _Rt * np.sin(_THt), _Zt,
                    color='silver', alpha=0.3, edgecolor='none')

    # sea level and monopile end planes
    x_sea, y_sea = np.meshgrid(np.linspace(-R_base*2, R_base*2, 2), np.linspace(-R_base*2, R_base*2, 2))
    z_sea = np.full(x_sea.shape, H_water)
    ax.plot_surface(x_sea, y_sea, z_sea, color='dodgerblue', alpha=0.2)

    x_mp, y_mp = np.meshgrid(np.linspace(-R_base*2, R_base*2, 2), np.linspace(-R_base*2, R_base*2, 2))
    z_mp = np.full(x_mp.shape, H_water + H_air_cyl)
    ax.plot_surface(x_mp, y_mp, z_mp, color='red', alpha=0.2)

    # water path
    if water_config["flight_mode"] == "lawnmower":
        zw, tw = get_lawnmower_coords(0, H_water, R_base, w_arc_w * (1 - cw["h_overlap"]))
    else:
        zw, tw = get_spiral_coords(0, H_water, v_frame_w * (1 - cw["v_overlap"]))
    rw = np.array([get_radius(z) + cw["D"] for z in zw])
    ax.plot(rw*np.cos(tw), rw*np.sin(tw), zw, color='blue', linewidth=1.5, label=f'Water Phase ({water_config["flight_mode"]})')

    theta_at_waterline = tw[-1]

    # air path
    if air_config["flight_mode"] == "lawnmower":
        za, ta = get_lawnmower_coords(H_water, H_total, R_base, w_arc_a * (1 - ca["h_overlap"]), theta_start=theta_at_waterline)
    else:
        za, ta = get_spiral_coords(H_water, H_total, v_frame_a * (1 - ca["v_overlap"]), theta_start=theta_at_waterline)
    ra = np.array([get_radius(z) + ca["D"] for z in za])
    ax.plot(ra*np.cos(ta), ra*np.sin(ta), za, color='red', linewidth=1.5, label=f'Air Phase ({air_config["flight_mode"]})')

    # turbine path and surfaces — slab model (top + bottom faces only)
    D_t        = ct["D"]
    strip_w_t  = 2.0 * D_t * np.tan(np.radians(ct["h_fov"] / 2))
    strip_step = strip_w_t * (1.0 - ct["h_overlap"])
    n_strips_t = int(np.ceil(blade_chord / strip_step))
    x_strips   = np.linspace(-blade_chord / 2, blade_chord / 2, n_strips_t)

    # Blade slab surface mesh (flat rectangle at y=0 in blade-local frame)
    xb_sg, zb_sg = np.meshgrid(np.linspace(-blade_chord / 2, blade_chord / 2, 4),
                                np.linspace(0, blade_span, 30))
    yb_sg = np.zeros_like(xb_sg)

    # Lawnmower path — one continuous connected path: top surface → root transition → bottom surface
    xb_path, yb_path, zb_path = [], [], []

    # Top surface (+D)
    for i, xb in enumerate(x_strips):
        z0 = 0.0 if i % 2 == 0 else float(blade_span)
        z1 = float(blade_span) if i % 2 == 0 else 0.0
        for z in np.linspace(z0, z1, 20):
            xb_path.append(xb); yb_path.append(D_t); zb_path.append(z)
        if i < n_strips_t - 1:
            z_here = zb_path[-1]
            for x2 in np.linspace(xb, x_strips[i + 1], 5)[1:]:
                xb_path.append(x2); yb_path.append(D_t); zb_path.append(z_here)

    # Transition at blade TIP: fly to z=blade_span, sweep y from +D to -D
    x_end = xb_path[-1]; z_end = zb_path[-1]
    if abs(z_end - float(blade_span)) > 0.01:
        for z in np.linspace(z_end, float(blade_span), 10)[1:]:
            xb_path.append(x_end); yb_path.append(D_t); zb_path.append(z)
    for y in np.linspace(D_t, -D_t, 8)[1:]:
        xb_path.append(x_end); yb_path.append(y); zb_path.append(float(blade_span))

    # Bottom surface (-D): starts at tip, reversed z parity (scans toward root)
    x_strips_bot = x_strips[::-1]
    for i, xb in enumerate(x_strips_bot):
        z0 = float(blade_span) if i % 2 == 0 else 0.0
        z1 = 0.0 if i % 2 == 0 else float(blade_span)
        for z in np.linspace(z0, z1, 20):
            xb_path.append(xb); yb_path.append(-D_t); zb_path.append(z)
        if i < n_strips_t - 1:
            z_here = zb_path[-1]
            for x2 in np.linspace(xb, x_strips_bot[i + 1], 5)[1:]:
                xb_path.append(x2); yb_path.append(-D_t); zb_path.append(z_here)

    xb_path = np.array(xb_path);  yb_path = np.array(yb_path);  zb_path = np.array(zb_path)

    for idx, angle in enumerate([0, 120, 240]):
        x_surf, y_surf, z_surf = transform_coords(xb_sg, yb_sg, zb_sg, angle, H_total)
        ax.plot_surface(x_surf, y_surf, z_surf, color='gold', alpha=0.35, edgecolor='none')
        x_path, y_path, z_path = transform_coords(xb_path, yb_path, zb_path, angle, H_total)
        lbl = 'Turbine Phase (slab)' if idx == 0 else ""
        ax.plot(x_path, y_path, z_path, color='orange', linewidth=1.5, label=lbl)

    ax.set_title("Full Wind Turbine Inspection Path", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude Z (m)")
    
    max_dim = max(R_base * 4, H_total + blade_span)
    ax.set_xlim([-max_dim / 2, max_dim / 2])
    ax.set_ylim([-max_dim / 2, max_dim / 2])
    ax.set_zlim([0, max_dim])
    ax.set_box_aspect([1, 1, 1])

    # legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label['Waterline'] = mpatches.Patch(color='dodgerblue', alpha=0.4)
    by_label['Monopile End'] = mpatches.Patch(color='red', alpha=0.4)
    ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()


