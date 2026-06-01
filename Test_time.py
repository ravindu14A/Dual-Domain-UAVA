# drone inspection routing - main interface
import math
import numpy as np
import matplotlib.pyplot as plt

# core math from the functions file
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

# --- Tower & turbine geometry ---
R_base = 4        # monopile radius
R_top  = 2.85     # nacelle interface radius at cone top

H_water    = 60   # underwater section
H_air_cyl  = 30   # above-water cylinder (before cone)
H_air_cone = 135  # tapered cone height

# turbine blades modelled as 3 cylinders
R_blade = 3.5
H_blade = 110

# --- Camera configs ---
# h_fov, v_fov: full-angle FOV in degrees; D: standoff from surface (m)

rgb_camera = {
    "gsd": 0.003, "max_blur": 2.0, "shutter": 1/(2*90),
    "h_fov": 106.62, "v_fov": 71.08, "D":2 ,
    "h_overlap": 0.1, "v_overlap": 0.1, "fps": 90,
}
event_camera = {
    "h_fov": 60.0, "v_fov": 45.0, "D": 2.0,
    "h_overlap": 0.2, "v_overlap": 0.2,
}
hyperspectral_camera = {
    "gsd": 0.004, "max_blur": 2.0, "h_fov": 38.0,
    "D": 3.0, "line_rate": 330, "aintegration": 0.003, "h_overlap": 0.2,
}  # [Specimen AFX10]

cameras = {"RGB": rgb_camera, "EVENT": event_camera, "HYPERSPECTRAL": hyperspectral_camera}

# --- Phase configs ---

# water
water_config = {
    "camera_type": "RGB",
    "flight_mode": "lawnmower",
    "v_max":   1.5,    # [m/s] max scan speed (camera may limit this further)
    "v_horiz": 0.2,    # [m/s] horizontal step speed
}

# air (imported by quadcopterSC.py)
air_config = {
    "camera_type": "RGB",
    "flight_mode": "lawnmower",
    "v_max":   5,
    "v_horiz": 0.2,
}

# turbine
turbine_config = {
    "camera_type": "RGB",
    "flight_mode": "lawnmower",
    "v_max":   5,
    "v_horiz": 0.2,
}

transition_penalty_seconds = 45.0

# ─────────────────────────────────────────────────────────────────────────────
# TRAJECTORY BUILDERS + MATLAB EXPORT
# Run this file directly to regenerate trajectory .mat files.
# ─────────────────────────────────────────────────────────────────────────────

AERIAL_ACCEL_MAX = 10.0   # [m/s²] trapezoidal ramp limit — aerial phase
WATER_ACCEL_MAX  = 2.0    # [m/s²] trapezoidal ramp limit — underwater phase

from Test_time_functions import get_lawnmower_coords, get_spiral_coords

# hover rotor speed — same formula and constants as Params.omega_h in quadcopterSC.py
from geometry import total_mass as _geo_mass
_kT = 0.039 * 1.225 * 0.80**4 / (4 * math.pi**2)   # CT_prop * rho * D^4 / (4pi^2)
omega_h = math.sqrt(_geo_mass * 9.81 / (4 * _kT))

def _arc_lengths(pts):
    s = np.zeros(len(pts))
    for i in range(1, len(pts)):
        s[i] = s[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    return s

def _trap_speeds(s_arr, v_cruise, a):
    L = float(s_arr[-1])
    if L < 1e-10:
        return np.zeros(len(s_arr)), 0.0
    d_ramp = v_cruise**2 / (2.0 * a)
    v_peak = v_cruise if L >= 2.0 * d_ramp else math.sqrt(a * L)
    d_ramp = v_peak**2 / (2.0 * a)
    speeds = np.zeros(len(s_arr))
    for i, s in enumerate(s_arr):
        if s <= d_ramp:
            speeds[i] = max(math.sqrt(2.0 * a * s), 1e-3)
        elif s <= L - d_ramp:
            speeds[i] = v_peak
        else:
            speeds[i] = max(math.sqrt(2.0 * a * (L - s)), 1e-3)
    return speeds, v_peak

def _classify_and_profile(xyz, dists, v_vert, v_horiz, a_max):
    """Classify path gaps as vertical (0) or horizontal (1), apply trap_speeds per segment."""
    N_pts = len(xyz)
    gap_type = np.full(N_pts - 1, -1, dtype=int)
    for i in range(N_pts - 1):
        if dists[i] > 1e-10:
            vf = abs(xyz[i+1, 2] - xyz[i, 2]) / dists[i]
            gap_type[i] = 0 if vf > 0.7 else 1
    speed_arr = np.zeros(N_pts)
    i = 0
    while i < N_pts - 1:
        while i < N_pts - 1 and gap_type[i] == -1:
            i += 1
        if i >= N_pts - 1:
            break
        seg_t = gap_type[i]
        v_cru = v_vert if seg_t == 0 else v_horiz
        j = i
        while j < N_pts - 1 and gap_type[j] == seg_t:
            j += 1
        s_arr = _arc_lengths(xyz[i:j+1])
        speeds, _ = _trap_speeds(s_arr, v_cru, a_max)
        speed_arr[i:j+1] = speeds
        i = j
    return speed_arr

def _project_velocities(xyz, ra, ta, dists, speed_arr):
    """Project scalar speed along path tangent into cylindrical [vr, vth, vz]."""
    N_pts = len(ra)
    vr_arr = np.zeros(N_pts); vth_arr = np.zeros(N_pts); vz_arr = np.zeros(N_pts)
    for i in range(N_pts - 1):
        if dists[i] < 1e-10:
            continue
        dv   = xyz[i+1] - xyz[i]
        vx_i = (dv[0] / dists[i]) * speed_arr[i]
        vy_i = (dv[1] / dists[i]) * speed_arr[i]
        vz_i = (dv[2] / dists[i]) * speed_arr[i]
        r_i  = max(ra[i], 1e-6);  th_i = ta[i]
        vr_arr[i]  =  vx_i * np.cos(th_i) + vy_i * np.sin(th_i)
        vth_arr[i] = (-vx_i * np.sin(th_i) + vy_i * np.cos(th_i)) / r_i
        vz_arr[i]  = vz_i
    vr_arr[-1] = vr_arr[-2]; vth_arr[-1] = vth_arr[-2]; vz_arr[-1] = vz_arr[-2]
    return vr_arr, vth_arr, vz_arr

def _timestamps(dists, speed_arr):
    times = [0.0]
    for i in range(1, len(speed_arr)):
        d = dists[i-1]
        if d < 1e-10:
            times.append(times[-1])
        else:
            v_avg = max((speed_arr[i-1] + speed_arr[i]) / 2, 1e-6)
            times.append(times[-1] + d / v_avg)
    return np.array(times)

def _aerial_timed_waypoints():
    """Returns (M,7) [t, r, theta, z_sim, vr, vth, vz] and (v_scan, v_horiz).
    z_sim = 0 at sea level. vth is theta_dot [rad/s]."""
    ca    = cameras[air_config["camera_type"]]
    D     = ca["D"];  fmode = air_config["flight_mode"];  v_max = float(air_config["v_max"])
    v_frame_a = get_v_frame(D, ca["v_fov"]) if "v_fov" in ca else None
    w_arc_a   = get_w_arc(R_base, D, ca["h_fov"], label="tower")
    H_total   = H_water + H_air_cyl + H_air_cone

    def _r(z_abs):   # radius at absolute z (seabed frame)
        if z_abs <= H_water + H_air_cyl:
            return R_base
        return R_base - ((R_base - R_top) / H_air_cone) * (z_abs - (H_water + H_air_cyl))

    if fmode == "lawnmower":
        if air_config["camera_type"] == "RGB":
            v_scan, _ = get_velocity_rgb_lawnmower(v_max, ca["gsd"], ca["max_blur"], ca["shutter"], v_frame_a, ca["v_overlap"], ca["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_scan, _ = get_velocity_event(v_max)
        elif air_config["camera_type"] == "HYPERSPECTRAL":
            v_scan, _ = get_velocity_hyper_lawnmower(v_max, ca["gsd"], ca["line_rate"], ca["integration"], ca["max_blur"])
        else:
            v_scan = v_max
        v_vert  = v_scan
        v_horiz = float(air_config.get("v_horiz", v_max))
        w_arc_step = w_arc_a * (1.0 - ca["h_overlap"])
        za, ta = get_lawnmower_coords(H_water, H_total, R_base, w_arc_step)
    elif fmode == "spiral":
        pitch = v_frame_a * (1.0 - ca["v_overlap"]) if v_frame_a else w_arc_a * (1 - ca["h_overlap"])
        if air_config["camera_type"] == "RGB":
            v_scan, _ = get_velocity_rgb_spiral(v_max, R_base, pitch, w_arc_a, ca["gsd"], ca["max_blur"], ca["shutter"], ca["h_overlap"], ca["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_scan, _ = get_velocity_event_spiral(v_max)
        else:
            v_scan = v_max
        v_vert = v_horiz = v_scan
        za, ta = get_spiral_coords(H_water, H_total, pitch)
    else:
        raise ValueError(f"Unknown flight_mode: '{fmode}'")

    za = np.array(za);  ta = np.array(ta)
    ra    = np.array([_r(z) for z in za]) + D
    z_sim = za - H_water   # sim frame: z=0 at sea level
    xyz   = np.column_stack([ra * np.cos(ta), ra * np.sin(ta), z_sim])
    dists = np.array([np.linalg.norm(xyz[i+1] - xyz[i]) for i in range(len(za)-1)])

    speed_arr          = _classify_and_profile(xyz, dists, v_vert, v_horiz, AERIAL_ACCEL_MAX)
    vr, vth, vz        = _project_velocities(xyz, ra, ta, dists, speed_arr)
    times              = _timestamps(dists, speed_arr)
    return np.column_stack([times, ra, ta, z_sim, vr, vth, vz]), v_scan, v_horiz


def _build_water_waypoints():
    """Returns (M,7) [t, r, theta, z, vr, vth, vz] and (v_scan, v_horiz).
    z=0 at surface, negative downward. vth is theta_dot [rad/s]."""
    cw      = cameras[water_config["camera_type"]]
    D       = cw["D"];  v_max = float(water_config["v_max"])
    v_horiz = float(water_config.get("v_horiz", v_max));  fmode = water_config["flight_mode"]
    v_frame_w = get_v_frame(D, cw["v_fov"]) if "v_fov" in cw else None
    w_arc_w   = get_w_arc(R_base, D, cw["h_fov"], label="monopile UW")
    r_inspect = R_base + D

    if fmode == "lawnmower":
        if water_config["camera_type"] == "RGB":
            v_scan, _ = get_velocity_rgb_lawnmower(v_max, cw["gsd"], cw["max_blur"], cw["shutter"], v_frame_w, cw["v_overlap"], cw["fps"])
        elif water_config["camera_type"] == "HYPERSPECTRAL":
            v_scan, _ = get_velocity_hyper_lawnmower(v_max, cw["gsd"], cw["line_rate"], cw["integration"], cw["max_blur"])
        else:
            v_scan = v_max
        strip_w  = w_arc_w * (1.0 - cw["h_overlap"])
        n_strips = int(np.ceil(2.0 * np.pi * R_base / strip_w))
        d_theta  = (2.0 * np.pi) / n_strips
        ra_list, ta_list, za_list = [], [], []
        theta = 0.0
        for s_idx in range(n_strips):
            z_start = 0.0      if (s_idx % 2 == 0) else -H_water
            z_end   = -H_water if (s_idx % 2 == 0) else 0.0
            ra_list.extend([r_inspect] * 30);  ta_list.extend([theta] * 30)
            za_list.extend(np.linspace(z_start, z_end, 30))
            if s_idx < n_strips - 1:
                theta_next = theta + d_theta
                ra_list.extend([r_inspect] * 30)
                ta_list.extend(np.linspace(theta, theta_next, 30))
                za_list.extend([z_end] * 30)
                theta = theta_next
        ra = np.array(ra_list);  ta = np.array(ta_list);  za = np.array(za_list)
    else:
        raise ValueError(f"Underwater export only supports lawnmower. Got: '{fmode}'")

    xyz   = np.column_stack([ra * np.cos(ta), ra * np.sin(ta), za])
    dists = np.array([np.linalg.norm(xyz[i+1] - xyz[i]) for i in range(len(za)-1)])
    speed_arr   = _classify_and_profile(xyz, dists, v_scan, v_horiz, WATER_ACCEL_MAX)
    vr, vth, vz = _project_velocities(xyz, ra, ta, dists, speed_arr)
    times       = _timestamps(dists, speed_arr)
    return np.column_stack([times, ra, ta, za, vr, vth, vz]), v_scan, v_horiz


def _waypoints_to_matrix(wp):
    """Convert (M,7) [t,r,theta,z,vr,vth,vz] to (M,9) [r,theta,z,vr,vth,vz,ax,ay,az].
    vth is theta_dot [rad/s]. ax,ay,az are Cartesian feedforward accelerations."""
    t  = wp[:, 0];  r  = wp[:, 1];  th = wp[:, 2];  z  = wp[:, 3]
    vr = wp[:, 4];  vth= wp[:, 5];  vz = wp[:, 6]
    # make timestamps strictly monotonic to avoid divide-by-zero in np.gradient
    # at zero-length junctions where consecutive t values are identical
    t_safe = t.copy()
    for i in range(1, len(t_safe)):
        if t_safe[i] <= t_safe[i-1]:
            t_safe[i] = t_safe[i-1] + 1e-9
    ar  = np.gradient(vr,  t_safe)
    ath = np.gradient(vth, t_safe)
    az  = np.gradient(vz,  t_safe)
    ax  = (ar - r*vth**2) * np.cos(th) - (r*ath + 2*vr*vth) * np.sin(th)
    ay  = (ar - r*vth**2) * np.sin(th) + (r*ath + 2*vr*vth) * np.cos(th)
    return np.column_stack([r, th, z, vr, vth, vz, ax, ay, az])


def export_trajectories(save_dir=r'C:\Users\banda\Documents\MATLAB\UAUV Control\Aerial'):
    """Build aerial and underwater trajectory matrices and save as .m init scripts."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    for label, build_fn, a_max in [
        ("aerial",      _aerial_timed_waypoints, AERIAL_ACCEL_MAX),
        ("underwater",  _build_water_waypoints,  WATER_ACCEL_MAX ),
    ]:
        print(f"Building {label} trajectory...", flush=True)
        wp, v_scan, v_horiz = build_fn()
        mat = _waypoints_to_matrix(wp)

        # append ref_yaw: drone faces inward (theta + pi), wrapped to +-pi
        ref_yaw = np.arctan2(np.sin(wp[:, 2] + np.pi), np.cos(wp[:, 2] + np.pi))
        mat = np.column_stack([mat, ref_yaw])

        # verify 3-D speed magnitude never exceeds v_scan or v_horiz
        speed_3d = np.sqrt(wp[:,4]**2 + (wp[:,1]*wp[:,5])**2 + wp[:,6]**2)
        v_limit  = max(v_scan, v_horiz)
        excess   = float(np.max(speed_3d)) - v_limit
        if excess > 0.01:
            print(f"  WARNING: max speed {np.max(speed_3d):.4f} m/s exceeds limit {v_limit:.3f} m/s by {excess:.4f} m/s")
        else:
            print(f"  Speed OK : max {np.max(speed_3d):.4f} m/s  <=  v_limit {v_limit:.3f} m/s")

        fname = os.path.join(save_dir, f'trajectory_{label}.m')
        M, N_col = mat.shape
        with open(fname, 'w') as f:
            f.write(f"%% Trajectory: {label}\n")
            f.write(f"% Auto-generated by Test_time.py\n")
            f.write(f"% Columns: r [m], theta [rad], z [m], vr [m/s], vth [rad/s], vz [m/s], ax [m/s^2], ay [m/s^2], az [m/s^2], ref_yaw [rad]\n")
            f.write(f"% {M} waypoints  |  duration {wp[-1,0]:.1f} s  |  v_scan {v_scan:.4f} m/s\n\n")

            f.write(f"v_scan_{label}      = {v_scan:.6f};   % [m/s]\n")
            f.write(f"v_horiz_{label}     = {v_horiz:.6f};   % [m/s]\n")
            f.write(f"a_max_{label}       = {a_max:.6f};   % [m/s^2]\n")
            f.write(f"duration_{label}    = {wp[-1,0]:.6f};  % [s]\n")
            f.write(f"n_waypoints_{label} = {M};\n")
            if label == "aerial":
                f.write(f"omega_h             = {omega_h:.6f};   % [rad/s] hover rotor speed\n")
            f.write("\n")

            f.write(f"trajectory = [ ...\n")
            for i, row in enumerate(mat):
                vals = "  " + "  ".join(f"{v:15.8f}" for v in row)
                sep  = "; ..." if i < M - 1 else "];"
                f.write(vals + sep + "\n")

            if label == "aerial":
                f.write(f"\nx0        = zeros(16,1);\n")
                f.write(f"x0(1)     = trajectory(1, 1) * cos(deg2rad(trajectory(1, 2)));\n")
                f.write(f"x0(2)     = trajectory(1, 1) * sin(deg2rad(trajectory(1, 2)));\n")
                f.write(f"x0(3)     = trajectory(1, 3);\n")
                f.write(f"x0(5)     = trajectory(1, 2);\n")
                f.write(f"x0(13:16) = omega_h;\n")
                f.write(f"x0(6)     = trajectory(1, 10);\n")

        print(f"  Saved  : {fname}  ({M} waypoints, {wp[-1,0]:.0f} s)")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # phase 1: underwater
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
    # phase 2: air
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
    # phase 3: turbine (3 blades)
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
    # results
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
    # visualise flight path
    # ---------------------------------------------------------
    print("\nGenerating 3D interactive plot...")
    plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, H_blade, R_blade, water_config, air_config, turbine_config, cameras)

    # ---------------------------------------------------------
    # export trajectory .mat files for MATLAB
    # ---------------------------------------------------------
    print("\nExporting trajectory matrices to MATLAB...")
    export_trajectories()