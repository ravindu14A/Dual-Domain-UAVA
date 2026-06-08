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

# turbine blades modelled as rectangular slabs (only top + bottom surfaces scanned)
blade_chord = 3.5   # [m]  blade chord (width across aerofoil)
blade_span  = 110   # [m]  blade length (root to tip)

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
    "gsd": 0.004, "max_blur": 2.0, "h_fov": 47.5,
    "D": 2.0, "line_rate": 330, "integration": 0.003, "h_overlap": 0.05,
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


def _water_timed_waypoints():
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
    elif fmode == "spiral":
        from Test_time_functions import get_velocity_rgb_spiral
        pitch_w = v_frame_w * (1.0 - cw["v_overlap"])
        if water_config["camera_type"] == "RGB":
            v_scan, _ = get_velocity_rgb_spiral(
                v_max, R_base, pitch_w, w_arc_w, cw["gsd"],
                cw["max_blur"], cw["shutter"], cw["h_overlap"], cw["fps"])
        else:
            v_scan = v_max
        v_horiz  = v_scan
        num_revs = H_water / pitch_w
        n_pts    = max(int(num_revs * 100), 100)
        za = np.linspace(0.0, -H_water, n_pts)
        ta = (za / (-H_water)) * num_revs * 2.0 * np.pi
        ra = np.full(n_pts, r_inspect)
        n_strips = 0
    else:
        raise ValueError(f"Unknown underwater flight_mode: '{fmode}'")

    xyz   = np.column_stack([ra * np.cos(ta), ra * np.sin(ta), za])
    dists = np.array([np.linalg.norm(xyz[i+1] - xyz[i]) for i in range(len(za)-1)])
    speed_arr   = _classify_and_profile(xyz, dists, v_scan, v_horiz, WATER_ACCEL_MAX)
    vr, vth, vz = _project_velocities(xyz, ra, ta, dists, speed_arr)
    times       = _timestamps(dists, speed_arr)
    wp = np.column_stack([times, ra, ta, za, vr, vth, vz])
    print(f"[UW traj] {fmode}, {n_strips if fmode == 'lawnmower' else 0} strips, "
          f"{len(za)} pts, duration={wp[-1,0]:.0f} s, "
          f"v_scan={v_scan:.3f} m/s, v_horiz={v_horiz:.3f} m/s")
    return wp, v_scan, v_horiz


def _turbine_timed_waypoints(blade_number=1):
    """Returns (M,7) [t, r, theta, z, vr, vth, vz] in SC sim frame and (v_scan, v_horiz).
    Blade modelled as a rectangular slab (blade_chord × blade_span). Scans top then
    bottom surface with a lawnmower pattern along the span direction.
    blade_number: 1, 2, or 3 — evenly spaced 120 deg apart (Y-axis rotation)."""
    ct    = cameras[turbine_config["camera_type"]]
    D     = ct["D"];  v_max = float(turbine_config["v_max"])

    # Camera footprint across chord (h_fov spans the chord direction)
    strip_w    = 2.0 * D * np.tan(np.radians(ct["h_fov"] / 2))
    strip_step = strip_w * (1.0 - ct["h_overlap"])
    n_strips   = int(np.ceil(blade_chord / strip_step))

    if turbine_config["camera_type"] == "RGB":
        v_frame_t = get_v_frame(D, ct["v_fov"])
        v_scan, _ = get_velocity_rgb_lawnmower(v_max, ct["gsd"], ct["max_blur"],
                                                ct["shutter"], v_frame_t, ct["v_overlap"], ct["fps"])
    elif turbine_config["camera_type"] == "EVENT":
        v_scan, _ = get_velocity_event(v_max)
    elif turbine_config["camera_type"] == "HYPERSPECTRAL":
        v_scan, _ = get_velocity_hyper_lawnmower(v_max, ct["gsd"], ct["line_rate"],
                                                  ct["integration"], ct["max_blur"])
    else:
        v_scan = v_max
    v_horiz = float(turbine_config.get("v_horiz", v_max))

    # Strip chord positions
    x_strips = np.linspace(-blade_chord / 2, blade_chord / 2, n_strips)
    N_SPAN   = 30   # points per span segment

    pts = []  # blade-local Cartesian [xb, yb, zb]

    # ── Top surface (drone at y = +D) ─────────────────────────────────────────
    for i, xb in enumerate(x_strips):
        z0 = 0.0          if i % 2 == 0 else float(blade_span)
        z1 = float(blade_span) if i % 2 == 0 else 0.0
        for z in np.linspace(z0, z1, N_SPAN):
            pts.append([xb, D, z])
        if i < n_strips - 1:
            z_here = pts[-1][2]
            for x2 in np.linspace(xb, x_strips[i + 1], 6)[1:]:
                pts.append([x2, D, z_here])

    # ── Transition at blade TIP: fly to z=blade_span, sweep y from +D to −D ─
    x_end = pts[-1][0];  z_end = pts[-1][2]
    if abs(z_end - float(blade_span)) > 0.01:
        for z in np.linspace(z_end, float(blade_span), 15)[1:]:
            pts.append([x_end, D, z])
    for y in np.linspace(D, -D, 10)[1:]:
        pts.append([x_end, y, float(blade_span)])

    # ── Bottom surface (drone at y = −D): starts at tip, scans toward root ──
    x_strips_bot = x_strips[::-1]
    for i, xb in enumerate(x_strips_bot):
        z0 = float(blade_span) if i % 2 == 0 else 0.0
        z1 = 0.0          if i % 2 == 0 else float(blade_span)
        for z in np.linspace(z0, z1, N_SPAN):
            pts.append([xb, -D, z])
        if i < n_strips - 1:
            z_here = pts[-1][2]
            for x2 in np.linspace(xb, x_strips_bot[i + 1], 6)[1:]:
                pts.append([x2, -D, z_here])

    pts   = np.array(pts)
    xb_arr, yb_arr, zb_arr = pts[:, 0], pts[:, 1], pts[:, 2]
    N     = len(pts)

    dists     = np.array([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(N - 1)])
    speed_arr = _classify_and_profile(pts, dists, v_scan, v_horiz, AERIAL_ACCEL_MAX)
    times     = _timestamps(dists, speed_arr)

    # Cartesian velocities from path tangent
    vx_b = np.zeros(N);  vy_b = np.zeros(N);  vz_b = np.zeros(N)
    for i in range(N - 1):
        if dists[i] > 1e-10:
            tan = (pts[i + 1] - pts[i]) / dists[i]
            vx_b[i], vy_b[i], vz_b[i] = tan * speed_arr[i]
    vx_b[-1], vy_b[-1], vz_b[-1] = vx_b[-2], vy_b[-2], vz_b[-2]

    # Rotate to global frame (around Y-axis by blade azimuth)
    alpha   = np.radians((blade_number - 1) * 120.0)
    ca, sa  = np.cos(alpha), np.sin(alpha)
    H_tower = float(H_air_cyl + H_air_cone)

    x_g = xb_arr * ca + zb_arr * sa
    y_g = yb_arr
    z_g = -xb_arr * sa + zb_arr * ca + H_tower

    vx_g = vx_b * ca + vz_b * sa
    vy_g = vy_b
    vz_g = -vx_b * sa + vz_b * ca

    r_g   = np.maximum(np.sqrt(x_g**2 + y_g**2), 1e-6)
    th_g  = np.unwrap(np.arctan2(y_g, x_g))
    vr_g  = (x_g * vx_g + y_g * vy_g) / r_g
    vth_g = (x_g * vy_g - y_g * vx_g) / r_g**2

    wp = np.column_stack([times, r_g, th_g, z_g, vr_g, vth_g, vz_g])
    print(f"[Turbine traj] blade={blade_number} ({(blade_number-1)*120}°), slab "
          f"{blade_chord}m chord × {blade_span}m span, "
          f"{n_strips} strips × 2 surfaces, {N} pts, duration={wp[-1,0]:.0f} s, "
          f"v_scan={v_scan:.3f} m/s, v_horiz={v_horiz:.3f} m/s")
    return wp, v_scan, v_horiz



def write_matlab_traj(fname, mat, header_vars, label, extra_footer=''):
    """Write a trajectory matrix to a MATLAB .m file.

    header_vars: list of (name, value, comment) tuples written before the matrix.
    extra_footer: string appended after the matrix closing bracket.
    Columns: r, theta, z, vr, vth, vz, ax, ay, az, ref_yaw
    """
    import os
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    M = mat.shape[0]
    with open(fname, 'w') as f:
        f.write(f"%% Trajectory: {label}\n")
        f.write(f"% Columns: r [m], theta [rad], z [m], vr [m/s], vth [rad/s], vz [m/s], "
                f"ax [m/s^2], ay [m/s^2], az [m/s^2], ref_yaw [rad]\n\n")
        for name, val, comment in header_vars:
            f.write(f"{name:<20} = {val:.6f};   % {comment}\n")
        f.write(f"\ntrajectory = [ ...\n")
        for i, row in enumerate(mat):
            vals = "  " + "  ".join(f"{v:15.8f}" for v in row)
            f.write(vals + ("; ..." if i < M - 1 else "];") + "\n")
        if extra_footer:
            f.write(extra_footer)


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

    # Slab model: strips along span, step across chord; two surfaces per blade
    strip_w_t    = 2.0 * ct["D"] * math.tan(math.radians(ct["h_fov"] / 2))
    strip_step_t = strip_w_t * (1.0 - ct["h_overlap"])
    nstrips_t    = math.ceil(blade_chord / strip_step_t)

    if turbine_config["camera_type"] == "RGB":
        v_frame_t = get_v_frame(ct["D"], ct["v_fov"])
        v_vert, limit = get_velocity_rgb_lawnmower(turbine_config["v_max"], ct["gsd"], ct["max_blur"], ct["shutter"], v_frame_t, ct["v_overlap"], ct["fps"])
    elif turbine_config["camera_type"] == "EVENT":
        v_vert, limit = get_velocity_event(turbine_config["v_max"])
    elif turbine_config["camera_type"] == "HYPERSPECTRAL":
        v_vert, limit = get_velocity_hyper_lawnmower(turbine_config["v_max"], ct["gsd"], ct["line_rate"], ct["integration"], ct["max_blur"])
    else:
        v_vert, limit = float(turbine_config["v_max"]), "v_max"

    dist_span_t  = nstrips_t * blade_span                    # scan travel per surface
    dist_chord_t = (nstrips_t - 1) * strip_step_t            # chord steps per surface
    # 2 surfaces × 3 blades
    t_turbine = 3 * 2 * time_lawnmower(dist_span_t, dist_chord_t, v_vert, turbine_config["v_horiz"])
    turbine_vel_str  = f"v_scan={v_vert:.3f} m/s [{limit}], v_horiz={turbine_config['v_horiz']:.3f} m/s"
    turbine_dist_str = (f"span={3*2*dist_span_t:.1f}m, chord_steps={3*2*dist_chord_t:.1f}m, "
                        f"strip_w={strip_w_t:.2f}m, n_strips={nstrips_t}×2surf×3blades")

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
    plot_inspection_route(R_base, R_top, H_water, H_air_cyl, H_air_cone, blade_span, blade_chord, water_config, air_config, turbine_config, cameras)

