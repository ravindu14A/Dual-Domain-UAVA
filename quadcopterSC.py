"""
quadcopterSC.py  --  Quadcopter Aerial Phase: Stability & Control
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

6-DOF nonlinear model + cascaded PID + wind gust disturbance analysis.

STATE (16):  [x  y  z | phi  theta  psi | xd  yd  zd | p  q  r | w1  w2  w3  w4]
              pos(3)    euler(3)           vel(3)        ang_rate(3) rotors(4)

MOTOR LAYOUT (X config, ENU, body x=forward, y=left, z=up):
  1=front-left  2=front-right  3=rear-left  4=rear-right
  Motors 1,3: CCW (+z reaction)  |  2,4: CW (-z reaction)
  Tower faces between motors 1 and 2 (forward face of X)

EULER CONVENTION: ZYX
  positive theta = nose tilted DOWN (forward tilt -> +x force)
  positive phi   = right side UP    (tilt toward +y -> +y force when phi<0)

Dependencies:  numpy  scipy  matplotlib  control
    pip install numpy scipy matplotlib control
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
import control
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from Test_time import air_config, cameras, R_base, R_top, H_air_cyl, H_air_cone, _aerial_timed_waypoints
from geometry import total_mass as _geo_mass, Ixx as _geo_Ixx, Iyy as _geo_Iyy, Izz as _geo_Izz, L_arm as _geo_Larm, L_box as _geo_Lbox, W_box as _geo_Wbox, H_box as _geo_Hbox

# trajectory helpers -- shared by custom and test_time_air modes

def arc_lengths(pts):
    s = np.zeros(len(pts))
    for i in range(1, len(pts)):
        s[i] = s[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    return s

def trap_speeds(s_arr, v_cruise, a):
    L      = s_arr[-1]
    d_ramp = v_cruise**2 / (2 * a)
    if L >= 2 * d_ramp:
        v_peak = v_cruise
    else:
        v_peak = np.sqrt(a * L)
        d_ramp = L / 2
    speeds = np.zeros(len(s_arr))
    for i, s in enumerate(s_arr):
        if s <= d_ramp:
            speeds[i] = max(np.sqrt(2 * a * s), 1e-3)
        elif s <= L - d_ramp:
            speeds[i] = v_peak
        else:
            speeds[i] = max(np.sqrt(2 * a * (L - s)), 1e-3)
    return speeds, v_peak

def vel_vecs(pts, speeds):
    vels = np.zeros_like(pts)
    for i in range(len(pts)):
        dv = (pts[i+1] - pts[i]) if i < len(pts)-1 else (pts[i] - pts[i-1])
        dist = np.linalg.norm(dv)
        if dist > 1e-10:
            vels[i] = (dv / dist) * speeds[i]
    return vels

def make_times(pts, speeds, t_start):
    times = [t_start]
    for i in range(1, len(pts)):
        dist  = np.linalg.norm(pts[i] - pts[i-1])
        v_avg = (speeds[i-1] + speeds[i]) / 2
        times.append(times[-1] + dist / max(v_avg, 1e-3))
    return np.array(times)


def _lawnmower_turn_waypoints(a_max=2.0, r_corner=0.0):
    """
    Dense time-stamped waypoints for ONE full lawnmower cycle:
      up strip -> horizontal step -> down strip
    with trapezoidal velocity profiles and optional corner rounding.

    r_corner > 0 inserts a quarter-circle arc at each 90 deg turn.
    Both arcs use their own trap speed profile.

    Returns ndarray (M, 7): [t, x, y, z, vx, vy, vz]
    """
    from Test_time_functions import (
        get_w_arc, get_v_frame,
        get_velocity_rgb_lawnmower, get_velocity_event,
        get_velocity_hyper_lawnmower,
    )

    ca    = cameras[air_config["camera_type"]]
    D     = ca["D"]
    v_max = float(air_config["v_max"])

    v_frame_a = get_v_frame(D, ca["v_fov"]) if "v_fov" in ca else None
    w_arc_a   = get_w_arc(R_base, D, ca["h_fov"], label="tower")

    if air_config["camera_type"] == "RGB":
        v_vert, _ = get_velocity_rgb_lawnmower(
            v_max, ca["gsd"], ca["max_blur"], ca["shutter"],
            v_frame_a, ca["v_overlap"], ca["fps"])
    elif air_config["camera_type"] == "EVENT":
        v_vert, _ = get_velocity_event(v_max)
    elif air_config["camera_type"] == "HYPERSPECTRAL":
        v_vert, _ = get_velocity_hyper_lawnmower(
            v_max, ca["gsd"], ca["line_rate"], ca["integration"], ca["max_blur"])
    else:
        v_vert = v_max
    v_horiz = float(air_config.get("v_horiz", v_max))

    w_arc_step = w_arc_a * (1.0 - ca["h_overlap"])
    nstrips    = int(np.ceil((2 * np.pi * float(R_base)) / w_arc_step))
    d_theta    = (2 * np.pi) / nstrips

    # Total above-water height: cylindrical monopile + tapered cone
    H_cyl  = float(H_air_cyl)
    H_cone = float(H_air_cone)
    H_tot  = H_cyl + H_cone
    Rb     = float(R_base)
    Rt     = float(R_top)

    def get_r(z):
        """Drone standoff radius at sim height z (z=0 = sea level)."""
        if z <= H_cyl:
            return Rb + D
        frac = (z - H_cyl) / H_cone
        return Rb + (Rt - Rb) * frac + D

    r_top  = Rt + D   # standoff radius at top of tower
    rc     = float(r_corner)
    n_vert = 60

    def make_corner(P_corner, t1, t2, n_pts=20):
        """Quarter-circle arc of radius rc from direction t1 to direction t2."""
        t1 = np.asarray(t1, float)
        t2 = np.asarray(t2, float)
        P_s = P_corner - rc * t1
        C   = P_s + rc * t2          # arc centre
        e1  = P_s - C                # = -rc * t2
        e2  = P_corner + rc * t2 - C # = +rc * t1
        alphas = np.linspace(0.0, np.pi / 2, n_pts)
        return np.array([C + np.cos(a)*e1 + np.sin(a)*e2 for a in alphas])

    # path coordinates
    # Strips stop/start rc below the top so corners fit exactly
    z_up_top = H_tot - rc if rc > 0 else H_tot
    z_dn_bot = H_tot - rc if rc > 0 else H_tot

    z_up  = np.linspace(0.0,     z_up_top, n_vert)
    r_up  = np.array([get_r(z) for z in z_up])
    xyz_up = np.column_stack([r_up, np.zeros(n_vert), z_up])

    z_dn  = np.linspace(z_dn_bot, 0.0,     n_vert)
    r_dn  = np.array([get_r(z) for z in z_dn])
    xyz_dn = np.column_stack([r_dn * np.cos(d_theta),
                               r_dn * np.sin(d_theta), z_dn])

    # Corner arcs and trimmed horizontal step
    if rc > 0:
        # corner 1: vertical up -> horizontal at theta=0
        P_c1   = np.array([r_top, 0.0, H_tot])
        arc1   = make_corner(P_c1, [0, 0, 1], [0, 1, 0])

        # corner 2: horizontal at theta=d_theta -> vertical down
        P_c2   = np.array([r_top * np.cos(d_theta),
                            r_top * np.sin(d_theta), H_tot])
        arc2   = make_corner(P_c2,
                             [-np.sin(d_theta), np.cos(d_theta), 0],
                             [0, 0, -1])

        # Horizontal step trimmed by the angular offset each corner consumes
        th_off    = rc / r_top          # arc-length rc on circle radius r_top
        th_start  = th_off
        th_end    = d_theta - th_off
        n_step    = max(int((th_end - th_start) / th_off * 4), 2) if th_end > th_start else 2
        theta_arc = np.linspace(th_start, max(th_end, th_start), n_step)
    else:
        arc1 = arc2 = None
        theta_arc = np.linspace(0.0, d_theta, 50)

    xyz_step = np.column_stack([
        r_top * np.cos(theta_arc),
        r_top * np.sin(theta_arc),
        np.full(len(theta_arc), H_tot),
    ])

    # apply trapezoidal speed profiles
    sp_up,   v_pk_v = trap_speeds(arc_lengths(xyz_up),   v_vert,  a_max)
    sp_step, v_pk_s = trap_speeds(arc_lengths(xyz_step), v_horiz, a_max)
    sp_dn,   _      = trap_speeds(arc_lengths(xyz_dn),   v_vert,  a_max)

    vel_up   = vel_vecs(xyz_up,   sp_up)
    vel_step = vel_vecs(xyz_step, sp_step)
    vel_dn   = vel_vecs(xyz_dn,   sp_dn)

    t_hold = 2.0
    t_up   = make_times(xyz_up,   sp_up,   t_hold)

    if rc > 0:
        sp_arc1, _ = trap_speeds(arc_lengths(arc1), v_vert, a_max)
        sp_arc2, _ = trap_speeds(arc_lengths(arc2), v_vert, a_max)
        vel_arc1   = vel_vecs(arc1, sp_arc1)
        vel_arc2   = vel_vecs(arc2, sp_arc2)
        t_arc1 = make_times(arc1,     sp_arc1, t_up[-1])
        t_step = make_times(xyz_step, sp_step, t_arc1[-1])
        t_arc2 = make_times(arc2,     sp_arc2, t_step[-1])
        t_dn   = make_times(xyz_dn,   sp_dn,   t_arc2[-1])
    else:
        t_step = make_times(xyz_step, sp_step, t_up[-1])
        t_dn   = make_times(xyz_dn,   sp_dn,   t_step[-1])

    # hold segments at start and end
    n_h = max(int(t_hold / 0.05), 2)
    n_e = max(int(2.0   / 0.05), 2)

    if rc > 0:
        all_t   = np.concatenate([
            np.linspace(0, t_hold, n_h, endpoint=False),
            t_up, t_arc1, t_step, t_arc2, t_dn,
            np.linspace(t_dn[-1], t_dn[-1] + 2.0, n_e, endpoint=False),
        ])
        all_xyz = np.vstack([
            np.tile(xyz_up[0], (n_h, 1)),
            xyz_up, arc1, xyz_step, arc2, xyz_dn,
            np.tile(xyz_dn[-1], (n_e, 1)),
        ])
        all_vel = np.vstack([
            np.zeros((n_h, 3)),
            vel_up, vel_arc1, vel_step, vel_arc2, vel_dn,
            np.zeros((n_e, 3)),
        ])
    else:
        all_t   = np.concatenate([
            np.linspace(0, t_hold, n_h, endpoint=False),
            t_up, t_step, t_dn,
            np.linspace(t_dn[-1], t_dn[-1] + 2.0, n_e, endpoint=False),
        ])
        all_xyz = np.vstack([
            np.tile(xyz_up[0],  (n_h, 1)),
            xyz_up, xyz_step, xyz_dn,
            np.tile(xyz_dn[-1], (n_e, 1)),
        ])
        all_vel = np.vstack([
            np.zeros((n_h, 3)),
            vel_up, vel_step, vel_dn,
            np.zeros((n_e, 3)),
        ])

    print(f"Lawnmower turn: r_base={Rb+D:.1f} m  r_top={r_top:.1f} m  "
          f"H={H_tot:.0f} m (cyl={H_cyl:.0f} + cone={H_cone:.0f})  "
          f"corner_r={rc:.1f} m  "
          f"v_vert={v_vert:.2f} m/s (peak {v_pk_v:.2f})  "
          f"v_horiz={v_horiz:.1f} m/s (peak {v_pk_s:.2f})  "
          f"a_max={a_max:.1f} m/s2  duration={t_dn[-1]-t_hold:.1f} s")

    # convert Cartesian path to cylindrical coords
    r_arr = np.sqrt(all_xyz[:, 0]**2 + all_xyz[:, 1]**2)
    θ_arr = np.arctan2(all_xyz[:, 1], all_xyz[:, 0])
    z_arr = all_xyz[:, 2]

    vr_arr = ( all_vel[:, 0] * np.cos(θ_arr) + all_vel[:, 1] * np.sin(θ_arr))
    vθ_arr = (-all_vel[:, 0] * np.sin(θ_arr) + all_vel[:, 1] * np.cos(θ_arr)) / np.maximum(r_arr, 1e-6)
    vz_arr = all_vel[:, 2]

    return np.column_stack([all_t, r_arr, θ_arr, z_arr, vr_arr, vθ_arr, vz_arr])


# --- Simulation Config (edit only in this section) ---

# trajectory source
TRAJ_MODE = "test_time_air"
#   "hold"         - hold at origin for the whole simulation
#   "custom"       - waypoints from TRAJ_SEGMENTS below
#   "test_time_air" - full aerial path + velocity profile from Test_time.py

# custom waypoint table - only used when TRAJ_MODE = "custom"
#
# Each row: (t_start, x, y, z)  -- hold at position from t_start
#        or (t_start, x, y, z, vx, vy, vz)  -- move at constant velocity from t_start
#
# (x,y,z) is where the ref IS at t_start, not a target.
# Ref snaps to next row's position at that row's t_start.
#
# simple hold example:
# TRAJ_SEGMENTS = [
#     ( 0.0,  0.0, 0.0, 0.0),
#     ( 3.0,  0.0, 0.0, 2.0),
#     ( 8.0,  3.0, 0.0, 2.0),
# ]
TRAJ_ACCEL_MAX = 10.0   # [m/s²] ramp accel for trap velocity profile
CORNER_RADIUS  = 0   # [m] corner-rounding at strip top/bottom; 0 = sharp
TRAJ_SEGMENTS  = [
    (0,0,0,0),
    (2,0,0,1)
]


#_lawnmower_turn_waypoints(a_max=TRAJ_ACCEL_MAX, r_corner=CORNER_RADIUS)

# event-triggered waypoint advancement
# False: reference advances in lockstep with simulation time (standard)
# True:  reference pointer only advances when |z_actual - z_ref| < EVENT_Z_TOL
#        horizontal/yaw references are held until altitude has caught up
USE_EVENT_TRIG = True
EVENT_Z_TOL    = 0.5   # [m] Z tracking tolerance to release next reference slice

# disturbances
DIST_ENABLED = False
# rows: (t_on, t_off, Fx, Fy, Fz [N inertial], tx, ty, tz [N·m body])
# multiple rows stack on top of each other
DISTURBANCES = [
    (2.0, 2.1,   1.0, 0.0, 0.0,   0.0, 0.0, 0.0),
]

# impulse = instantaneous momentum kick at a single step
# rows: (t_impulse, Jx, Jy, Jz [N·s], Jtx, Jty, Jtz [N·m·s])
# converted to F = J/dt for the one step that contains t_impulse
# Δv = J/m, so 25 N·s on a 25 kg drone = 1 m/s
IMPULSE_ENABLED = False
IMPULSES = [
    #  t [s]   Jx    Jy    Jz    Jtx   Jty   Jtz
    (  2.0,  50.0,  0.0,  0.0,  0.0,  0.0,  0.0),
]

# wind profile - inertial frame [m/s]
# enters drag as F_drag = -kd * (vel_drone - v_wind)
#
# WIND_INTERP: 'linear' (default) or 'cubic' (needs >= 4 rows for true cubic)
#
# example: linear ramp 0->8 m/s over 30s then hold:
#   [(0,0,0,0), (30,8,0,0), (600,8,0,0)]  'linear'
# example: parabolic gust peaking at t=60:
#   [(0,0,0,0), (30,0,0,0), (60,10,0,0), (90,0,0,0)]  'cubic'

WIND_ENABLED = False
WIND_INTERP  = 'linear'
# step gust: 0 before t=1s then 2 m/s in x
WIND_PROFILE = np.array([
    [0.00,  0.0, 0.0, 0.0],
    [0.999, 0.0, 0.0, 0.0],
    [1.000, 2.0, 0.0, 0.0],
])

# step response analysis (custom mode only)
STEP_ANALYSIS_WINDOW  = 20.0   # [s]
SETTLING_THRESHOLD_PCT = 5.0   # [%] band around final value
_step_vis = None   # populated below, used by position plot


# timing
T_BUFFER = 500   # [s] extra time after last event

# output options
PLOT_MODE       = "sim"        # "sim" -> full sim plots, "root_locus" -> pole map only
PLOT_REFERENCE  = True
PLOT_ACTUAL     = True
ENABLE_PLOTS    = PLOT_MODE in ("sim", "root_locus")

# linearisation point for root locus (only used when PLOT_MODE = "root_locus")
LIN_POS   = np.array([0.0, 0.0,  0.0])   # [m]   inertial position
LIN_EULER = np.array([0.0, 0.0,  0.0])   # [rad] phi, theta, psi  (0 = level hover)
LIN_VEL   = np.array([0.0, 0.0,  0.0])   # [m/s] inertial velocity (affects drag linearisation)

# --- Vehicle parameters ---

class Params:
    m     = _geo_mass  # total mass from geometry.py
    Ixx   = _geo_Ixx   # roll inertia
    Iyy   = _geo_Iyy   # pitch inertia
    Izz   = _geo_Izz   # yaw inertia
    Ixz   = 0.0        # xz product of inertia (ZX-plane symmetry assumed)
    l     = _geo_Larm  # arm length CoM to rotor
    g     = 9.81

    # ── Propeller aerodynamics (physical parameterisation) ──────────────
    CT_prop = 0.039      # thrust coefficient (dimensionless, from blade data)
    CQ_prop = 9.8e-4     # torque coefficient (dimensionless, from blade data)
    D_prop  = 0.80       # propeller diameter [m]
    # kT = CT * rho * D^4 / (4π²),  kQ = CQ * rho * D^5 / (4π²)
    kT = CT_prop * 1.225 * D_prop**4 / (4 * np.pi**2)   # ≈ 5.0e-4 N·s²/rad²
    kQ = CQ_prop * 1.225 * D_prop**5 / (4 * np.pi**2)   # ≈ 1.0e-5 N·m·s²/rad²
    tau_m = 0.06      # [s] motor lag
    rho_air = 1.225
    Cd      = np.array([1.28, 1.28, 1.28])       # bluff-body drag coeff [x, y, z]
    A_face  = np.array([
        _geo_Wbox * _geo_Hbox,                 # frontal area in x
        _geo_Lbox * _geo_Hbox,                 # frontal area in y
        _geo_Lbox * _geo_Wbox,                 # frontal area in z
    ])
    Jr      = 6.0e-5   # [kg·m²] rotor spin inertia

    omega_max = 700.0  # [rad/s] rotor speed limit

    @property
    def omega_h(self):
        """Hover rotor speed from 4·kT·ωh² = m·g"""
        return np.sqrt(self.m * self.g / (4 * self.kT))

p = Params()
if TRAJ_MODE == "test_time_air":
    print("=" * 55)
    print("  VEHICLE PHYSICAL PARAMETERS")
    print("=" * 55)
    print(f"  {'m':<6}  Total mass                   {p.m:.4f}   kg")
    print(f"  {'Ixx':<6}  Roll inertia                 {p.Ixx:.4f}   kg·m²")
    print(f"  {'Iyy':<6}  Pitch inertia                {p.Iyy:.4f}   kg·m²")
    print(f"  {'Izz':<6}  Yaw inertia                  {p.Izz:.4f}   kg·m²")
    print(f"  {'Ixz':<6}  XZ product of inertia        {p.Ixz:.4f}   kg·m²")
    print(f"  {'l':<6}  Arm length (CoM→rotor)       {p.l:.4f}   m")
    print(f"  {'CT':<6}  Prop thrust coeff (dim'less)  {p.CT_prop:.4e}  [-]")
    print(f"  {'CQ':<6}  Prop torque coeff (dim'less)  {p.CQ_prop:.4e}  [-]")
    print(f"  {'D':<6}  Prop diameter                {p.D_prop:.4f}   m")
    print(f"  {'kT':<6}  → kT=CT·ρ·D⁴/(4π²)          {p.kT:.4e}  N·s²/rad²")
    print(f"  {'kQ':<6}  → kQ=CQ·ρ·D⁵/(4π²)          {p.kQ:.4e}  N·m·s²/rad²")
    print(f"  {'tau_m':<6}  Motor time constant          {p.tau_m:.4f}   s")
    print(f"  {'Cd':<6}  Drag coeff [x,y,z]           [{p.Cd[0]:.2f}, {p.Cd[1]:.2f}, {p.Cd[2]:.2f}]                   [-]")
    print(f"  {'A_face':<6}  Frontal areas [x,y,z]        [{p.A_face[0]:.4f}, {p.A_face[1]:.4f}, {p.A_face[2]:.4f}]  m²")
    print(f"  {'Jr':<6}  Rotor spin inertia           {p.Jr:.2e}              kg·m²")
    print(f"  {'w_max':<6}  Rotor speed saturation       {p.omega_max:.2f}   rad/s  ({p.omega_max*60/2/np.pi:.0f} RPM)")
    print(f"  {'w_h':<6}  Hover rotational velocity    {p.omega_h:.2f}   rad/s  ({p.omega_h*60/2/np.pi:.0f} RPM)")
    print("=" * 55)

# PID gains per mode (inner att bandwidth >> outer pos bandwidth)
_gains = {
    "hold": dict(
        att_Kp    = np.array([66.9,  132.1, 36.96]),
        att_Ki    = np.array([0.3,   0.3,   0.1  ]),
        att_Kd    = np.array([38.55, 71.1,  49.32]),
        att_i_lim = np.array([10.0,  10.0,  5.0  ]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([0.12,  0.15,  0.33 ]),
        cyl_Ki    = np.array([0.01,  0.01,  0.02 ]),
        cyl_Kd    = np.array([0.55,  0.63,  1.12 ]),
        cyl_i_lim = np.array([5.0,   5.0,   10.0 ]),
    ),
    "custom": dict(
        att_Kp    = np.array([66.9,  132.1, 36.96]),
        att_Ki    = np.array([0.3,   0.3,   0.1  ]),
        att_Kd    = np.array([38.55, 71.1,  49.32]),
        att_i_lim = np.array([10.0,  10.0,  5.0  ]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([0.244, 0.300, 0.656]),
        cyl_Ki    = np.array([0.10,  0.10,  0.05 ]),
        cyl_Kd    = np.array([0.784, 0.904, 1.598]),
        cyl_i_lim = np.array([30.0,  30.0,  10.0 ]),
    ),
    "spiral": dict(
        att_Kp    = np.array([66.9,  132.1, 36.96]),
        att_Ki    = np.array([0.3,   0.3,   0.1  ]),
        att_Kd    = np.array([38.55, 71.1,  49.32]),
        att_i_lim = np.array([10.0,  10.0,  5.0  ]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([0.244, 0.300, 0.656]),
        cyl_Ki    = np.array([0.02,  0.02,  0.05 ]),
        cyl_Kd    = np.array([0.784, 0.904, 1.598]),
        cyl_i_lim = np.array([5.0,   5.0,   10.0 ]),
    ),
    "lawnmower": dict(
        att_Kp    = np.array([61.4,  198.1,  39.14]),
        att_Ki    = np.array([0.3,   0.3,    0.1  ]),
        att_Kd    = np.array([18.39, 45.6,   20.71]),
        att_i_lim = np.array([10.0,  10.0,   5.0  ]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([0.15,  0.20,   2.00 ]),
        cyl_Ki    = np.array([0.02,  0.02,   0.15 ]),
        cyl_Kd    = np.array([1.272, 1.076,  2.50 ]),
        cyl_i_lim = np.array([5.0,   5.0,   10.0  ]),
    ),
    # Derivative-on-measurement gains for event-triggered waypoint mode.
    # Kp_z boosted to compensate for removed velocity feedforward (old D term
    # was implicitly adding Kd*ref_vel as upward thrust).
    # Kd sized for critical damping: Kd ≈ 2*sqrt(Kp) per axis.
    "lawnmower_event": dict(
        att_Kp    = np.array([61.4,  198.1,  39.14]),
        att_Ki    = np.array([0.3,   0.3,    0.1  ]),
        att_Kd    = np.array([18.39, 45.6,   20.71]),
        att_i_lim = np.array([10.0,  10.0,   5.0  ]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([0.25,  0.30,   1.0  ]),
        cyl_Ki    = np.array([0.02,  0.02,   0.10 ]),
        cyl_Kd    = np.array([1.0,   1.1,    2.0  ]),
        cyl_i_lim = np.array([5.0,   5.0,   10.0  ]),
    ),
}

# pick the right gain set
_flight_mode = air_config["flight_mode"] if TRAJ_MODE == "test_time_air" else TRAJ_MODE
_g = _gains.get(_flight_mode, _gains["custom"])

att_Kp    = _g["att_Kp"]
att_Ki    = _g["att_Ki"]
att_Kd    = _g["att_Kd"]
att_i_lim = _g["att_i_lim"]
att_lim   = _g["att_lim"]
# outer loop [r/x, tangential/y, z]
cyl_Kp    = _g["cyl_Kp"]
cyl_Ki    = _g["cyl_Ki"]
cyl_Kd    = _g["cyl_Kd"]
cyl_i_lim = _g["cyl_i_lim"]

print(f"PID gains       : {_flight_mode} set")

# A_mix @ [w1^2 w2^2 w3^2 w4^2] = [T, tau_phi, tau_theta, tau_psi]
# X config, arms at 45 deg, tower between FL(1) and FR(2):
#
#   FL(1)  FR(2)        CCW: 1, 4   CW: 2, 3
#     \   /
#      \ /
#      / \
#     /   \
#   RL(3)  RR(4)

_lx = p.l / np.sqrt(2) * p.kT   # effective moment arm

A_mix = np.array([
    [ p.kT,  p.kT,  p.kT,  p.kT ],
    [ _lx,  -_lx,   _lx,  -_lx  ],   # tau_phi:   [+,−,+,−]
    [-_lx,  -_lx,   _lx,   _lx  ],   # tau_theta: [−,−,+,+]
    [ p.kQ, -p.kQ, -p.kQ,  p.kQ ],   # tau_psi:   [+,−,−,+]  diagonal pairs
])

# --- Trajectory builders ---

def build_custom_traj(segments, t_arr):
    """
    Interpolate position and velocity ref from waypoints.

    segments can be an ndarray (M, 7) [t, x, y, z, vx, vy, vz] or a list of
    tuples (t_start, x, y, z) for holds or (t_start, x, y, z, vx, vy, vz) for moving segments.
    """
    ref_p = np.zeros((3, len(t_arr)))
    ref_v = np.zeros((3, len(t_arr)))

    if isinstance(segments, np.ndarray):
        for ax in range(3):
            ref_p[ax, :] = np.interp(t_arr, segments[:, 0], segments[:, 1 + ax])
            ref_v[ax, :] = np.interp(t_arr, segments[:, 0], segments[:, 4 + ax])
        return ref_p, ref_v

    for i, row in enumerate(segments):
        t0 = row[0]
        x0, y0, z0 = row[1], row[2], row[3]
        vx, vy, vz = (row[4], row[5], row[6]) if len(row) == 7 else (0.0, 0.0, 0.0)

        t1 = segments[i + 1][0] if i < len(segments) - 1 else t_arr[-1] + 1
        mask = (t_arr >= t0) & (t_arr < t1)
        dt = t_arr[mask] - t0

        ref_p[0, mask] = x0 + vx * dt
        ref_p[1, mask] = y0 + vy * dt
        ref_p[2, mask] = z0 + vz * dt
        ref_v[0, mask] = vx
        ref_v[1, mask] = vy
        ref_v[2, mask] = vz

    return ref_p, ref_v


def build_test_time_aerial_traj(t_arr):
    """Interpolate aerial path onto the sim time array.
    Returns ref_pos (3,N), ref_vel (3,N), duration (s), start_xyz (3,).
    """
    wp, _, _ = _aerial_timed_waypoints()   # (M, 7): [t, r, θ, z, ṙ, θ̇, ż]
    ref_p = np.zeros((3, len(t_arr)))
    ref_v = np.zeros((3, len(t_arr)))
    for ax in range(3):
        ref_p[ax, :] = np.interp(t_arr, wp[:, 0], wp[:, 1 + ax])
        ref_v[ax, :] = np.interp(t_arr, wp[:, 0], wp[:, 4 + ax])
    r0, θ0, z0 = wp[0, 1], wp[0, 2], wp[0, 3]
    start_xyz = np.array([r0 * np.cos(θ0), r0 * np.sin(θ0), z0])
    return ref_p, ref_v, float(wp[-1, 0]), start_xyz

# --- disturbance helper ---

def get_disturbance(t_k):
    """Return (Fd [3], taud [3]) from disturbances and impulses at t_k."""
    Fd   = np.zeros(3)
    taud = np.zeros(3)

    if DIST_ENABLED:
        for row in DISTURBANCES:
            if row[0] <= t_k < row[1]:
                Fd   += np.asarray(row[2:5], dtype=float)
                taud += np.asarray(row[5:8], dtype=float)

    # impulses: J [N·s] -> F = J/dt for the one step containing t_impulse
    if IMPULSE_ENABLED:
        for row in IMPULSES:
            t_imp = row[0]
            if t_k <= t_imp < t_k + dt:
                Fd   += np.asarray(row[1:4], dtype=float) / dt
                taud += np.asarray(row[4:7], dtype=float) / dt

    return Fd, taud


def get_wind(t_k):
    """Inertial wind velocity [vx, vy, vz] at t_k."""
    if not WIND_ENABLED:
        return np.zeros(3)
    t_col = WIND_PROFILE[:, 0]
    if WIND_INTERP == 'cubic' and len(WIND_PROFILE) >= 4:
        from scipy.interpolate import interp1d
        f = interp1d(t_col, WIND_PROFILE[:, 1:4], axis=0, kind='cubic',
                     bounds_error=False,
                     fill_value=(WIND_PROFILE[0, 1:4], WIND_PROFILE[-1, 1:4]))
        return f(t_k)
    return np.array([
        np.interp(t_k, t_col, WIND_PROFILE[:, 1]),
        np.interp(t_k, t_col, WIND_PROFILE[:, 2]),
        np.interp(t_k, t_col, WIND_PROFILE[:, 3]),
    ])

# --- Simulation setup ---

dt = 0.05   # [s] timestep

# end time = latest event + buffer
_t_events = [T_BUFFER]
if DIST_ENABLED and DISTURBANCES:
    _t_events.append(max(row[1] for row in DISTURBANCES))

if TRAJ_MODE == "custom" and len(TRAJ_SEGMENTS):
    _t_events.append(TRAJ_SEGMENTS[-1][0])
elif TRAJ_MODE == "test_time_air":
    _wp, _, _ = _aerial_timed_waypoints()
    _t_events.append(float(_wp[-1, 0]))

t_end = max(_t_events) + T_BUFFER
t     = np.arange(0, t_end + dt, dt)
N     = len(t)

# build reference trajectory
x0_override = None   # optionally start at first waypoint

if TRAJ_MODE == "hold":
    ref_pos = np.zeros((3, N))
    ref_vel = np.zeros((3, N))
    print("Trajectory mode : HOLD at origin")

elif TRAJ_MODE == "custom":
    ref_pos, ref_vel = build_custom_traj(TRAJ_SEGMENTS, t)
    # cylindrical (r,theta,z) -- convert first point to Cartesian for init
    if isinstance(TRAJ_SEGMENTS, np.ndarray):
        r0, θ0, z0 = TRAJ_SEGMENTS[0][1], TRAJ_SEGMENTS[0][2], TRAJ_SEGMENTS[0][3]
        x0_override = np.array([r0 * np.cos(θ0), r0 * np.sin(θ0), z0])
    else:
        x0_override = np.array([TRAJ_SEGMENTS[0][1], TRAJ_SEGMENTS[0][2], TRAJ_SEGMENTS[0][3]])
    print(f"Trajectory mode : CUSTOM  ({len(TRAJ_SEGMENTS)} segments, "
          f"t_end={t_end:.1f} s)")

elif TRAJ_MODE == "test_time_air":
    ref_pos, ref_vel, _dur, _start = build_test_time_aerial_traj(t)
    x0_override = _start
    print(f"Trajectory mode : TEST_TIME_AIR  (path duration={_dur:.0f} s, "
          f"t_end={t_end:.1f} s)")
    print(f"  Start waypoint : x={_start[0]:.2f} m  y={_start[1]:.2f} m  "
          f"z={_start[2]:.2f} m")

else:
    raise ValueError(f"Unknown TRAJ_MODE: '{TRAJ_MODE}'")

# cylindrical ref used for custom (ndarray) and test_time_air
_USE_CYL_REF = TRAJ_MODE in ("custom", "test_time_air") and isinstance(
    TRAJ_SEGMENTS if TRAJ_MODE == "custom" else True, (np.ndarray, bool))

# face inward toward tower, so yaw = theta + pi
if _USE_CYL_REF:
    ref_yaw = np.arctan2(np.sin(ref_pos[1, :] + np.pi), np.cos(ref_pos[1, :] + np.pi))
else:
    ref_yaw = np.zeros(N)

# acceleration feedforward (includes centripetal + Coriolis in cylindrical mode)
if _USE_CYL_REF:
    # ref_pos = (r, θ, z),  ref_vel = (ṙ, θ̇, ż)
    r_r  = ref_pos[0]
    θ_r  = ref_pos[1]
    ṙ_r  = ref_vel[0]
    θ̇_r  = ref_vel[1]
    ż_r  = ref_vel[2]
    r̈_r  = np.gradient(ṙ_r,  dt)
    θ̈_r  = np.gradient(θ̇_r, dt)
    z̈_r  = np.gradient(ż_r,  dt)
    ref_acc = np.zeros((3, N))
    ref_acc[0] = (r̈_r - r_r*θ̇_r**2)*np.cos(θ_r) - (r_r*θ̈_r + 2*ṙ_r*θ̇_r)*np.sin(θ_r)
    ref_acc[1] = (r̈_r - r_r*θ̇_r**2)*np.sin(θ_r) + (r_r*θ̈_r + 2*ṙ_r*θ̇_r)*np.cos(θ_r)
    ref_acc[2] = z̈_r
else:
    ref_acc = np.gradient(ref_vel, dt, axis=1)

print(f"Event trig WP   : {'ON' if USE_EVENT_TRIG else 'OFF'}  (Z tol = {EVENT_Z_TOL} m)")
print(f"Disturbances    : {'ON' if DIST_ENABLED else 'OFF'}  "
      f"({len(DISTURBANCES)} row(s) defined)")
print(f"Plots           : {'ON' if ENABLE_PLOTS else 'OFF'}")

# --- physics functions ---

def rot_ZYX(phi, theta, psi):
    """Body-to-inertial rotation matrix, ZYX Euler."""
    Rx = np.array([[1,        0,          0       ],
                   [0,  np.cos(phi), -np.sin(phi) ],
                   [0,  np.sin(phi),  np.cos(phi) ]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0,             1, 0             ],
                   [-np.sin(theta), 0, np.cos(theta) ]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi),  np.cos(psi), 0],
                   [0,            0,            1]])
    return Rz @ Ry @ Rx


def euler_kin(phi, theta):
    """Maps body rates [p,q,r] → Euler rates [φ̇,θ̇,ψ̇]. Singular at θ=±90°."""
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([
        [1,  sp*st/ct,  cp*st/ct],
        [0,  cp,       -sp       ],
        [0,  sp/ct,     cp/ct    ],
    ])


def quad_ode(s, wr_cmd, Fd, taud, v_wind=None):
    """
    Nonlinear 6-DOF quadrotor ODE.
    s       : state vector (16,)
    wr_cmd  : commanded rotor speeds (4,)
    Fd      : disturbance force in inertial frame (3,)
    taud    : disturbance torque in body frame (3,)
    v_wind  : inertial wind velocity (3,) [m/s] — drag uses relative velocity
    returns : ds/dt (16,)
    """
    if v_wind is None:
        v_wind = np.zeros(3)
    euler = s[3:6];  phi, theta, psi = euler
    vel   = s[6:9]
    wb    = s[9:12]
    wr    = s[12:16]

    R = rot_ZYX(phi, theta, psi)

    u_actual  = A_mix @ (wr**2)
    T         = u_actual[0]
    tau_body  = u_actual[1:4] + taud

    Omega_net = wr[0] - wr[1] - wr[2] + wr[3]
    tau_gyro  = p.Jr * Omega_net * np.array([-wb[1], wb[0], 0.0])

    F_thrust  = R @ np.array([0, 0, T])
    F_grav    = np.array([0, 0, -p.m * p.g])
    v_rel  = vel - v_wind
    F_drag = -0.5 * p.rho_air * p.Cd * p.A_face * v_rel * np.abs(v_rel)
    pos_ddot  = (F_thrust + F_grav + F_drag + Fd) / p.m

    I      = np.array([[p.Ixx,  0.0,   p.Ixz],
                       [0.0,    p.Iyy, 0.0  ],
                       [p.Ixz,  0.0,   p.Izz]])
    wb_dot = np.linalg.solve(I, tau_body - tau_gyro - np.cross(wb, I @ wb))

    euler_dot = euler_kin(phi, theta) @ wb
    wr_dot    = (wr_cmd - wr) / p.tau_m

    return np.concatenate([vel, euler_dot, pos_ddot, wb_dot, wr_dot])


def rk4_step(s, wr_cmd, Fd, taud, dt, v_wind=None):
    """Classic RK4 integrator step."""
    if v_wind is None:
        v_wind = np.zeros(3)
    k1 = quad_ode(s,            wr_cmd, Fd, taud, v_wind)
    k2 = quad_ode(s + dt/2*k1, wr_cmd, Fd, taud, v_wind)
    k3 = quad_ode(s + dt/2*k2, wr_cmd, Fd, taud, v_wind)
    k4 = quad_ode(s + dt*k3,   wr_cmd, Fd, taud, v_wind)
    return s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# --- main simulation loop (skipped in root_locus mode) ---

X      = np.zeros((16, N))
X[12:16, 0] = p.omega_h   # all rotors at hover speed

if x0_override is not None:
    X[0:3, 0] = x0_override   # start at first trajectory waypoint
X[5, 0] = ref_yaw[0]           # initialise yaw to match reference — avoids 180° spike at t=0

int_att = np.zeros(3)
int_pos = np.zeros(3)

U_log    = np.zeros((4, N))   # [T, τ_φ, τ_θ, τ_ψ]
Wr_log   = np.zeros((4, N))   # commanded rotor speeds
Ref_log  = np.zeros((3, N))   # event-triggered reference position logged at each step

k_ref = 0   # event-triggered reference pointer (equals k when USE_EVENT_TRIG=False)

_sim_iter = range(N - 1) if PLOT_MODE != "root_locus" else []
if tqdm is not None and PLOT_MODE != "root_locus":
    _sim_iter = tqdm(_sim_iter, desc="Simulating", unit="step",
                     mininterval=5, dynamic_ncols=True)
for k in _sim_iter:
    s     = X[:, k]
    pos   = s[0:3]
    euler = s[3:6];  phi, theta, psi = euler
    vel   = s[6:9]
    wb    = s[9:12]
    wr    = s[12:16]

    # reference pointer: advance only when Z has caught up (event-triggered mode)
    if USE_EVENT_TRIG:
        if abs(pos[2] - ref_pos[2, k_ref]) < EVENT_Z_TOL and k_ref < N - 1:
            k_ref += 1
        kr = k_ref
    else:
        kr = k

    Ref_log[:, k] = ref_pos[:, kr]   # log which reference slice the controller actually sees

    # outer PID + feedforward: position/velocity error + acceleration feedforward
    if _USE_CYL_REF:
        r_m = max(np.sqrt(pos[0]**2 + pos[1]**2), 1e-6)
        θ_m = np.arctan2(pos[1], pos[0])
        cs  = np.cos(θ_m);  sn = np.sin(θ_m)

        e_r = ref_pos[0, kr] - r_m
        e_θ = np.arctan2(np.sin(ref_pos[1, kr] - θ_m),
                         np.cos(ref_pos[1, kr] - θ_m))
        e_z = ref_pos[2, kr] - pos[2]
        e_t = r_m * e_θ   # arc-length tangential error [m]

        ṙ_m  =  vel[0] * cs + vel[1] * sn
        θ̇_m  = (-vel[0] * sn + vel[1] * cs) / r_m
        e_ṙ  = ref_vel[0, kr] - ṙ_m
        e_ṫ  = r_m * (ref_vel[1, kr] - θ̇_m)
        e_ż  = ref_vel[2, kr] - vel[2]

        int_pos = np.clip(int_pos + np.array([e_r, e_t, e_z]) * dt,
                          -cyl_i_lim, cyl_i_lim)

        a_r = cyl_Kp[0]*e_r + cyl_Ki[0]*int_pos[0] + cyl_Kd[0]*e_ṙ
        a_t = cyl_Kp[1]*e_t + cyl_Ki[1]*int_pos[1] + cyl_Kd[1]*e_ṫ
        a_z = cyl_Kp[2]*e_z + cyl_Ki[2]*int_pos[2] + cyl_Kd[2]*e_ż

        # acceleration feedforward: centripetal + Coriolis already in Cartesian
        a_ff = ref_acc[:, kr]
        a_cmd = np.array([a_r * cs - a_t * sn,
                          a_r * sn + a_t * cs,
                          a_z]) + a_ff
    else:
        e_pos = ref_pos[:, kr] - pos
        e_vel = ref_vel[:, kr] - vel
        int_pos = np.clip(int_pos + e_pos * dt, -cyl_i_lim, cyl_i_lim)
        a_cmd = cyl_Kp * e_pos + cyl_Ki * int_pos + cyl_Kd * e_vel + ref_acc[:, kr]

    T_cmd = max(p.m * (a_cmd[2] + p.g), 0.1 * p.m * p.g)

    theta_d = ( a_cmd[0]*np.cos(psi) + a_cmd[1]*np.sin(psi)) * p.m / T_cmd
    phi_d   = ( a_cmd[0]*np.sin(psi) - a_cmd[1]*np.cos(psi)) * p.m / T_cmd
    psi_d   = ref_yaw[kr]

    theta_d = np.clip(theta_d, -att_lim, att_lim)
    phi_d   = np.clip(phi_d,   -att_lim, att_lim)

    # inner PID: attitude error -> body torques
    e_att    = np.array([phi_d, theta_d, psi_d]) - euler
    e_att[2] = np.arctan2(np.sin(e_att[2]), np.cos(e_att[2]))   # yaw wrap ±π
    int_att  = np.clip(int_att + e_att * dt, -att_i_lim, att_i_lim)

    # Derivative on measured body rate — avoids derivative kick on ref step
    tau_cmd = att_Kp * e_att + att_Ki * int_att - att_Kd * wb

    u_virt      = np.array([T_cmd, tau_cmd[0], tau_cmd[1], tau_cmd[2]])
    U_log[:, k] = u_virt

    # mixing: virtual inputs -> rotor speed commands
    wr_sq_cmd = np.linalg.solve(A_mix, u_virt)
    wr_sq_cmd = np.maximum(wr_sq_cmd, 0.0)
    wr_cmd    = np.minimum(np.sqrt(wr_sq_cmd), p.omega_max)
    Wr_log[:, k] = wr_cmd

    Fd, taud = get_disturbance(t[k])
    v_wind_k = get_wind(t[k])
    X[:, k+1] = rk4_step(s, wr_cmd, Fd, taud, dt, v_wind_k)

U_log[:, -1] = U_log[:, -2]

# numerical linearisation

_s0 = np.zeros(16)
_s0[0:3]   = LIN_POS
_s0[3:6]   = LIN_EULER
_s0[6:9]   = LIN_VEL
_s0[12:16] = p.omega_h
_u0 = np.full(4, p.omega_h)
_F0 = np.zeros(3);   _td0 = np.zeros(3)        # no disturbances
_eps = 1e-5
ns, nu = 16, 4

A_lin = np.zeros((ns, ns))
B_lin = np.zeros((ns, nu))

for i in range(ns):
    sp, sm = _s0.copy(), _s0.copy()
    sp[i] += _eps;  sm[i] -= _eps
    A_lin[:, i] = (quad_ode(sp, _u0, _F0, _td0) -
                   quad_ode(sm, _u0, _F0, _td0)) / (2 * _eps)

for j in range(nu):
    up, um = _u0.copy(), _u0.copy()
    up[j] += _eps;  um[j] -= _eps
    B_lin[:, j] = (quad_ode(_s0, up, _F0, _td0) -
                   quad_ode(_s0, um, _F0, _td0)) / (2 * _eps)

C_lin     = np.eye(ns)
D_lin     = np.zeros((ns, nu))
sys_hover = control.ss(A_lin, B_lin, C_lin, D_lin)

print(f"\nLinearised system: {ns} states, {nu} inputs  (full model incl. motor lag)")
print(f"Open-loop poles (hover):\n{np.sort_complex(np.linalg.eigvals(A_lin))}")

def build_K_cl(pKp, pKi, pKd, aKp, aKi, aKd):
    """Returns K_pid (4x16) mapping full state to rotor speed commands. Ki ignored (no integrator states)."""
    g  = p.g;   m  = p.m;   wh = p.omega_h
    # virtual-input gain (4x16): u_virt = K_virt @ state
    Kv = np.zeros((4, 16))
    # thrust from z, vz
    Kv[0, 2]  = -m * pKp[2]
    Kv[0, 8]  = -m * pKd[2]
    # tau_phi from y -> phi_d then phi, p
    Kv[1, 1]  =  aKp[0] * pKp[1] / g
    Kv[1, 7]  =  aKp[0] * pKd[1] / g
    Kv[1, 3]  = -aKp[0]
    Kv[1, 9]  = -aKd[0]
    # tau_theta from x -> theta_d then theta, q
    Kv[2, 0]  = -aKp[1] * pKp[0] / g
    Kv[2, 6]  = -aKp[1] * pKd[0] / g
    Kv[2, 4]  = -aKp[1]
    Kv[2, 10] = -aKd[1]
    # tau_psi from psi, r
    Kv[3, 5]  = -aKp[2]
    Kv[3, 11] = -aKd[2]
    return (1.0 / (2.0 * wh)) * np.linalg.inv(A_mix) @ Kv

# convert cylindrical ref to Cartesian for plots
if _USE_CYL_REF:
    _r  = ref_pos[0];  _θ = ref_pos[1]
    _ṙ  = ref_vel[0];  _θ̇ = ref_vel[1]
    ref_pos_cart = np.array([_r * np.cos(_θ),
                              _r * np.sin(_θ),
                              ref_pos[2]])
    ref_vel_cart = np.array([_ṙ * np.cos(_θ) - _r * _θ̇ * np.sin(_θ),
                              _ṙ * np.sin(_θ) + _r * _θ̇ * np.cos(_θ),
                              ref_vel[2]])
else:
    ref_pos_cart = ref_pos
    ref_vel_cart = ref_vel

# --- analysis output ---

if PLOT_MODE == "sim" and DIST_ENABLED and DISTURBANCES:
    tols = [0.15, 0.05, 0.10]   # per-axis settling tolerance [m]
    for row in DISTURBANCES:
        t_on, t_off = row[0], row[1]
        F_str = f"F=({row[2]:.1f},{row[3]:.1f},{row[4]:.1f}) N"
        print(f"\n{'═'*48}")
        print(f"GUST  {t_on}–{t_off} s  {F_str}  (5% tolerance)")
        print(f"{'═'*48}")
        for ax, lbl in enumerate('xyz'):
            post = t > t_off
            err  = np.abs(X[ax, :] - ref_pos_cart[ax, :])
            idx  = np.where(post & (err < tols[ax]))[0]
            if len(idx):
                print(f"  {lbl}-axis : {t[idx[0]] - t_off:.2f} s after gust ends")
            else:
                print(f"  {lbl}-axis : did not settle within window")

        gm = (t >= t_on) & (t <= t_off)
        if gm.any():
            print("\nPEAK VIRTUAL INPUTS DURING GUST")
            print(f"  Thrust       : {U_log[0,gm].max():6.2f} N     (hover = {p.m*p.g:.2f} N)")
            print(f"  Roll torque  : {np.abs(U_log[1,gm]).max():6.4f} N·m")
            print(f"  Pitch torque : {np.abs(U_log[2,gm]).max():6.4f} N·m")
            print(f"  Yaw torque   : {np.abs(U_log[3,gm]).max():6.4f} N·m")

            dev = np.abs(X[0:3, gm] - ref_pos_cart[:, gm]).max(axis=1)
            print(f"\nPEAK POSITION DEVIATION DURING GUST")
            print(f"  dx={dev[0]:.3f} m  dy={dev[1]:.3f} m  dz={dev[2]:.3f} m")


if PLOT_MODE == "sim" and TRAJ_MODE == "custom" and not _USE_CYL_REF:
    # Detect first step: find timestep where reference position changes
    _rp = ref_pos_cart   # (3, N)
    _step_k = None
    for _k in range(1, N):
        if np.linalg.norm(_rp[:, _k] - _rp[:, _k-1]) > 1e-6:
            _step_k = _k
            break

    if _step_k is not None:
        _t_step  = t[_step_k]
        _t_end_w = _t_step + STEP_ANALYSIS_WINDOW
        _win     = (t >= _t_step) & (t <= _t_end_w)
        _t_w     = t[_win]
        _x_w     = X[0:3, _win]      # actual position in window
        _r_w     = _rp[:, _win]      # reference in window

        _step_mag  = _rp[:, _step_k] - _rp[:, _step_k - 1]   # signed step size per axis
        _final_ref = _rp[:, -1]                                 # steady-state reference
        _step_vis  = [None, None, None]

        print(f"\n{'═'*60}")
        print(f"  STEP RESPONSE ANALYSIS  (window = {STEP_ANALYSIS_WINDOW:.0f} s after step)")
        print(f"  Step at t = {_t_step:.2f} s  |  settling band = ±{SETTLING_THRESHOLD_PCT:.1f}%")
        print(f"{'═'*60}")
        print(f"  {'Axis':<6} {'Step':>8} {'Rise time':>12} {'Overshoot':>12} {'Settling':>12}")
        print(f"  {'─'*56}")

        for _i, _ax in enumerate(['x', 'y', 'z']):
            _mag = _step_mag[_i]
            if abs(_mag) < 1e-6:
                print(f"  {_ax:<6}  (no step on this axis)")
                continue

            _act  = _x_w[_i]               # actual trajectory in window
            _ref  = _r_w[_i]               # reference in window
            _err  = _act - _final_ref[_i]  # error relative to final setpoint
            _band = abs(_mag) * SETTLING_THRESHOLD_PCT / 100.0

            # rise time: 10% to 90% of step
            _10 = 0.10 * _mag;  _90 = 0.90 * _mag
            _delta = _act - _act[0]        # displacement from initial
            _rise_idx10 = np.where(np.abs(_delta) >= np.abs(_10))[0]
            _rise_idx90 = np.where(np.abs(_delta) >= np.abs(_90))[0]
            _rise = ((_t_w[_rise_idx90[0]] - _t_w[_rise_idx10[0]])
                     if len(_rise_idx10) and len(_rise_idx90) else float('nan'))

            # Overshoot: max exceedance beyond final value in step direction
            if _mag > 0:
                _peak = np.max(_act)
            else:
                _peak = np.min(_act)
            _overshoot_pct = 100.0 * (_peak - _final_ref[_i]) / abs(_mag)

            # Settling time: interpolated crossing where |error| = band (last entry into band)
            _unsettled = np.where(np.abs(_err) > _band)[0]
            if len(_unsettled):
                _j = _unsettled[-1]
                if _j + 1 < len(_t_w):
                    _e0, _e1 = np.abs(_err[_j]), np.abs(_err[_j + 1])
                    _frac = (_band - _e0) / (_e1 - _e0) if (_e1 - _e0) != 0 else 0.0
                    _t_settle_abs = float(_t_w[_j]) + _frac * float(_t_w[_j + 1] - _t_w[_j])
                else:
                    _t_settle_abs = float(_t_w[_j])
                _settle = _t_settle_abs - _t_step
            else:
                _settle = 0.0
                _t_settle_abs = _t_step

            # Save for position plot visual markers
            _step_vis[_i] = dict(
                final_ref    = _final_ref[_i],
                band         = _band,
                t_rise_abs   = float(_t_w[_rise_idx90[0]]) if len(_rise_idx90) else float('nan'),
                t_settle_abs = _t_settle_abs,
            )

            _rise_str   = f"{_rise:.2f} s"   if not np.isnan(_rise) else "n/a"
            _over_str   = f"{_overshoot_pct:+.1f} %"
            _settle_str = f"{_settle:.2f} s" if _settle < STEP_ANALYSIS_WINDOW else f">{STEP_ANALYSIS_WINDOW:.0f} s"

            print(f"  {_ax:<6}  {_mag:>+7.3f} m  {_rise_str:>12}  {_over_str:>12}  {_settle_str:>12}")

        print(f"  {'─'*56}")

# tracking error summary
if PLOT_MODE == "sim":
    print(f"\n{'═'*56}")
    print("TRACKING ERROR SUMMARY")
    print(f"{'═'*56}")
    if _USE_CYL_REF:
        _r_m  = np.maximum(np.sqrt(X[0]**2 + X[1]**2), 1e-6)
        _θ_m  = np.arctan2(X[1], X[0])
        _er   = ref_pos[0] - _r_m
        _eθ   = np.arctan2(np.sin(ref_pos[1] - _θ_m), np.cos(ref_pos[1] - _θ_m))
        _ez   = ref_pos[2] - X[2]
        _ṙ_m  =  X[6]*np.cos(_θ_m) + X[7]*np.sin(_θ_m)
        _θ̇_m  = (-X[6]*np.sin(_θ_m) + X[7]*np.cos(_θ_m)) / _r_m
        _eṙ   = ref_vel[0] - _ṙ_m
        _eθ̇   = ref_vel[1] - _θ̇_m
        _eż   = ref_vel[2] - X[8]
        print(f"  {'':12s}  {'mean':>10s}   {'max':>10s}")
        print(f"  {'─'*38}")
        print(f"  pos r     :  {np.abs(_er).mean():>10.3f} m    {np.abs(_er).max():>10.3f} m")
        print(f"  pos θ     :  {np.abs(_eθ).mean():>10.4f} rad  {np.abs(_eθ).max():>10.4f} rad")
        print(f"  pos z     :  {np.abs(_ez).mean():>10.3f} m    {np.abs(_ez).max():>10.3f} m")
        print(f"  {'─'*38}")
        print(f"  vel ṙ     :  {np.abs(_eṙ).mean():>10.3f} m/s  {np.abs(_eṙ).max():>10.3f} m/s")
        print(f"  vel θ̇     :  {np.abs(_eθ̇).mean():>10.4f} r/s  {np.abs(_eθ̇).max():>10.4f} r/s")
        print(f"  vel ż     :  {np.abs(_eż).mean():>10.3f} m/s  {np.abs(_eż).max():>10.3f} m/s")
        if USE_EVENT_TRIG:
            # event-triggered error: drone vs the reference the controller actually saw
            _ev_r_m  = np.maximum(np.sqrt(X[0]**2 + X[1]**2), 1e-6)
            _ev_θ_m  = np.arctan2(X[1], X[0])
            _ev_er   = Ref_log[0] - _ev_r_m
            _ev_eθ   = np.arctan2(np.sin(Ref_log[1] - _ev_θ_m), np.cos(Ref_log[1] - _ev_θ_m))
            _ev_ez   = Ref_log[2] - X[2]
            print(f"\n  (event-triggered reference — what controller actually tracked)")
            print(f"  {'─'*38}")
            print(f"  pos r     :  {np.abs(_ev_er).mean():>10.3f} m    {np.abs(_ev_er).max():>10.3f} m")
            print(f"  pos θ     :  {np.abs(_ev_eθ).mean():>10.4f} rad  {np.abs(_ev_eθ).max():>10.4f} rad")
            print(f"  pos z     :  {np.abs(_ev_ez).mean():>10.3f} m    {np.abs(_ev_ez).max():>10.3f} m")
    else:
        _ep = ref_pos_cart - X[0:3]
        _ev = ref_vel_cart - X[6:9]
        print(f"  {'':12s}  {'mean':>10s}   {'max':>10s}")
        print(f"  {'─'*38}")
        print(f"  pos x     :  {np.abs(_ep[0]).mean():>10.3f} m    {np.abs(_ep[0]).max():>10.3f} m")
        print(f"  pos y     :  {np.abs(_ep[1]).mean():>10.3f} m    {np.abs(_ep[1]).max():>10.3f} m")
        print(f"  pos z     :  {np.abs(_ep[2]).mean():>10.3f} m    {np.abs(_ep[2]).max():>10.3f} m")
        print(f"  {'─'*38}")
        print(f"  vel x     :  {np.abs(_ev[0]).mean():>10.3f} m/s  {np.abs(_ev[0]).max():>10.3f} m/s")
        print(f"  vel y     :  {np.abs(_ev[1]).mean():>10.3f} m/s  {np.abs(_ev[1]).max():>10.3f} m/s")
        print(f"  vel z     :  {np.abs(_ev[2]).mean():>10.3f} m/s  {np.abs(_ev[2]).max():>10.3f} m/s")

if PLOT_MODE == "sim" and TRAJ_MODE == "test_time_air":
    # rotor ceiling requirements
    def _rpm_sc(w): return w * 60.0 / (2.0 * np.pi)

    _wr  = X[12:16, :]         # actual rotor speeds (4, N), always >= 0
    _wrc = Wr_log               # commanded (4, N)

    _wr_max  = float(np.max(_wr))
    _wr_min  = float(np.min(_wr))

    # Angular acceleration from first-order motor model: dw/dt = (w_cmd - w) / tau_m
    _alpha_r     = np.abs(_wrc[:, :-1] - _wr[:, :-1]) / p.tau_m
    _alpha_r_max = float(np.max(_alpha_r))
    _t_hover_to_max = (_wr_max - p.omega_h) / _alpha_r_max if _alpha_r_max > 0 else float('inf')
    _t_max_to_zero  = _wr_max  / _alpha_r_max if _alpha_r_max > 0 else float('inf')
    _t_min_to_hover = (p.omega_h - _wr_min) / _alpha_r_max if _alpha_r_max > 0 else float('inf')

    # Thrust, torque, power per rotor
    _T_max_r   = p.kT * _wr_max**2
    _T_min_r   = p.kT * _wr_min**2
    _T_hover_r = p.kT * p.omega_h**2
    _Q_max_r   = p.kQ * _wr_max**2
    _Q_hover_r = p.kQ * p.omega_h**2
    _P_max_r   = p.kQ * _wr_max**3    # P ≈ torque × speed
    _P_hover_r = p.kQ * p.omega_h**3

    W = 64
    print(f"\n{'═'*W}")
    print("  ROTOR CEILING REQUIREMENTS  (motor / ESC sizing)")
    print(f"{'═'*W}")
    print(f"  {'Design hover speed':<34}  {p.omega_h:>9.1f} rad/s  ({_rpm_sc(p.omega_h):>6.0f} RPM)")
    print(f"  {'Saturation limit':<34}  {p.omega_max:>9.1f} rad/s  ({_rpm_sc(p.omega_max):>6.0f} RPM)")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max speed reached':<34}  {_wr_max:>9.1f} rad/s  ({_rpm_sc(_wr_max):>6.0f} RPM)")
    print(f"  {'Min speed reached':<34}  {_wr_min:>9.1f} rad/s  ({_rpm_sc(_wr_min):>6.0f} RPM)")
    print(f"  {'Saturation margin remaining':<34}  {p.omega_max - _wr_max:>9.1f} rad/s  ({_rpm_sc(p.omega_max - _wr_max):>6.0f} RPM)")
    print(f"  {'Saturation utilisation':<34}  {100*_wr_max/p.omega_max:>9.1f} %")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max angular acceleration':<34}  {_alpha_r_max:>9.0f} rad/s²")
    print(f"  {'Time  hover → max speed':<34}  {_t_hover_to_max:>9.3f} s")
    print(f"  {'Time  max speed → zero':<34}  {_t_max_to_zero:>9.3f} s")
    print(f"  {'Time  min speed → hover':<34}  {_t_min_to_hover:>9.3f} s")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max thrust    (per rotor)':<34}  {_T_max_r:>9.2f} N")
    print(f"  {'Min thrust    (per rotor)':<34}  {_T_min_r:>9.2f} N")
    print(f"  {'Hover thrust  (per rotor)':<34}  {_T_hover_r:>9.2f} N")
    print(f"  {'Max total thrust (4 rotors)':<34}  {4*_T_max_r:>9.2f} N")
    print(f"  {'Hover total thrust':<34}  {4*_T_hover_r:>9.2f} N   (= {p.m*p.g:.1f} N = m·g  ✓)")
    print(f"  {'Thrust-to-weight ratio (max)':<34}  {4*_T_max_r/(p.m*p.g):>9.2f}")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max reaction torque  (per rotor)':<34}  {_Q_max_r:>9.4f} N·m")
    print(f"  {'Hover reaction torque (per rotor)':<34}  {_Q_hover_r:>9.4f} N·m")
    print(f"  {'Max power estimate   (per rotor)':<34}  {_P_max_r:>9.2f} W   (kQ × w³)")
    print(f"  {'Hover power estimate  (per rotor)':<34}  {_P_hover_r:>9.2f} W")
    print(f"{'═'*W}")

# --- plots ---

if PLOT_MODE not in ("sim", "root_locus"):
    print("\nPlots suppressed (PLOT_MODE not recognised)")

elif PLOT_MODE == "root_locus":
    from matplotlib.widgets import Slider

    _pKp = cyl_Kp.copy().astype(float)
    _pKd = cyl_Kd.copy().astype(float)
    _aKp = att_Kp.copy().astype(float)
    _aKd = att_Kd.copy().astype(float)
    _z3  = np.zeros(3)

    def _inner_poles():
        K = build_K_cl(_z3, _z3, _z3, _aKp, _z3, _aKd)
        return np.linalg.eigvals(A_lin + B_lin @ K)

    def _A_inner_cl():
        K = build_K_cl(_z3, _z3, _z3, _aKp, _z3, _aKd)
        return A_lin + B_lin @ K

    def _full_poles():
        K = build_K_cl(_pKp, _z3, _pKd, _aKp, _z3, _aKd)
        return np.linalg.eigvals(A_lin + B_lin @ K)

    fig_rl = plt.figure(figsize=(20, 13))
    fig_rl.suptitle(
        f"Sequential closed-loop pole analysis  —  gains: '{_flight_mode}'  |  Ki omitted",
        fontsize=12, fontweight='bold')

    ax_in  = fig_rl.add_axes([0.06, 0.42, 0.40, 0.50])
    ax_out = fig_rl.add_axes([0.55, 0.42, 0.40, 0.50])

    def _setup(ax, title):
        ax.axvline(0, color='k', lw=0.9, ls='--')
        ax.axhline(0, color='k', lw=0.9, ls='--')
        ax.set_xlabel("Real  [rad/s]", fontsize=10)
        ax.set_ylabel("Imaginary  [rad/s]", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True)

    _setup(ax_in,
           "INNER LOOP  (attitude + motor)\n"
           "Plant: open-loop  |  feedback: att_Kp, att_Kd only")
    _setup(ax_out,
           "OUTER LOOP  (position)\n"
           "Plant: inner-loop-closed  |  feedback: pos_Kp, pos_Kd  +  fixed att gains")

    ol_poles = np.linalg.eigvals(A_lin)
    ax_in.scatter(ol_poles.real, ol_poles.imag,
                  marker='x', s=80, color='red', lw=1.5,
                  label='Plant (open-loop)', zorder=7)
    ip0    = _inner_poles()
    sc_in  = ax_in.scatter(ip0.real, ip0.imag,
                           marker='x', s=130, lw=2.5,
                           color='darkorange', label='Attitude closed-loop', zorder=5)
    ax_in.legend(loc='upper right', fontsize=8)

    icp0   = np.linalg.eigvals(_A_inner_cl())
    sc_ref = ax_out.scatter(icp0.real, icp0.imag,
                            marker='o', s=50, color='lightgray', edgecolors='gray',
                            lw=1, label='Plant (inner loop closed)', zorder=3)
    fp0    = _full_poles()
    sc_out = ax_out.scatter(fp0.real, fp0.imag,
                            marker='x', s=130, lw=2.5,
                            color='royalblue', label='Full closed-loop', zorder=5)
    ax_out.legend(loc='upper right', fontsize=8)

    def _set_initial_lims(ax, *pole_sets):
        r  = np.concatenate([p.real for p in pole_sets])
        im = np.concatenate([p.imag for p in pole_sets])
        pr = max(abs(r).max()  * 0.15, 1.0)
        pi = max(abs(im).max() * 0.15, 1.0)
        ax.set_xlim(r.min()  - pr, max(r.max()  + pr, 0.5))
        ax.set_ylim(im.min() - pi, im.max() + pi)

    def _refresh_inner():
        ip = _inner_poles()
        sc_in.set_offsets(np.column_stack([ip.real, ip.imag]))
        icp = np.linalg.eigvals(_A_inner_cl())
        sc_ref.set_offsets(np.column_stack([icp.real, icp.imag]))
        fp = _full_poles()
        sc_out.set_offsets(np.column_stack([fp.real, fp.imag]))
        fig_rl.canvas.draw_idle()

    def _refresh_outer():
        fp = _full_poles()
        sc_out.set_offsets(np.column_stack([fp.real, fp.imag]))
        fig_rl.canvas.draw_idle()

    _refresh_inner()

    _set_initial_lims(ax_in,  ol_poles, ip0)
    _set_initial_lims(ax_out, icp0,     fp0)

    SL_H   = 0.060
    SL_W   = 0.115
    SL_GAP = 0.020
    y_kd   = 0.05
    y_kp   = y_kd + SL_H + 0.05

    att_lbls = [u'φ', u'θ', u'ψ']   # φ θ ψ
    pos_lbls = ['r', 't', 'z']   # r=radial, t=tangential(arc), z=height
    att_cols = ['darkorange', 'gold',      'coral'    ]
    pos_cols = ['royalblue',  'steelblue', 'dodgerblue']

    all_sl_refs = []
    sl_data     = []

    def _add_sliders(x0, Kp_arr, Kd_arr, lbls, cols, prefix):
        for ci in range(3):
            x_col = x0 + ci * (SL_W + SL_GAP)
            for gname, garr, y_row in [('Kp', Kp_arr, y_kp), ('Kd', Kd_arr, y_kd)]:
                ax_sl = fig_rl.add_axes([x_col, y_row, SL_W, SL_H])
                v0    = float(garr[ci])
                vmax  = max(v0 * 1.5, 2.0)
                sl    = Slider(ax_sl, lbls[ci], 0.0, vmax,
                               valinit=v0, valstep=vmax/500, color=cols[ci])
                # Anchor label inside the axes so wide text can't bleed left
                sl.label.set_fontsize(11)
                sl.label.set_position((0.03, 0.5))
                sl.label.set_horizontalalignment('left')
                # Anchor value text inside the axes so it can't bleed right
                sl.valtext.set_fontsize(8)
                sl.valtext.set_position((0.97, 0.5))
                sl.valtext.set_horizontalalignment('right')
                all_sl_refs.append(sl)
                sl_data.append((sl, gname, ci, prefix))

    _add_sliders(0.06, _aKp, _aKd, att_lbls, att_cols, 'att')
    _add_sliders(0.55, _pKp, _pKd, pos_lbls, pos_cols, 'pos')

    # Row and column headers
    for y_row, lbl in [(y_kp, 'Kp'), (y_kd, 'Kd')]:
        fig_rl.text(0.005, y_row + SL_H/2, lbl, fontsize=12,
                    fontweight='bold', va='center', color='dimgray')
        fig_rl.text(0.505, y_row + SL_H/2, lbl, fontsize=12,
                    fontweight='bold', va='center', color='dimgray')

    fig_rl.text(0.06 + 1*(SL_W+SL_GAP), y_kp + SL_H + 0.010,
                u'INNER  —  att_Kp / att_Kd  (φ, θ, ψ)', fontsize=9,
                fontweight='bold', ha='center', color='dimgray')
    fig_rl.text(0.55 + 1*(SL_W+SL_GAP), y_kp + SL_H + 0.010,
                u'OUTER  —  cyl_Kp / cyl_Kd  (r, t, z)', fontsize=9,
                fontweight='bold', ha='center', color='dimgray')

    def _make_cb(gname, ci, prefix):
        def cb(val):
            if prefix == 'att':
                if gname == 'Kp': _aKp[ci] = val
                else:             _aKd[ci] = val
                _refresh_inner()   # att change affects both plots
            else:
                if gname == 'Kp': _pKp[ci] = val
                else:             _pKd[ci] = val
                _refresh_outer()   # pos change affects outer plot only
        return cb

    for sl, gname, ci, prefix in sl_data:
        sl.on_changed(_make_cb(gname, ci, prefix))

    plt.show()

else:  # PLOT_MODE == "sim"
    _ps        = max(1, N // 10_000)
    def _zclip(a): return np.where(np.abs(a) < 1e-10, 0.0, a)
    t_p        = t[::_ps]
    X_p        = _zclip(X[:, ::_ps])
    rp_p       = _zclip(ref_pos_cart[:, ::_ps])
    rv_p       = _zclip(ref_vel_cart[:, ::_ps])
    Ul_p       = _zclip(U_log[:, ::_ps])
    ref_yaw_p  = _zclip(ref_yaw[::_ps])

    if _USE_CYL_REF:
        _r_p  = np.maximum(np.sqrt(X_p[0,:]**2 + X_p[1,:]**2), 1e-6)
        _θ_p  = np.unwrap(np.arctan2(X_p[1,:], X_p[0,:]))
        _z_p  = X_p[2,:]
        _vr_p =  X_p[6,:]*np.cos(_θ_p) + X_p[7,:]*np.sin(_θ_p)
        _vθ_p = (-X_p[6,:]*np.sin(_θ_p) + X_p[7,:]*np.cos(_θ_p)) / _r_p
        _vz_p =  X_p[8,:]

        _rr_p  = ref_pos[0, ::_ps];  _θr_p = ref_pos[1, ::_ps];  _zr_p = ref_pos[2, ::_ps]
        _vr_r  = ref_vel[0, ::_ps];  _vθ_r = ref_vel[1, ::_ps];  _vz_r = ref_vel[2, ::_ps]

        _er_p  = _rr_p - _r_p
        _eθ_p  = np.arctan2(np.sin(_θr_p - _θ_p), np.cos(_θr_p - _θ_p))
        _ez_p  = _zr_p - _z_p
        _evr_p = _vr_r - _vr_p
        _evθ_p = _vθ_r - _vθ_p
        _evz_p = _vz_r - _vz_p

    gc = (0.85, 0.95, 0.85)   # gust window shading colour

    def shade_gusts(ax):
        yl = ax.get_ylim()
        if DIST_ENABLED:
            for row in DISTURBANCES:
                ax.axvspan(row[0], row[1], color=gc, alpha=0.5, zorder=0)
        ax.set_ylim(yl)

    # fig 1: position + attitude
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig1.suptitle(f"Position & Attitude  [{TRAJ_MODE}]", fontsize=13)

    if _USE_CYL_REF:
        pos_labels  = ['r  [m]', 'θ  [rad]', 'z  [m]']
        pos_actual  = [_r_p, _θ_p, _z_p]
        pos_ref     = [_rr_p, _θr_p, _zr_p]
    else:
        pos_labels  = ['x  [m]', 'y  [m]', 'z  [m]']
        pos_actual  = [X_p[0,:], X_p[1,:], X_p[2,:]]
        pos_ref     = [rp_p[0,:], rp_p[1,:], rp_p[2,:]]

    att_labels = ['φ  [deg]', 'θ  [deg]', 'ψ  [deg]']

    for i in range(3):
        ax = axes1[i, 0]
        if PLOT_ACTUAL:
            ax.plot(t_p, pos_actual[i], 'b',   lw=1.6, label='Actual')
        if PLOT_REFERENCE:
            ax.plot(t_p, pos_ref[i],    'r--', lw=1.2, label='Reference')
        # Step response visual markers (custom mode only)
        if _step_vis is not None and _step_vis[i] is not None:
            _sv = _step_vis[i]
            ax.axhline(_sv['final_ref'] + _sv['band'], color='limegreen',    ls=':', lw=1.0)
            ax.axhline(_sv['final_ref'] - _sv['band'], color='limegreen',    ls=':', lw=1.0)
            if not np.isnan(_sv['t_rise_abs']):
                ax.axvline(_sv['t_rise_abs'],   color='darkorange',   ls=':', lw=1.2)
            ax.axvline(_sv['t_settle_abs'],     color='mediumpurple', ls=':', lw=1.2)
        ax.set_ylabel(pos_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0:
            ax.set_title("Position (cylindrical)" if _USE_CYL_REF else "Position")
        _leg_h = []
        if PLOT_ACTUAL:    _leg_h.append(ax.plot([], [], 'b',   lw=1.6, label='Actual')[0])
        if PLOT_REFERENCE: _leg_h.append(ax.plot([], [], 'r--', lw=1.2, label='Reference')[0])
        if _step_vis is not None and _step_vis[i] is not None:
            _leg_h.append(ax.plot([], [], color='limegreen',    ls=':', lw=1.0, label=f'Settle band (±{SETTLING_THRESHOLD_PCT:.0f}%)')[0])
            _leg_h.append(ax.plot([], [], color='darkorange',   ls=':', lw=1.2, label='Rise time (90%)')[0])
            _leg_h.append(ax.plot([], [], color='mediumpurple', ls=':', lw=1.2, label='Settle time')[0])
        ax.legend(handles=_leg_h, loc='lower right')

        ax = axes1[i, 1]
        ax.plot(t_p, np.degrees(X_p[3+i, :]), 'b', lw=1.6, label='Actual')
        if i == 2 and PLOT_REFERENCE:
            ax.plot(t_p, np.degrees(np.unwrap(ref_yaw_p)), 'r--', lw=1.2, label='Reference')
            ax.legend(loc='lower right')
        ax.set_ylabel(att_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0:
            ax.set_title("Attitude")

    axes1[2, 0].set_xlabel("Time  [s]")
    axes1[2, 1].set_xlabel("Time  [s]")
    for ax in fig1.axes: ax.tick_params(labelbottom=True)
    fig1.tight_layout()

    # fig 2: velocity tracking
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig2.suptitle(f"Velocity Tracking  [{TRAJ_MODE}]", fontsize=13)

    if _USE_CYL_REF:
        vel_labels  = ['ṙ  [m/s]', 'θ̇  [rad/s]', 'ż  [m/s]']
        vel_actual  = [_vr_p, _vθ_p, _vz_p]
        vel_ref     = [_vr_r, _vθ_r, _vz_r]
    else:
        vel_labels  = ['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]']
        vel_actual  = [X_p[6,:], X_p[7,:], X_p[8,:]]
        vel_ref     = [rv_p[0,:], rv_p[1,:], rv_p[2,:]]

    for i in range(3):
        ax = axes2[i]
        if PLOT_ACTUAL:
            ax.plot(t_p, vel_actual[i], 'b',   lw=1.6, label='Actual')
        if PLOT_REFERENCE:
            ax.plot(t_p, vel_ref[i],    'r--', lw=1.2, label='Reference')
        ax.set_ylabel(vel_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0:
            ax.legend(loc='lower right')

    axes2[-1].set_xlabel("Time  [s]")
    for ax in fig2.axes: ax.tick_params(labelbottom=True)
    fig2.tight_layout()

    if _USE_CYL_REF:
        fig2b, axes2b = plt.subplots(3, 2, figsize=(13, 8), sharex=True)
        fig2b.suptitle(f"Cylindrical Tracking Errors  [{TRAJ_MODE}]", fontsize=13)

        pos_err_data = [(_er_p,  'e_r  [m]',      'Standoff error'),
                        (_eθ_p,  'e_θ  [rad]',     'Azimuth error'),
                        (_ez_p,  'e_z  [m]',       'Height error')]
        vel_err_data = [(_evr_p, 'e_ṙ  [m/s]',    'Radial vel error'),
                        (_evθ_p, 'e_θ̇  [rad/s]',  'Angular vel error'),
                        (_evz_p, 'e_ż  [m/s]',    'Vertical vel error')]

        for i, ((pe, pl, pt), (ve, vl, vt)) in enumerate(zip(pos_err_data, vel_err_data)):
            ax = axes2b[i, 0]
            ax.plot(t_p, pe, 'b', lw=1.6)
            ax.axhline(0, color='k', ls=':', lw=0.8)
            ax.set_ylabel(pl); ax.grid(True)
            shade_gusts(ax)
            if i == 0: ax.set_title("Position error  (ref − actual)")

            ax = axes2b[i, 1]
            ax.plot(t_p, ve, 'darkorange', lw=1.6)
            ax.axhline(0, color='k', ls=':', lw=0.8)
            ax.set_ylabel(vl); ax.grid(True)
            shade_gusts(ax)
            if i == 0: ax.set_title("Velocity error  (ref − actual)")

        axes2b[2, 0].set_xlabel("Time  [s]")
        axes2b[2, 1].set_xlabel("Time  [s]")
        for ax in fig2b.axes: ax.tick_params(labelbottom=True)
        fig2b.tight_layout()

    fig3, axes3 = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig3.suptitle("Virtual Control Inputs", fontsize=13)
    u_labels = ['T  [N]', 'tau_phi  [N·m]', 'tau_th  [N·m]', 'tau_psi  [N·m]']

    for i in range(4):
        ax = axes3[i]
        ax.plot(t_p, Ul_p[i, :], 'b', lw=1.6)
        if i == 0:
            ax.axhline(p.m*p.g, color='r', ls='--', lw=1.0, label='Hover T')
            ax.legend(loc='lower right')
        else:
            ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.set_ylabel(u_labels[i]); ax.grid(True)
        shade_gusts(ax)

    axes3[-1].set_xlabel("Time  [s]")
    for ax in fig3.axes: ax.tick_params(labelbottom=True)
    fig3.tight_layout()

    fig4, axes4 = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig4.suptitle("Rotor Speeds", fontsize=13)
    motor_names = ['Front (1)', 'Right (2)', 'Rear (3)', 'Left (4)']

    for i in range(4):
        ax = axes4[i]
        ax.plot(t_p, X_p[12+i, :], 'b', lw=1.6)
        ax.axhline(p.omega_h,   color='r',  ls='--', lw=1.0, label='hover')
        ax.axhline(p.omega_max, color='k',  ls='--', lw=1.0, label='max')
        ax.set_ylabel(f"{motor_names[i]}  [rad/s]"); ax.grid(True)
        if i == 0: ax.legend(loc='lower right')
        shade_gusts(ax)

    axes4[-1].set_xlabel("Time  [s]")
    for ax in fig4.axes: ax.tick_params(labelbottom=True)
    fig4.tight_layout()

    fig5, ax5 = plt.subplots(figsize=(7, 6))
    poles = np.linalg.eigvals(A_lin)
    ax5.scatter(poles.real, poles.imag, marker='x', s=80, color='b', zorder=5)
    ax5.axvline(0, color='k', lw=0.8, ls='--')
    ax5.axhline(0, color='k', lw=0.8, ls='--')
    ax5.set_xlabel("Real"); ax5.set_ylabel("Imaginary")
    ax5.set_title("Open-Loop Poles at Hover (linearised)")
    ax5.grid(True)
    fig5.tight_layout()

    fig6 = plt.figure(figsize=(11, 10))
    ax6  = fig6.add_subplot(111, projection='3d')

    if TRAJ_MODE == "test_time_air":
        from Test_time import (R_base as _R_base, R_top as _R_top,
                               H_air_cyl as _H_cyl, H_air_cone as _H_cone,
                               R_blade as _R_blade, H_blade as _H_blade)
        _H_air = _H_cyl + _H_cone

        def _r_at_z(z):
            if z <= _H_cyl:
                return _R_base
            return _R_base - ((_R_base - _R_top) / _H_cone) * (z - _H_cyl)

        def _blade_transform(x, y, z, angle_deg, z_offset):
            rad = np.radians(angle_deg)
            xr  =  x * np.cos(rad) + z * np.sin(rad)
            yr  =  y
            zr  = -x * np.sin(rad) + z * np.cos(rad) + z_offset
            return xr, yr, zr

        _z_s    = np.linspace(0, _H_air, 60)
        _th_s   = np.linspace(0, 2 * np.pi, 60)
        _TH, _Z = np.meshgrid(_th_s, _z_s)
        _R      = np.vectorize(_r_at_z)(_Z)
        ax6.plot_surface(_R * np.cos(_TH), _R * np.sin(_TH), _Z,
                         color='silver', alpha=0.30, edgecolor='none')

        _zb  = np.linspace(0, _H_blade, 20)
        _thb = np.linspace(0, 2 * np.pi, 20)
        _TB, _ZB = np.meshgrid(_thb, _zb)
        _XB = _R_blade * np.cos(_TB)
        _YB = _R_blade * np.sin(_TB)

        for _ang in [0, 120, 240]:
            _xr, _yr, _zr = _blade_transform(_XB, _YB, _ZB, _ang, _H_air)
            ax6.plot_surface(_xr, _yr, _zr, color='gold', alpha=0.25, edgecolor='none')

    if PLOT_REFERENCE:
        if TRAJ_MODE == "test_time_air":
            _rx = _wp[:, 1] * np.cos(_wp[:, 2])
            _ry = _wp[:, 1] * np.sin(_wp[:, 2])
            _rz = _wp[:, 3]
        else:
            _rx, _ry, _rz = rp_p[0, :], rp_p[1, :], rp_p[2, :]
        ax6.plot(_rx, _ry, _rz,
                 color='red', lw=1.8, ls='--', label='Reference', zorder=5)
    if PLOT_ACTUAL:
        ax6.plot(X_p[0, :], X_p[1, :], X_p[2, :],
                 color='blue', lw=1.8, label='Actual (controller)', zorder=6)

    ax6.set_xlabel("X  [m]")
    ax6.set_ylabel("Y  [m]")
    ax6.set_zlabel("Z  [m]")
    ax6.set_title(f"3D Flight Path: Reference vs Actual  [{TRAJ_MODE}]", fontsize=13)
    ax6.legend(loc='upper left')

    if TRAJ_MODE == "test_time_air":
        all_x = np.concatenate([_rx, X_p[0, :]])
        all_y = np.concatenate([_ry, X_p[1, :]])
        all_z = np.concatenate([_rz, X_p[2, :]])
        x_mid = (all_x.max() + all_x.min()) / 2
        y_mid = (all_y.max() + all_y.min()) / 2
        z_mid = (all_z.max() + all_z.min()) / 2
        half  = max(all_x.max() - all_x.min(),
                    all_y.max() - all_y.min(),
                    all_z.max() - all_z.min()) / 2
        ax6.set_xlim(x_mid - half, x_mid + half)
        ax6.set_ylim(y_mid - half, y_mid + half)
        ax6.set_zlim(z_mid - half, z_mid + half)
        ax6.set_box_aspect([1, 1, 1])

    fig6.tight_layout()

    # wind profile plot
    if WIND_ENABLED:
        _vw = np.array([get_wind(tk) for tk in t_p])
        fig_w, axes_w = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        fig_w.suptitle("Wind Profile  v(t)  [inertial frame]", fontsize=13)
        for i, (ax, lbl, col) in enumerate(zip(
                axes_w, ['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]'],
                ['steelblue', 'darkorange', 'seagreen'])):
            ax.plot(t_p, _vw[:, i], color=col, lw=1.8)
            ax.axhline(0, color='k', lw=0.7, ls=':')
            ax.set_ylabel(lbl); ax.grid(True)
            shade_gusts(ax)
        axes_w[-1].set_xlabel("Time  [s]")
        for ax in fig_w.axes: ax.tick_params(labelbottom=True)
        fig_w.tight_layout()

    plt.show()
