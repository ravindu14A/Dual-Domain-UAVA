"""
quadcopterSC.py  —  Quadcopter Aerial Phase: Stability & Control
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

6-DOF nonlinear model + cascaded PID + wind gust disturbance analysis.

STATE (16):  [x  y  z | phi  theta  psi | xd  yd  zd | p  q  r | w1  w2  w3  w4]
              pos(3)    euler(3)           vel(3)        ang_rate(3) rotors(4)

MOTOR LAYOUT (+config, ENU, body x=forward, y=left, z=up):
  1=front(+x)  2=right(-y)  3=rear(-x)  4=left(+y)
  Motors 1,3: CCW (+z reaction)  |  2,4: CW (-z reaction)

EULER CONVENTION: ZYX
  positive theta = nose tilted DOWN (forward tilt → +x force)
  positive phi   = right side UP    (tilt toward +y → +y force when phi<0)

Dependencies:  numpy  scipy  matplotlib  control
    pip install numpy scipy matplotlib control
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
import control
from Test_time import air_config, cameras, R_base, R_top, H_air_cyl, H_air_cone


def _lawnmower_turn_waypoints(a_max=2.0, r_corner=0.0):
    """
    Dense time-stamped waypoints for ONE full lawnmower cycle:
      up strip → horizontal step → down strip
    with trapezoidal velocity profiles and optional corner rounding.

    r_corner > 0  inserts a quarter-circle arc at each 90° turn so the
    geometric path is smooth. Both arcs use their own trap speed profile
    (drone slows through corners, accelerates away).

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
    v_horiz = v_max

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

    # ── Helpers ────────────────────────────────────────────────────────────────
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

    # ── Path coordinates ───────────────────────────────────────────────────────
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
        # Corner 1: vertical up → horizontal at theta=0
        P_c1   = np.array([r_top, 0.0, H_tot])
        arc1   = make_corner(P_c1, [0, 0, 1], [0, 1, 0])

        # Corner 2: horizontal at theta=d_theta → vertical down
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

    # ── Apply trapezoidal profiles ─────────────────────────────────────────────
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

    # ── Hold at start and end ─────────────────────────────────────────────────
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

    # ── Convert Cartesian path to cylindrical (r, θ, z) and (ṙ, θ̇, ż) ──────────
    r_arr = np.sqrt(all_xyz[:, 0]**2 + all_xyz[:, 1]**2)
    θ_arr = np.arctan2(all_xyz[:, 1], all_xyz[:, 0])
    z_arr = all_xyz[:, 2]

    vr_arr = ( all_vel[:, 0] * np.cos(θ_arr) + all_vel[:, 1] * np.sin(θ_arr))
    vθ_arr = (-all_vel[:, 0] * np.sin(θ_arr) + all_vel[:, 1] * np.cos(θ_arr)) / np.maximum(r_arr, 1e-6)
    vz_arr = all_vel[:, 2]

    return np.column_stack([all_t, r_arr, θ_arr, z_arr, vr_arr, vθ_arr, vz_arr])


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONFIGURATION  —  edit only in this section
# ══════════════════════════════════════════════════════════════════════════════

# ── Trajectory source ────────────────────────────────────────────────────────
TRAJ_MODE = "test_time_air"
#   "hold"         : hold at origin for the whole simulation
#   "custom"       : waypoints from TRAJ_SEGMENTS below
#   "test_time_air": full aerial path + velocity profile from Test_time.py

# Custom waypoint table — only used when TRAJ_MODE = "custom"
#
# Each row: (t_start, x, y, z)              — hold at (x,y,z) from t_start
#        or (t_start, x, y, z, vx, vy, vz)  — ref starts at (x,y,z) and moves
#                                              continuously at (vx,vy,vz) from t_start
#
# (x,y,z) is WHERE the reference IS at t_start — not a target to reach.
# The reference moves at (vx,vy,vz) until the next row's t_start, then snaps
# to the next row's (x,y,z) and continues at the next row's velocity.
#
# Example A — simple holds (no velocity):
# TRAJ_SEGMENTS = [
#     ( 0.0,  0.0, 0.0, 0.0),
#     ( 3.0,  0.0, 0.0, 2.0),
#     ( 8.0,  3.0, 0.0, 2.0),
# ]
#
# Example B — one lawnmower turn from Test_time RGB lawnmower config:
#   r = R_base + D = 10 m,  H = H_air_cyl = 30 m
#   v_vert = 4.0 m/s (blur-limited),  v_horiz = 10.0 m/s,  strip angle = 32.7 deg
#
#   t_start    x       y      z     vx       vy      vz
TRAJ_ACCEL_MAX = 2.0   # [m/s²] ramp acceleration for trapezoidal velocity profile
CORNER_RADIUS  = 0   # [m]    corner-rounding radius at strip top/bottom; 0 = sharp corners
TRAJ_SEGMENTS  = _lawnmower_turn_waypoints(a_max=TRAJ_ACCEL_MAX, r_corner=CORNER_RADIUS)

# ── Disturbances ─────────────────────────────────────────────────────────────
DIST_ENABLED = False
# Each row: (t_on, t_off,  Fx, Fy, Fz [N inertial],  tx, ty, tz [N·m body])
# Multiple rows are superimposed. Set DIST_ENABLED = False to disable all.
DISTURBANCES = [
    (14.0, 16.0,   4.0, 0.0, 0.0,   0.0, 0.0, 0.0),
]

# ── Simulation timing ─────────────────────────────────────────────────────────
T_BUFFER = 4.0    # [s] extra run-time appended after the last event

# ── Output ───────────────────────────────────────────────────────────────────
PLOT_MODE       = "sim"        # "sim"        → full simulation plots (figs 1–6)
                               # "root_locus" → closed-loop pole map only
PLOT_REFERENCE  = True         # show reference trajectory lines  (sim mode only)
PLOT_ACTUAL     = True         # show actual (controller) trajectory lines (sim mode only)
ENABLE_PLOTS    = PLOT_MODE in ("sim", "root_locus")

# ══════════════════════════════════════════════════════════════════════════════
# VEHICLE PARAMETERS  (placeholders — replace with design values)
# ══════════════════════════════════════════════════════════════════════════════

class Params:
    m     = 10.0      # [kg]         total mass (incl. ballast + UW propellers)
    Ixx   = 0.15      # [kg·m²]      roll inertia
    Iyy   = 0.15      # [kg·m²]      pitch inertia
    Izz   = 0.26      # [kg·m²]      yaw inertia
    Ixz   = 0.0       # [kg·m²]      xz product of inertia (ZX-plane symmetry: Ixy=Iyz=0)
    l     = 0.35      # [m]          arm length (CoM to rotor centre)
    g     = 9.81      # [m/s²]

    kT    = 1.5e-4    # [N·s²/rad²]    thrust coeff  F = kT·ω²
    kQ    = 3.0e-6    # [N·m·s²/rad²]  torque coeff  Q = kQ·ω²
    tau_m = 0.06      # [s]            motor first-order lag
    kd    = 0.15      # [N·s/m]        translational drag
    Jr    = 0.0       # [kg·m²]        rotor spin inertia (0 = ignore gyroscopic)

    omega_max = 700.0  # [rad/s]  rotor saturation

    @property
    def omega_h(self):
        """Hover rotor speed from 4·kT·ωh² = m·g"""
        return np.sqrt(self.m * self.g / (4 * self.kT))

p = Params()
print(f"Hover w = {p.omega_h:.1f} rad/s  ({p.omega_h * 60 / (2*np.pi):.0f} RPM)")

# ══════════════════════════════════════════════════════════════════════════════
# PID GAINS  —  one set per trajectory mode
# ══════════════════════════════════════════════════════════════════════════════
#
# Cascade rule: inner-loop bandwidth >> outer-loop bandwidth
#   Inner (attitude) ~50–200 Hz equivalent,  Outer (position) ~5–20 Hz
#
# Spiral:    smooth continuous curved motion  → moderate gains, low damping
# Lawnmower: sharp vertical reversals at each strip end  → high z-axis Kp/Kd,
#            strong damping to kill overshoot at corners
# Custom / hold: original step-response tuning

_gains = {
    # ── mode : att gains  +  pos gains (Cartesian, used only for hold mode)
    #           +  cyl gains (cylindrical r/t/z, used for custom/test_time_air) ──
    #
    # Cylindrical outer loop axes:
    #   r  = standoff distance  [m]        — radial
    #   t  = arc-length tangential [m]     — r·e_θ keeps units consistent with r
    #   z  = height             [m]
    "hold": dict(
        att_Kp    = np.array([5.0,  5.0,  2.5]),
        att_Ki    = np.array([0.05, 0.05, 0.02]),
        att_Kd    = np.array([2.0,  2.0,  1.2]),
        att_i_lim = np.array([0.5,  0.5,  0.3]),
        pos_Kp    = np.array([1.8,  1.8,  2.5]),
        pos_Ki    = np.array([0.08, 0.08, 0.15]),
        pos_Kd    = np.array([1.2,  1.2,  1.8]),
        pos_i_lim = np.array([2.0,  2.0,  3.0]),
        att_lim   = 0.45,
        # cylindrical gains unused in hold mode but kept for consistent dict structure
        cyl_Kp    = np.array([1.8,  1.8,  2.5]),
        cyl_Ki    = np.array([0.08, 0.08, 0.15]),
        cyl_Kd    = np.array([1.2,  1.2,  1.8]),
        cyl_i_lim = np.array([2.0,  2.0,  3.0]),
    ),
    "custom": dict(
        att_Kp    = np.array([16.43,  16.32,  3.0]),
        att_Ki    = np.array([0.02, 0.02, 0.01]),
        att_Kd    = np.array([3.45,  3.4,  1.495]),
        att_i_lim = np.array([0.3,  0.3,  0.2]),
        pos_Kp    = np.array([1.876,  4.887,  4.73]),
        pos_Ki    = np.array([0.02, 0.02, 0.05]),
        pos_Kd    = np.array([2.23,  4.06,  4.08]),
        pos_i_lim = np.array([1.0,  1.0,  2.0]),
        att_lim   = 0.45,
        # cyl[0]=r  cyl[1]=tangential(arc)  cyl[2]=z
        cyl_Kp    = np.array([1.876, 4.887, 4.73]),
        cyl_Ki    = np.array([0.02,  0.02,  0.05]),
        cyl_Kd    = np.array([2.23,  4.06,  4.08]),
        cyl_i_lim = np.array([1.0,   1.0,   2.0]),
    ),
    # Spiral: smooth orbit — moderate bandwidth, no excess damping
    "spiral": dict(
        att_Kp    = np.array([6.0,  6.0,  2.5]),
        att_Ki    = np.array([0.05, 0.05, 0.02]),
        att_Kd    = np.array([2.5,  2.5,  1.2]),
        att_i_lim = np.array([0.5,  0.5,  0.3]),
        pos_Kp    = np.array([2.0,  2.0,  3.0]),
        pos_Ki    = np.array([0.05, 0.05, 0.10]),
        pos_Kd    = np.array([1.5,  1.5,  2.5]),
        pos_i_lim = np.array([2.0,  2.0,  3.0]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([2.0,  2.0,  3.0]),
        cyl_Ki    = np.array([0.05, 0.05, 0.10]),
        cyl_Kd    = np.array([1.5,  1.5,  2.5]),
        cyl_i_lim = np.array([2.0,  2.0,  3.0]),
    ),
    # Lawnmower: hard vertical reversals — strong z damping, fast attitude response
    "lawnmower": dict(
        att_Kp    = np.array([16.43,  16.32,  3.0]),
        att_Ki    = np.array([0.02, 0.02, 0.01]),
        att_Kd    = np.array([3.45,  3.4,  1.495]),
        att_i_lim = np.array([0.3,  0.3,  0.2]),
        pos_Kp    = np.array([1.876,  4.887,  4.73]),
        pos_Ki    = np.array([0.02, 0.02, 0.05]),
        pos_Kd    = np.array([2.23,  4.06,  4.08]),
        pos_i_lim = np.array([1.0,  1.0,  2.0]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([1.876, 4.887, 4.73]),
        cyl_Ki    = np.array([0.02,  0.02,  0.05]),
        cyl_Kd    = np.array([2.23,  4.06,  4.08]),
        cyl_i_lim = np.array([1.0,   1.0,   2.0]),
    ),
}

# Select gains based on current flight mode
_flight_mode = air_config["flight_mode"] if TRAJ_MODE == "test_time_air" else TRAJ_MODE
_g = _gains.get(_flight_mode, _gains["custom"])

att_Kp    = _g["att_Kp"]
att_Ki    = _g["att_Ki"]
att_Kd    = _g["att_Kd"]
att_i_lim = _g["att_i_lim"]
pos_Kp    = _g["pos_Kp"]
pos_Ki    = _g["pos_Ki"]
pos_Kd    = _g["pos_Kd"]
pos_i_lim = _g["pos_i_lim"]
att_lim   = _g["att_lim"]
# Cylindrical outer loop gains — [r, tangential(arc), z]
cyl_Kp    = _g["cyl_Kp"]
cyl_Ki    = _g["cyl_Ki"]
cyl_Kd    = _g["cyl_Kd"]
cyl_i_lim = _g["cyl_i_lim"]

print(f"PID gains       : {_flight_mode} set")

# ══════════════════════════════════════════════════════════════════════════════
# MIXING MATRIX   A_mix @ [ω1² ω2² ω3² ω4²] = [T  τ_φ  τ_θ  τ_ψ]
# ══════════════════════════════════════════════════════════════════════════════
#   τ_φ   = l·kT·(ω4² − ω2²)               left − right
#   τ_θ   = l·kT·(ω3² − ω1²)               rear  − front (+→ nose down → fwd)
#   τ_ψ   = kQ·(ω1² − ω2² + ω3² − ω4²)

A_mix = np.array([
    [ p.kT,          p.kT,         p.kT,        p.kT       ],
    [ 0,            -p.l*p.kT,     0,            p.l*p.kT  ],
    [-p.l*p.kT,      0,            p.l*p.kT,    0           ],
    [ p.kQ,         -p.kQ,         p.kQ,        -p.kQ      ],
])

# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_custom_traj(segments, t_arr):
    """
    Interpolate position and velocity reference from a dense waypoint array.

    If segments is an ndarray (M, 7): [t, x, y, z, vx, vy, vz]
        → smooth np.interp on position and velocity columns (same as test_time_air).

    If segments is a list of tuples (sparse, legacy):
        Each row: (t_start, x, y, z)              — hold
              or  (t_start, x, y, z, vx, vy, vz)  — constant-velocity segment
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


def _aerial_timed_waypoints():
    """
    Build time-stamped (position, velocity) waypoints for the aerial phase
    using exactly the same path geometry, flight mode, camera parameters,
    and computed velocities as Test_time.py (air_config section).

    The path coordinates reproduce the 3D visualisation from
    plot_inspection_route exactly; timing is then derived from the same
    velocity constraints Test_time uses for its duration estimates.

    z is normalised so z=0 corresponds to sea level (H_water is subtracted),
    matching the quadcopter sim's world frame.

    Returns
    -------
    wp : ndarray  shape (M, 7)
         columns: [t, x, y, z, vx, vy, vz]
    """
    from Test_time_functions import (
        get_w_arc, get_v_frame,
        get_velocity_rgb_lawnmower, get_velocity_rgb_spiral,
        get_velocity_event, get_velocity_event_spiral,
        get_velocity_hyper_lawnmower,
        get_lawnmower_coords, get_spiral_coords,
    )
    from Test_time import (
        R_base, R_top, H_water, H_air_cyl, H_air_cone,
        air_config, cameras,
    )

    H_total = H_water + H_air_cyl + H_air_cone

    ca    = cameras[air_config["camera_type"]]
    D     = ca["D"]
    fmode = air_config["flight_mode"]
    v_max = float(air_config["v_max"])

    v_frame_a = get_v_frame(D, ca["v_fov"]) if "v_fov" in ca else None
    w_arc_a   = get_w_arc(R_base, D, ca["h_fov"], label="tower")

    def get_radius(z):
        if z <= H_water + H_air_cyl:
            return R_base
        return R_base - ((R_base - R_top) / H_air_cone) * (z - (H_water + H_air_cyl))

    # ── Compute Test_time velocities for the air phase ────────────────────────
    if fmode == "lawnmower":
        if air_config["camera_type"] == "RGB":
            v_scan, _ = get_velocity_rgb_lawnmower(
                v_max, ca["gsd"], ca["max_blur"], ca["shutter"],
                v_frame_a, ca["v_overlap"], ca["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_scan, _ = get_velocity_event(v_max)
        elif air_config["camera_type"] == "HYPERSPECTRAL":
            v_scan, _ = get_velocity_hyper_lawnmower(
                v_max, ca["gsd"], ca["line_rate"], ca["integration"], ca["max_blur"])
        else:
            v_scan = v_max

        v_vert  = v_scan   # speed along vertical strips
        v_horiz = v_max    # speed of horizontal steps between strips

        w_arc_step = w_arc_a * (1.0 - ca["h_overlap"])
        za, ta = get_lawnmower_coords(H_water, H_total, R_base, w_arc_step)

    elif fmode == "spiral":
        pitch = v_frame_a * (1.0 - ca["v_overlap"]) if v_frame_a \
                else w_arc_a * (1 - ca["h_overlap"])
        if air_config["camera_type"] == "RGB":
            v_path, _ = get_velocity_rgb_spiral(
                v_max, R_base, pitch, w_arc_a,
                ca["gsd"], ca["max_blur"], ca["shutter"], ca["h_overlap"], ca["fps"])
        elif air_config["camera_type"] == "EVENT":
            v_path, _ = get_velocity_event_spiral(v_max)
        else:
            v_path = v_max

        v_vert  = v_path
        v_horiz = v_path

        za, ta = get_spiral_coords(H_water, H_total, pitch)

    else:
        raise ValueError(f"Unknown flight_mode: '{fmode}'")

    # ── Build cylindrical path (r, θ, z) directly — no Cartesian conversion ─────
    za = np.array(za)
    ta = np.array(ta)
    ra = np.array([get_radius(z) for z in za]) + D   # standoff radius
    z_norm = za - H_water                             # z=0 at sea level

    # Velocity direction from path tangent; classify vertical vs horizontal
    # Cartesian tangent only used for direction classification, not stored
    xyz_tmp = np.column_stack([ra * np.cos(ta), ra * np.sin(ta), z_norm])
    vr_arr  = np.zeros(len(za))
    vθ_arr  = np.zeros(len(za))
    vz_arr  = np.zeros(len(za))

    for i in range(len(za) - 1):
        dv   = xyz_tmp[i + 1] - xyz_tmp[i]
        dist = np.linalg.norm(dv)
        if dist < 1e-10:
            continue
        vert_frac = abs(dv[2]) / dist
        speed     = v_vert if vert_frac > 0.7 else v_horiz
        vx_i = (dv[0] / dist) * speed
        vy_i = (dv[1] / dist) * speed
        vz_i = (dv[2] / dist) * speed
        r_i  = max(ra[i], 1e-6)
        θ_i  = ta[i]
        vr_arr[i]  =  vx_i * np.cos(θ_i) + vy_i * np.sin(θ_i)
        vθ_arr[i]  = (-vx_i * np.sin(θ_i) + vy_i * np.cos(θ_i)) / r_i
        vz_arr[i]  = vz_i

    vr_arr[-1] = vr_arr[-2]
    vθ_arr[-1] = vθ_arr[-2]
    vz_arr[-1] = vz_arr[-2]

    # ── Timestamps from arc-length and speeds ─────────────────────────────────
    times = [0.0]
    for i in range(1, len(za)):
        dv   = xyz_tmp[i] - xyz_tmp[i - 1]
        dist = np.linalg.norm(dv)
        if dist < 1e-10:
            times.append(times[-1])
            continue
        vert_frac = abs(dv[2]) / dist
        speed     = v_vert if vert_frac > 0.7 else v_horiz
        times.append(times[-1] + dist / speed)

    times = np.array(times)
    return np.column_stack([times, ra, ta, z_norm, vr_arr, vθ_arr, vz_arr])


def build_test_time_aerial_traj(t_arr):
    """
    Interpolate aerial path position and velocity onto the sim time array.

    Returns
    -------
    ref_pos    : (3, N) position reference  [m]
    ref_vel    : (3, N) velocity reference  [m/s]  — used as feedforward
    t_duration : float  total path duration [s]
    start_xyz  : (3,)   first waypoint position, to initialise sim state
    """
    wp = _aerial_timed_waypoints()   # (M, 7): [t, r, θ, z, ṙ, θ̇, ż]
    ref_p = np.zeros((3, len(t_arr)))
    ref_v = np.zeros((3, len(t_arr)))
    for ax in range(3):
        ref_p[ax, :] = np.interp(t_arr, wp[:, 0], wp[:, 1 + ax])
        ref_v[ax, :] = np.interp(t_arr, wp[:, 0], wp[:, 4 + ax])
    # start position in Cartesian for state initialisation
    r0, θ0, z0 = wp[0, 1], wp[0, 2], wp[0, 3]
    start_xyz = np.array([r0 * np.cos(θ0), r0 * np.sin(θ0), z0])
    return ref_p, ref_v, float(wp[-1, 0]), start_xyz

# ══════════════════════════════════════════════════════════════════════════════
# DISTURBANCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_disturbance(t_k):
    """Return (Fd [3], taud [3]) summed over all active disturbance rows."""
    if not DIST_ENABLED:
        return np.zeros(3), np.zeros(3)
    Fd   = np.zeros(3)
    taud = np.zeros(3)
    for row in DISTURBANCES:
        if row[0] <= t_k < row[1]:
            Fd   += np.asarray(row[2:5], dtype=float)
            taud += np.asarray(row[5:8], dtype=float)
    return Fd, taud

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION SETUP
# ══════════════════════════════════════════════════════════════════════════════

dt = 0.005   # [s]  timestep (200 Hz)

# Determine simulation end time from all configured events
_t_events = [T_BUFFER]
if DIST_ENABLED and DISTURBANCES:
    _t_events.append(max(row[1] for row in DISTURBANCES))

if TRAJ_MODE == "custom" and len(TRAJ_SEGMENTS):
    _t_events.append(TRAJ_SEGMENTS[-1][0])
elif TRAJ_MODE == "test_time_air":
    _wp = _aerial_timed_waypoints()
    _t_events.append(float(_wp[-1, 0]))

t_end = max(_t_events) + T_BUFFER
t     = np.arange(0, t_end + dt, dt)
N     = len(t)

# Build reference trajectory
x0_override = None   # optionally place drone at first trajectory waypoint

if TRAJ_MODE == "hold":
    ref_pos = np.zeros((3, N))
    ref_vel = np.zeros((3, N))
    print("Trajectory mode : HOLD at origin")

elif TRAJ_MODE == "custom":
    ref_pos, ref_vel = build_custom_traj(TRAJ_SEGMENTS, t)
    # TRAJ_SEGMENTS is cylindrical (r,θ,z) — convert first waypoint to Cartesian for state init
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

ref_yaw = np.zeros(N)

# Cylindrical reference is used for custom (ndarray) and test_time_air modes
_USE_CYL_REF = TRAJ_MODE in ("custom", "test_time_air") and isinstance(
    TRAJ_SEGMENTS if TRAJ_MODE == "custom" else True, (np.ndarray, bool))

# Acceleration feedforward — includes centripetal terms when in cylindrical mode
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
    # Cartesian acceleration: includes centripetal (r·θ̇²) and Coriolis (2ṙ·θ̇) terms
    ref_acc = np.zeros((3, N))
    ref_acc[0] = (r̈_r - r_r*θ̇_r**2)*np.cos(θ_r) - (r_r*θ̈_r + 2*ṙ_r*θ̇_r)*np.sin(θ_r)
    ref_acc[1] = (r̈_r - r_r*θ̇_r**2)*np.sin(θ_r) + (r_r*θ̈_r + 2*ṙ_r*θ̇_r)*np.cos(θ_r)
    ref_acc[2] = z̈_r
else:
    ref_acc = np.gradient(ref_vel, dt, axis=1)

print(f"Disturbances    : {'ON' if DIST_ENABLED else 'OFF'}  "
      f"({len(DISTURBANCES)} row(s) defined)")
print(f"Plots           : {'ON' if ENABLE_PLOTS else 'OFF'}")

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

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


def quad_ode(s, wr_cmd, Fd, taud):
    """
    Nonlinear 6-DOF quadrotor ODE.
    s       : state vector (16,)
    wr_cmd  : commanded rotor speeds (4,)
    Fd      : disturbance force in inertial frame (3,)
    taud    : disturbance torque in body frame (3,)
    returns : ds/dt (16,)
    """
    euler = s[3:6];  phi, theta, psi = euler
    vel   = s[6:9]
    wb    = s[9:12]
    wr    = s[12:16]

    R = rot_ZYX(phi, theta, psi)

    T         = p.kT * np.sum(wr**2)
    tau_phi   = p.l * p.kT * (wr[3]**2 - wr[1]**2)
    tau_theta = p.l * p.kT * (wr[2]**2 - wr[0]**2)
    tau_psi   = p.kQ * (wr[0]**2 - wr[1]**2 + wr[2]**2 - wr[3]**2)
    tau_body  = np.array([tau_phi, tau_theta, tau_psi]) + taud

    Omega_net = wr[0] - wr[1] + wr[2] - wr[3]
    tau_gyro  = p.Jr * Omega_net * np.array([-wb[1], wb[0], 0.0])

    F_thrust  = R @ np.array([0, 0, T])
    F_grav    = np.array([0, 0, -p.m * p.g])
    F_drag    = -p.kd * vel
    pos_ddot  = (F_thrust + F_grav + F_drag + Fd) / p.m

    I      = np.array([[p.Ixx,  0.0,   p.Ixz],
                       [0.0,    p.Iyy, 0.0  ],
                       [p.Ixz,  0.0,   p.Izz]])
    wb_dot = np.linalg.solve(I, tau_body - tau_gyro - np.cross(wb, I @ wb))

    euler_dot = euler_kin(phi, theta) @ wb
    wr_dot    = (wr_cmd - wr) / p.tau_m

    return np.concatenate([vel, euler_dot, pos_ddot, wb_dot, wr_dot])


def rk4_step(s, wr_cmd, Fd, taud, dt):
    """Classic RK4 integrator step."""
    k1 = quad_ode(s,            wr_cmd, Fd, taud)
    k2 = quad_ode(s + dt/2*k1, wr_cmd, Fd, taud)
    k3 = quad_ode(s + dt/2*k2, wr_cmd, Fd, taud)
    k4 = quad_ode(s + dt*k3,   wr_cmd, Fd, taud)
    return s + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP  (skipped in root_locus mode)
# ══════════════════════════════════════════════════════════════════════════════

X      = np.zeros((16, N))
X[12:16, 0] = p.omega_h   # all rotors at hover speed

if x0_override is not None:
    X[0:3, 0] = x0_override   # start at first trajectory waypoint

int_att = np.zeros(3)
int_pos = np.zeros(3)

U_log  = np.zeros((4, N))   # [T, τ_φ, τ_θ, τ_ψ]
Wr_log = np.zeros((4, N))   # commanded rotor speeds

for k in (range(N - 1) if PLOT_MODE != "root_locus" else []):
    s     = X[:, k]
    pos   = s[0:3]
    euler = s[3:6];  phi, theta, psi = euler
    vel   = s[6:9]
    wb    = s[9:12]
    wr    = s[12:16]

    # ── Outer PID: position error + velocity feedforward → thrust + att cmd ──
    if _USE_CYL_REF:
        # ── Measured cylindrical state ──────────────────────────────────────
        r_m = max(np.sqrt(pos[0]**2 + pos[1]**2), 1e-6)
        θ_m = np.arctan2(pos[1], pos[0])
        cs  = np.cos(θ_m);  sn = np.sin(θ_m)

        # ── Exact cylindrical position errors ───────────────────────────────
        e_r = ref_pos[0, k] - r_m
        e_θ = np.arctan2(np.sin(ref_pos[1, k] - θ_m),
                         np.cos(ref_pos[1, k] - θ_m))
        e_z = ref_pos[2, k] - pos[2]
        # Arc-length tangential error [m] — consistent units with e_r
        e_t = r_m * e_θ

        # ── Exact cylindrical velocity errors ────────────────────────────────
        ṙ_m  =  vel[0] * cs + vel[1] * sn
        θ̇_m  = (-vel[0] * sn + vel[1] * cs) / r_m
        e_ṙ  = ref_vel[0, k] - ṙ_m
        e_ṫ  = r_m * (ref_vel[1, k] - θ̇_m)   # tangential velocity error [m/s]
        e_ż  = ref_vel[2, k] - vel[2]

        # ── Cylindrical integrators ──────────────────────────────────────────
        int_pos = np.clip(int_pos + np.array([e_r, e_t, e_z]) * dt,
                          -cyl_i_lim, cyl_i_lim)

        # ── ref_acc projected into cylindrical ──────────────────────────────
        ra_r = ref_acc[0, k] * cs + ref_acc[1, k] * sn
        ra_t = -ref_acc[0, k] * sn + ref_acc[1, k] * cs
        ra_z = ref_acc[2, k]

        # ── Cylindrical PID → acceleration commands ─────────────────────────
        a_r = cyl_Kp[0]*e_r + cyl_Ki[0]*int_pos[0] + cyl_Kd[0]*e_ṙ + ra_r
        a_t = cyl_Kp[1]*e_t + cyl_Ki[1]*int_pos[1] + cyl_Kd[1]*e_ṫ + ra_t
        a_z = cyl_Kp[2]*e_z + cyl_Ki[2]*int_pos[2] + cyl_Kd[2]*e_ż + ra_z

        # ── Exact cylindrical → Cartesian acceleration (no approximation) ───
        a_cmd = np.array([a_r * cs - a_t * sn,
                          a_r * sn + a_t * cs,
                          a_z])
    else:
        # Cartesian PID (hold mode)
        e_pos = ref_pos[:, k] - pos
        e_vel = ref_vel[:, k] - vel
        int_pos = np.clip(int_pos + e_pos * dt, -pos_i_lim, pos_i_lim)
        a_cmd = pos_Kp * e_pos + pos_Ki * int_pos + pos_Kd * e_vel + ref_acc[:, k]

    T_cmd = max(p.m * (a_cmd[2] + p.g), 0.1 * p.m * p.g)

    # x,y acceleration → desired pitch/roll (small angle, yaw-rotated)
    theta_d = ( a_cmd[0]*np.cos(psi) + a_cmd[1]*np.sin(psi)) * p.m / T_cmd
    phi_d   = ( a_cmd[0]*np.sin(psi) - a_cmd[1]*np.cos(psi)) * p.m / T_cmd
    psi_d   = ref_yaw[k]

    theta_d = np.clip(theta_d, -att_lim, att_lim)
    phi_d   = np.clip(phi_d,   -att_lim, att_lim)

    # ── Inner PID: attitude error → body torques ────────────────────────────
    e_att    = np.array([phi_d, theta_d, psi_d]) - euler
    e_att[2] = np.arctan2(np.sin(e_att[2]), np.cos(e_att[2]))   # yaw wrap ±π
    int_att  = np.clip(int_att + e_att * dt, -att_i_lim, att_i_lim)

    # Derivative on measured body rate — avoids derivative kick on ref step
    tau_cmd = att_Kp * e_att + att_Ki * int_att - att_Kd * wb

    u_virt      = np.array([T_cmd, tau_cmd[0], tau_cmd[1], tau_cmd[2]])
    U_log[:, k] = u_virt

    # ── Mixing: virtual inputs → rotor speed commands ──────────────────────
    wr_sq_cmd = np.linalg.solve(A_mix, u_virt)
    wr_sq_cmd = np.maximum(wr_sq_cmd, 0.0)
    wr_cmd    = np.minimum(np.sqrt(wr_sq_cmd), p.omega_max)
    Wr_log[:, k] = wr_cmd

    # ── Disturbance ────────────────────────────────────────────────────────
    Fd, taud = get_disturbance(t[k])

    # ── RK4 integrate ──────────────────────────────────────────────────────
    X[:, k+1] = rk4_step(s, wr_cmd, Fd, taud, dt)

U_log[:, -1] = U_log[:, -2]

# ══════════════════════════════════════════════════════════════════════════════
# LINEARISATION around hover  →  full 16-state model (includes motor dynamics)
#
# State:  [x y z  phi theta psi  xd yd zd  p q r  w1 w2 w3 w4]   (16)
# Input:  [w1_cmd  w2_cmd  w3_cmd  w4_cmd]                         (4)
#
# Numerically differentiates quad_ode() directly — consistent with simulation.
# ══════════════════════════════════════════════════════════════════════════════

_s0 = np.zeros(16);  _s0[12:16] = p.omega_h   # hover: rotors at omega_h
_u0 = np.full(4, p.omega_h)                    # commanded = actual at hover
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

# ── Closed-loop gain matrix (PD only — Ki requires augmented integrator states) ──
#
# Linearised PD control law around hover (psi=0, T=m*g):
#   outer: a_cmd = -Kp_pos*pos - Kd_pos*vel
#          theta_d =  a_cmd[0]/g,  phi_d = -a_cmd[1]/g,  T_cmd = m*(a_cmd[2]+g)
#   inner: tau = -Kp_att*att - Kd_att*wb  (+ Kp_att*att_d from outer)
#   mix:   dw_cmd = (1/(2*wh)) * A_mix_inv @ [dT, dtau_phi, dtau_theta, dtau_psi]

def build_K_cl(pKp, pKi, pKd, aKp, aKi, aKd):
    """
    Returns K_pid (4x16): linearised gain from full state → rotor speed commands.
    pKi / aKi accepted for API consistency but not used (no integrator states).
    A_cl = A_lin + B_lin @ K_pid

    Cylindrical PID note: at hover θ=0, cylindrical coords align with Cartesian
    (r=x, tangential=y, z=z), so the linearised cylindrical PID is identical to
    a Cartesian PID with pKp = cyl_Kp, pKd = cyl_Kd. Pass cyl gains here.
    """
    g  = p.g;   m  = p.m;   wh = p.omega_h
    # Virtual-input gain (4x16): u_virt = K_virt @ state
    Kv = np.zeros((4, 16))
    # Thrust ← z, ż
    Kv[0, 2]  = -m * pKp[2]
    Kv[0, 8]  = -m * pKd[2]
    # tau_phi ← y→phi_d then phi, p
    Kv[1, 1]  =  aKp[0] * pKp[1] / g
    Kv[1, 7]  =  aKp[0] * pKd[1] / g
    Kv[1, 3]  = -aKp[0]
    Kv[1, 9]  = -aKd[0]
    # tau_theta ← x→theta_d then theta, q
    Kv[2, 0]  = -aKp[1] * pKp[0] / g
    Kv[2, 6]  = -aKp[1] * pKd[0] / g
    Kv[2, 4]  = -aKp[1]
    Kv[2, 10] = -aKd[1]
    # tau_psi ← psi, r
    Kv[3, 5]  = -aKp[2]
    Kv[3, 11] = -aKd[2]
    # Linearise mixing: dw_cmd = (1/(2*wh)) * A_mix_inv @ du_virt
    return (1.0 / (2.0 * wh)) * np.linalg.inv(A_mix) @ Kv

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

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

# ── Convert cylindrical ref to Cartesian for plotting and error analysis ─────
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

# ── Tracking error summary (sim mode only) ────────────────────────────────────
if PLOT_MODE == "sim":
    pos_err_vec = X[0:3, :] - ref_pos_cart
    vel_err_vec = X[6:9, :] - ref_vel_cart
    pos_err_mag = np.linalg.norm(pos_err_vec, axis=0)
    vel_err_mag = np.linalg.norm(vel_err_vec, axis=0)
    print(f"\n{'═'*48}")
    print("TRACKING ERROR SUMMARY")
    print(f"{'═'*48}")
    print(f"  Position  mean={pos_err_mag.mean():.3f} m    max={pos_err_mag.max():.3f} m")
    print(f"  Velocity  mean={vel_err_mag.mean():.3f} m/s  max={vel_err_mag.max():.3f} m/s")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

if PLOT_MODE not in ("sim", "root_locus"):
    print("\nPlots suppressed (PLOT_MODE not recognised)")

elif PLOT_MODE == "root_locus":
    from matplotlib.widgets import Slider

    # Mutable gain copies — outer loop uses cyl gains (consistent with sim)
    # At hover θ=0: cylindrical ≡ Cartesian, so cyl_Kp/Kd plug directly into build_K_cl
    _pKp = cyl_Kp.copy().astype(float)
    _pKd = cyl_Kd.copy().astype(float)
    _aKp = att_Kp.copy().astype(float)
    _aKd = att_Kd.copy().astype(float)
    _z3  = np.zeros(3)

    # ── Pole computers ────────────────────────────────────────────────────────
    def _inner_poles():
        """Close attitude loop only (pos gains = 0). Plant = A_lin."""
        K = build_K_cl(_z3, _z3, _z3, _aKp, _z3, _aKd)
        return np.linalg.eigvals(A_lin + B_lin @ K)

    def _A_inner_cl():
        K = build_K_cl(_z3, _z3, _z3, _aKp, _z3, _aKd)
        return A_lin + B_lin @ K

    def _full_poles():
        """Close both loops. Plant for outer = A_inner_cl."""
        K = build_K_cl(_pKp, _z3, _pKd, _aKp, _z3, _aKd)
        return np.linalg.eigvals(A_lin + B_lin @ K)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_rl = plt.figure(figsize=(20, 13))
    fig_rl.suptitle(
        f"Sequential closed-loop pole analysis  —  gains: '{_flight_mode}'  |  Ki omitted",
        fontsize=12, fontweight='bold')

    # Two pole plots side by side, upper 55%
    ax_in  = fig_rl.add_axes([0.06, 0.42, 0.40, 0.50])   # left  — inner loop
    ax_out = fig_rl.add_axes([0.55, 0.42, 0.40, 0.50])   # right — outer loop

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

    # ── Initial scatter artists ───────────────────────────────────────────────
    ol_poles = np.linalg.eigvals(A_lin)

    # Inner plot: open-loop plant (grey) + inner closed (blue)
    ax_in.scatter(ol_poles.real, ol_poles.imag,
                  marker='o', s=50, color='lightgray', edgecolors='gray',
                  lw=1, label='Plant (open-loop)', zorder=3)
    ip0    = _inner_poles()
    sc_in  = ax_in.scatter(ip0.real, ip0.imag,
                           marker='x', s=130, lw=2.5,
                           color='darkorange', label='Attitude closed-loop', zorder=5)
    ax_in.legend(loc='upper right', fontsize=8)

    # Outer plot: inner-closed plant (grey) + full closed (blue)
    icp0   = np.linalg.eigvals(_A_inner_cl())
    sc_ref = ax_out.scatter(icp0.real, icp0.imag,
                            marker='o', s=50, color='lightgray', edgecolors='gray',
                            lw=1, label='Plant (inner loop closed)', zorder=3)
    fp0    = _full_poles()
    sc_out = ax_out.scatter(fp0.real, fp0.imag,
                            marker='x', s=130, lw=2.5,
                            color='royalblue', label='Full closed-loop', zorder=5)
    ax_out.legend(loc='upper right', fontsize=8)

    # ── Refresh helpers ───────────────────────────────────────────────────────
    def _lims(ax, *pole_sets):
        r  = np.concatenate([p.real for p in pole_sets])
        im = np.concatenate([p.imag for p in pole_sets])
        pr = max(abs(r).max()  * 0.15, 1.0)
        pi = max(abs(im).max() * 0.15, 1.0)
        ax.set_xlim(r.min()  - pr, max(r.max()  + pr, 0.5))
        ax.set_ylim(im.min() - pi, im.max() + pi)

    def _refresh_inner():
        ip = _inner_poles()
        sc_in.set_offsets(np.column_stack([ip.real, ip.imag]))
        _lims(ax_in, ol_poles, ip)
        # Outer plot reference (grey) also updates when att gains change
        icp = np.linalg.eigvals(_A_inner_cl())
        sc_ref.set_offsets(np.column_stack([icp.real, icp.imag]))
        fp = _full_poles()
        sc_out.set_offsets(np.column_stack([fp.real, fp.imag]))
        _lims(ax_out, icp, fp)
        fig_rl.canvas.draw_idle()

    def _refresh_outer():
        fp = _full_poles()
        sc_out.set_offsets(np.column_stack([fp.real, fp.imag]))
        icp = np.linalg.eigvals(_A_inner_cl())
        _lims(ax_out, icp, fp)
        fig_rl.canvas.draw_idle()

    _refresh_inner()

    # ── Sliders ───────────────────────────────────────────────────────────────
    # Layout: 2 rows (Kp top, Kd bottom) × 3 columns per side
    # Left side (x=0.06): att sliders (phi, theta, psi)
    # Right side (x=0.55): pos sliders (x, y, z)
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
                vmax  = max(v0 * 4.0, 2.0)
                sl    = Slider(ax_sl, lbls[ci], 0.0, vmax,
                               valinit=v0, color=cols[ci])
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

    # ── Callbacks ─────────────────────────────────────────────────────────────
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
    # Decimate to at most 10 000 points so matplotlib renders quickly regardless
    # of simulation length (test_time_air can produce 180 000+ steps).
    _ps   = max(1, N // 10_000)
    t_p   = t[::_ps]
    X_p   = X[:, ::_ps]
    rp_p  = ref_pos_cart[:, ::_ps]
    rv_p  = ref_vel_cart[:, ::_ps]
    Ul_p  = U_log[:, ::_ps]

    # ── Cylindrical actual state (for cyl-mode plots) ────────────────────────
    if _USE_CYL_REF:
        _r_p  = np.maximum(np.sqrt(X_p[0,:]**2 + X_p[1,:]**2), 1e-6)
        _θ_p  = np.arctan2(X_p[1,:], X_p[0,:])
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

    # ── Figure 1: Position + Attitude ─────────────────────────────────────────
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
        ax.set_ylabel(pos_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0:
            ax.set_title("Position (cylindrical)" if _USE_CYL_REF else "Position")
            ax.legend(loc='lower right')

        ax = axes1[i, 1]
        ax.plot(t_p, np.degrees(X_p[3+i, :]), 'b', lw=1.6)
        ax.set_ylabel(att_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0:
            ax.set_title("Attitude")

    axes1[2, 0].set_xlabel("Time  [s]")
    axes1[2, 1].set_xlabel("Time  [s]")
    fig1.tight_layout()

    # ── Figure 2: Velocity tracking ───────────────────────────────────────────
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
    fig2.tight_layout()

    # ── Figure 2b: Cylindrical tracking errors (only in cylindrical mode) ────
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
        fig2b.tight_layout()

    # ── Figure 3: Control Inputs ───────────────────────────────────────────────
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
    fig3.tight_layout()

    # ── Figure 4: Rotor Speeds ────────────────────────────────────────────────
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
    fig4.tight_layout()

    # ── Figure 5: Open-loop pole map ──────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(7, 6))
    poles = np.linalg.eigvals(A_lin)
    ax5.scatter(poles.real, poles.imag, marker='x', s=80, color='b', zorder=5)
    ax5.axvline(0, color='k', lw=0.8, ls='--')
    ax5.axhline(0, color='k', lw=0.8, ls='--')
    ax5.set_xlabel("Real"); ax5.set_ylabel("Imaginary")
    ax5.set_title("Open-Loop Poles at Hover (linearised)")
    ax5.grid(True)
    fig5.tight_layout()

    # ── Figure 6: 3D flight path — reference vs actual ───────────────────────
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
        ax6.plot(rp_p[0, :], rp_p[1, :], rp_p[2, :],
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
        all_x = np.concatenate([rp_p[0, :], X_p[0, :]])
        all_y = np.concatenate([rp_p[1, :], X_p[1, :]])
        all_z = np.concatenate([rp_p[2, :], X_p[2, :]])
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

    plt.show()
