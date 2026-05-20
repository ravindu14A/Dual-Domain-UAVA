"""
quadcopterUW.py  —  UAUV Underwater Phase: Stability & Control
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

6-DOF nonlinear underwater model + cascaded PID.
COMPLETELY SEPARATE from quadcopterSC.py — no shared code, no risk of breaking aerial sim.

STATE (16):  [x  y  z | phi  theta  psi | xd  yd  zd | p  q  r | w1  w2  w3  w4]
              pos(3)    euler(3)           vel(3)        ang_rate(3) vert_props(4)

z CONVENTION: z=0 at water surface, NEGATIVE downward (z=-60 = 60 m depth)

ACTUATORS:
  Vertical props (4): aerial arms folded, sit on top of box.
    Bidirectional: positive spin = downward thrust (fights buoyancy),
                   negative spin = upward thrust.
    Control: Fz, tau_phi, tau_theta.

  Horizontal thrusters (4): dedicated UW thrusters at box corners, z=CoM height.
    Angle alpha from box wall toward interior. 4 thrusters for Fx, Fy, tau_psi.
    Modelled as instantaneous force (no motor lag states).

CONTROL LAW:
  Outer (cylindrical PID + feedforward) -> acceleration commands a_cmd
  Direct force allocation (no attitude tilt for horizontal motion):
    Fx_cmd = m * a_cmd[0]
    Fy_cmd = m * a_cmd[1]
    Fz_v_cmd = UW_BALLAST_RESIDUAL - m * a_cmd[2]  (cancel buoyancy + control z)
  Inner (attitude PID): keeps phi=0, theta=0, tracks psi reference
    tau_phi, tau_theta -> vertical props
    tau_psi            -> horizontal thrusters

Dependencies: numpy matplotlib control tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
import control
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Imports from shared modules ───────────────────────────────────────────────
from geometry import (
    L_box, W_box, H_box, m_box,
    L_arm, m_arm, m_motor,
    arm_angles_deg, CoM,
    total_mass as _geo_mass,
    Ixx_uw as _geo_Ixx_uw,
    Iyy_uw as _geo_Iyy_uw,
    Izz_uw as _geo_Izz_uw,
    UW_ARM_FOLD_FRAC, UW_L_fold,
)
from Test_time import (
    water_config, cameras,
    R_base, H_water,
)
from Test_time_functions import (
    get_w_arc, get_v_frame,
    get_velocity_rgb_lawnmower, get_velocity_event,
    get_velocity_hyper_lawnmower,
)

# ── Trajectory helper functions (mirrored from quadcopterSC.py) ───────────────

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

def make_times(pts, speeds, t_start):
    times = [t_start]
    for i in range(1, len(pts)):
        dist  = np.linalg.norm(pts[i] - pts[i-1])
        v_avg = (speeds[i-1] + speeds[i]) / 2
        times.append(times[-1] + dist / max(v_avg, 1e-3))
    return np.array(times)


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION CONFIGURATION  —  edit only in this section
# ══════════════════════════════════════════════════════════════════════════════

TRAJ_MODE = "test_time_water"
#   "hold"             : hold at origin underwater
#   "custom"           : user-defined segments (Cartesian, same format as quadcopterSC)
#   "test_time_water"  : full underwater lawnmower from water_config

TRAJ_SEGMENTS = [
    (0.0,  0.0, 0.0, 0.0),
    (5.0,  6.0, 0.0, 0.0),
]   # used only when TRAJ_MODE = "custom"

TRAJ_ACCEL_MAX = 2.0    # [m/s²] trapezoidal ramp acceleration underwater (slower than aerial)

DIST_ENABLED = False
DISTURBANCES = [
    (20.0, 22.0,  3.0, 0.0, 0.0,  0.0, 0.0, 0.0),
]

T_BUFFER = 4.0

PLOT_MODE      = "sim"   # "sim" | "root_locus"
PLOT_REFERENCE = True
PLOT_ACTUAL    = True
ENABLE_PLOTS   = PLOT_MODE in ("sim", "root_locus")

# ── Underwater geometry parameters ───────────────────────────────────────────
# Horizontal thruster position (FR corner x,y in body frame; others mirrored)
UW_HORIZ_XY_FR = (L_box / 2.0, -W_box / 2.0)   # (m, m)
UW_HORIZ_ALPHA = 45.0   # [deg] from box wall surface toward interior

# Ballast compensation percentage
#   100 = neutrally buoyant (ballast perfectly cancels gross net buoyancy, residual = 0 N)
#   0   = no flooding / worst case (full gross net buoyancy remains as residual)
BUOYANCY_COMP_PCT = 95   # [%]  tune this; UW_BALLAST_RESIDUAL computed below after uw init

# Depth intervals for ballast steps (informational — props handle continuous control)
UW_BALLAST_DEPTH_INTERVALS = [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0]   # [m]

# Root locus linearisation depth
UW_LINEARISE_DEPTH = -20.0   # [m]


# ══════════════════════════════════════════════════════════════════════════════
# VEHICLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

class Params:
    """Rigid-body parameters for underwater configuration (folded arms)."""
    m   = _geo_mass    # total mass unchanged
    Ixx = _geo_Ixx_uw  # roll inertia — recalculated for folded arms
    Iyy = _geo_Iyy_uw  # pitch inertia — recalculated for folded arms
    Izz = _geo_Izz_uw  # yaw inertia — recalculated for folded arms
    Ixz = 0.0
    g   = 9.81

p = Params()

class UWParams:
    """Underwater-specific thruster and hydrodynamic parameters."""
    # Vertical props — same coefficients as aerial motors
    kT_v    = 5.0e-4    # [N·s²/rad²] thrust coeff
    kQ_v    = 1.0e-5    # [N·m·s²/rad²] torque coeff
    tau_m_v = 0.06      # [s] motor lag
    omega_max_v = 700.0 # [rad/s] saturation

    # Horizontal thrusters (larger, dedicated UW units — placeholder values)
    kT_h    = 5.0e-3    # [N·s²/rad²] thrust coeff (not used in force-command model)
    F_h_max = 200.0     # [N] per-thruster force saturation

    # Hydrodynamics
    kd_uw   = np.array([25.0, 35.0, 20.0])  # [N·s/m] directional drag [x, y, z]
    rho_w   = 1025.0    # [kg/m³] seawater density
    V_sub   = L_box * W_box * H_box          # [m³] submerged volume

    @property
    def F_buoyancy(self):
        return self.rho_w * self.V_sub * p.g

    @property
    def F_buoy_gross_net(self):
        return self.F_buoyancy - p.m * p.g

    @property
    def omega_v_eq(self):
        """Vertical prop equilibrium speed to cancel ballast residual."""
        return np.sqrt(max(UW_BALLAST_RESIDUAL / (4.0 * self.kT_v), 0.0))

uw = UWParams()

# Derive residual from compensation percentage (evaluated after uw so F_buoy_gross_net is known)
UW_BALLAST_RESIDUAL = (1.0 - BUOYANCY_COMP_PCT / 100.0) * uw.F_buoy_gross_net

print("=" * 58)
print("  UNDERWATER VEHICLE PARAMETERS")
print("=" * 58)
print(f"  {'m':<8}  Total mass              {p.m:.4f}   kg")
print(f"  {'Ixx':<8}  Roll inertia            {p.Ixx:.4f}   kg·m²")
print(f"  {'Iyy':<8}  Pitch inertia           {p.Iyy:.4f}   kg·m²")
print(f"  {'Izz':<8}  Yaw inertia             {p.Izz:.4f}   kg·m²")
print(f"  {'kT_v':<8}  Vert prop thrust coeff  {uw.kT_v:.2e}  N·s²/rad²")
print(f"  {'kQ_v':<8}  Vert prop torque coeff  {uw.kQ_v:.2e}  N·m·s²/rad²")
print(f"  {'tau_m_v':<8}  Vert prop motor lag     {uw.tau_m_v:.4f}   s")
print(f"  {'kT_h':<8}  Horiz thruster coeff    {uw.kT_h:.2e}  N·s²/rad²")
print(f"  {'F_buoy':<8}  Gross buoyancy          {uw.F_buoyancy:.2f}   N")
print(f"  {'F_net':<8}  Gross net buoyancy      {uw.F_buoy_gross_net:.2f}   N")
print(f"  {'Comp%':<8}  Ballast compensation    {BUOYANCY_COMP_PCT:.2f}   %")
print(f"  {'Residual':<8}  Ballast residual        {UW_BALLAST_RESIDUAL:.2f}   N (upward)")
print(f"  {'w_v_eq':<8}  Vert prop eq speed      {-uw.omega_v_eq:.2f}   rad/s  ({uw.omega_v_eq*60/(2*np.pi):.0f} RPM)  (negative = down)")
print("=" * 58)


# ══════════════════════════════════════════════════════════════════════════════
# PID GAINS
# ══════════════════════════════════════════════════════════════════════════════

_gains_uw = {
    # att: inner attitude loop
    #   phi/theta: Ixx_uw=4.04, Iyy_uw=9.64 kg·m²  → wn=3 rad/s, ζ=0.9
    #              Kp = I*wn²,  Kd = 2*ζ*wn*I
    #   psi:       Izz_uw=11.91 kg·m²               → wn=2 rad/s, ζ=0.9
    # cyl: outer position loop (drag cancelled — see force commands below)
    #   After drag feedforward the plant is m*x''=Fx_cmd, so
    #   Kd targets the PD pole directly: wn=sqrt(Kp), ζ=Kd/(2*wn)
    "hold": dict(
        att_Kp    = np.array([36.0,  87.0,  48.0]),
        att_Ki    = np.array([0.2,   0.2,   0.1 ]),
        att_Kd    = np.array([22.0,  52.0,  43.0]),
        att_i_lim = np.array([10.0,  10.0,  5.0 ]),
        att_lim   = 0.30,
        cyl_Kp    = np.array([0.30, 0.30, 0.20]),   # wn ≈ 0.55, 0.55, 0.45
        cyl_Ki    = np.array([0.01, 0.01, 0.02]),
        cyl_Kd    = np.array([1.00, 1.00, 0.80]),   # ζ ≈ 0.9
        cyl_i_lim = np.array([5.0,  5.0,  10.0]),
    ),
    "custom": dict(
        att_Kp    = np.array([36.0,  87.0,  48.0]),
        att_Ki    = np.array([0.2,   0.2,   0.1 ]),
        att_Kd    = np.array([22.0,  52.0,  43.0]),
        att_i_lim = np.array([10.0,  10.0,  5.0 ]),
        att_lim   = 0.30,
        cyl_Kp    = np.array([0.60, 0.60, 0.30]),   # wn ≈ 0.77, 0.77, 0.55
        cyl_Ki    = np.array([0.02, 0.02, 0.05]),
        cyl_Kd    = np.array([1.40, 1.40, 0.90]),   # ζ ≈ 0.9
        cyl_i_lim = np.array([5.0,  5.0,  10.0]),
    ),
    "lawnmower": dict(
        att_Kp    = np.array([36.0,  87.0,  48.0]),
        att_Ki    = np.array([0.2,   0.2,   0.1 ]),
        att_Kd    = np.array([22.0,  52.0,  43.0]),
        att_i_lim = np.array([10.0,  10.0,  5.0 ]),
        att_lim   = 0.30,
        cyl_Kp    = np.array([1.00, 1.00, 0.50]),   # wn ≈ 1.0, 1.0, 0.71
        cyl_Ki    = np.array([0.02, 0.02, 0.05]),
        cyl_Kd    = np.array([1.80, 1.80, 1.30]),   # ζ ≈ 0.9
        cyl_i_lim = np.array([5.0,  5.0,  10.0]),
    ),
    "spiral": dict(
        att_Kp    = np.array([36.0,  87.0,  48.0]),
        att_Ki    = np.array([0.2,   0.2,   0.1 ]),
        att_Kd    = np.array([22.0,  52.0,  43.0]),
        att_i_lim = np.array([10.0,  10.0,  5.0 ]),
        att_lim   = 0.30,
        cyl_Kp    = np.array([0.80, 0.80, 0.40]),   # wn ≈ 0.89, 0.89, 0.63
        cyl_Ki    = np.array([0.02, 0.02, 0.05]),
        cyl_Kd    = np.array([1.60, 1.60, 1.10]),   # ζ ≈ 0.9
        cyl_i_lim = np.array([5.0,  5.0,  10.0]),
    ),
}

_flight_mode = water_config["flight_mode"] if TRAJ_MODE == "test_time_water" else TRAJ_MODE
_g = _gains_uw.get(_flight_mode, _gains_uw["lawnmower"])

att_Kp    = _g["att_Kp"];  att_Ki    = _g["att_Ki"]
att_Kd    = _g["att_Kd"];  att_i_lim = _g["att_i_lim"]
att_lim   = _g["att_lim"]
cyl_Kp    = _g["cyl_Kp"];  cyl_Ki    = _g["cyl_Ki"]
cyl_Kd    = _g["cyl_Kd"];  cyl_i_lim = _g["cyl_i_lim"]

print(f"PID gains       : {_flight_mode} (UW)")


# ══════════════════════════════════════════════════════════════════════════════
# MIXING MATRICES
# ══════════════════════════════════════════════════════════════════════════════

# ── Vertical props ────────────────────────────────────────────────────────────
# Folded arm length imported from geometry.py: UW_L_fold = (2*frac-1)*L_arm

# Prop positions in body CoM frame along arm directions
_arm_rad = np.radians(arm_angles_deg)
_vert_pos = np.array([
    [UW_L_fold * np.cos(a),
     UW_L_fold * np.sin(a),
     H_box / 2.0 - CoM[2]]   # top of box, shifted to CoM frame
    for a in _arm_rad
])   # shape (4, 3)

_lv_x = _vert_pos[:, 0]   # x positions of each vertical prop
_lv_y = _vert_pos[:, 1]   # y positions of each vertical prop

# Spin directions: diagonal CCW/CW pairs matching aerial X-config
# arm_angles_deg = [45, 135, 225, 315] → motor 0=FL, 1=RL, 2=RR, 3=FR
# Diagonal pair A: FL(0) + RR(2) at 45° and 225° → CCW (+1)
# Diagonal pair B: RL(1) + FR(3) at 135° and 315° → CW  (-1)
_spin_dir = np.array([+1.0, -1.0, +1.0, -1.0])   # diagonal pairs — alternating

# A_mix_vert maps signed squared speeds [q1, q2, q3, q4] to [Fz_v, tau_phi, tau_theta, tau_psi_v]
# CONVENTION: positive wi → upward thrust (+z_body), same as aerial.
# At equilibrium, props spin at negative speed to push DOWN and cancel residual buoyancy.
#
# For upward force F = +kT_v*qi in body-z at position r_i = (lv_x_i, lv_y_i, lv_z_i):
#   tau_x (phi)   =  r_iy * F_z = +lv_y_i * kT_v * qi
#   tau_y (theta) = -r_ix * F_z = -lv_x_i * kT_v * qi
#   tau_z (psi)   = kQ_v * spin_dir_i * qi  (reaction torque)
A_mix_vert = np.array([
    [ uw.kT_v] * 4,                                    # Fz_v (positive = upward)
    [+uw.kT_v * _lv_y[i] for i in range(4)],           # tau_phi
    [-uw.kT_v * _lv_x[i] for i in range(4)],           # tau_theta
    [ uw.kQ_v * _spin_dir[i] for i in range(4)],       # tau_psi_v
])

A_mix_vert_inv = np.linalg.inv(A_mix_vert)

print(f"Vertical prop L_fold = {UW_L_fold:.3f} m  (UW_ARM_FOLD_FRAC={UW_ARM_FOLD_FRAC})")
print(f"A_mix_vert condition number: {np.linalg.cond(A_mix_vert):.1f}")

# ── Horizontal thrusters ──────────────────────────────────────────────────────
# Positions: FR corner + mirrors. z = 0 (body CoM height) always.
_px = UW_HORIZ_XY_FR[0]
_py = abs(UW_HORIZ_XY_FR[1])
_a  = np.radians(UW_HORIZ_ALPHA)

# Thruster positions in body frame: [FR, FL, RL, RR]
_horiz_pos = np.array([
    [ _px, -_py, 0.0],   # FR
    [ _px,  _py, 0.0],   # FL
    [-_px,  _py, 0.0],   # RL
    [-_px, -_py, 0.0],   # RR
])

# Thrust directions (angle from wall toward interior):
_horiz_dir = np.array([
    [+np.cos(_a), +np.sin(_a), 0.0],   # FR: forward + leftward (inward from right wall)
    [+np.cos(_a), -np.sin(_a), 0.0],   # FL: forward + rightward (inward from left wall)
    [-np.cos(_a), -np.sin(_a), 0.0],   # RL: backward + rightward
    [-np.cos(_a), +np.sin(_a), 0.0],   # RR: backward + leftward
])

# Yaw moment arm: z-component of r_i × d_i
_arm_yaw = _px * np.sin(_a) + _py * np.cos(_a)

# A_mix_horiz (3x4): maps [Fh_FR, Fh_FL, Fh_RL, Fh_RR] to [Fx, Fy, Mz]
A_mix_horiz = np.array([
    [+np.cos(_a), +np.cos(_a), -np.cos(_a), -np.cos(_a)],   # Fx
    [+np.sin(_a), -np.sin(_a), -np.sin(_a), +np.sin(_a)],   # Fy
    [+_arm_yaw,   -_arm_yaw,   +_arm_yaw,   -_arm_yaw  ],   # Mz (yaw)
])
A_mix_horiz_pinv = np.linalg.pinv(A_mix_horiz)

print(f"Horizontal thruster alpha = {UW_HORIZ_ALPHA} deg,  arm_yaw = {_arm_yaw:.3f} m")
print(f"A_mix_horiz condition number: {np.linalg.cond(A_mix_horiz):.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_custom_traj(segments, t_arr):
    """Same sparse/dense segment format as quadcopterSC — Cartesian (x,y,z)."""
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
        t1   = segments[i + 1][0] if i < len(segments) - 1 else t_arr[-1] + 1
        mask = (t_arr >= t0) & (t_arr < t1)
        dt   = t_arr[mask] - t0
        ref_p[0, mask] = x0 + vx * dt
        ref_p[1, mask] = y0 + vy * dt
        ref_p[2, mask] = z0 + vz * dt
        ref_v[0, mask] = vx
        ref_v[1, mask] = vy
        ref_v[2, mask] = vz
    return ref_p, ref_v


def _water_timed_waypoints():
    """
    Build time-stamped (r, theta, z, vr, vtheta, vz) waypoints for the
    underwater lawnmower or spiral inspection of the monopile.

    Uses IDENTICAL approach to _aerial_timed_waypoints() in quadcopterSC.py:
      1. Generate all path waypoints (r, theta, z) for full path
      2. Convert to Cartesian for 3D distance / gap classification
      3. Classify each gap: 0=vertical strip, 1=horizontal turn (same threshold 0.7)
      4. Apply trap_speeds to each contiguous same-type segment group
      5. Project velocity components from path tangent x speed magnitude
      6. Build timestamps via trapezoidal integration over speed profile

    z=0 at surface, negative downward. Monopile is a constant-radius cylinder.
    Returns ndarray (M, 7): [t, r, theta, z, vr, vtheta, vz]
    """
    cw      = cameras[water_config["camera_type"]]
    D       = cw["D"]
    v_max   = float(water_config["v_max"])
    v_horiz = float(water_config.get("v_horiz", v_max))
    fmode   = water_config["flight_mode"]

    v_frame_w = get_v_frame(D, cw["v_fov"]) if "v_fov" in cw else None
    w_arc_w   = get_w_arc(R_base, D, cw["h_fov"], label="monopile UW")
    r_inspect = R_base + D

    if fmode == "lawnmower":
        if water_config["camera_type"] == "RGB":
            v_scan, _ = get_velocity_rgb_lawnmower(
                v_max, cw["gsd"], cw["max_blur"], cw["shutter"],
                v_frame_w, cw["v_overlap"], cw["fps"])
        elif water_config["camera_type"] == "HYPERSPECTRAL":
            v_scan, _ = get_velocity_hyper_lawnmower(
                v_max, cw["gsd"], cw["line_rate"], cw["integration"], cw["max_blur"])
        else:
            v_scan = v_max

        strip_w  = w_arc_w * (1.0 - cw["h_overlap"])
        n_strips = int(np.ceil(2.0 * np.pi * R_base / strip_w))
        d_theta  = (2.0 * np.pi) / n_strips

        # ── Step 1: generate all (r, theta, z) waypoints for full path ────────
        ra_list, ta_list, za_list = [], [], []
        theta = 0.0
        for s_idx in range(n_strips):
            z_start = 0.0      if (s_idx % 2 == 0) else -H_water
            z_end   = -H_water if (s_idx % 2 == 0) else 0.0
            ra_list.extend([r_inspect] * 30)
            ta_list.extend([theta] * 30)
            za_list.extend(np.linspace(z_start, z_end, 30))
            if s_idx < n_strips - 1:
                theta_next = theta + d_theta
                ra_list.extend([r_inspect] * 30)
                ta_list.extend(np.linspace(theta, theta_next, 30))
                za_list.extend([z_end] * 30)
                theta = theta_next

        ra = np.array(ra_list)
        ta = np.array(ta_list)
        za = np.array(za_list)

    else:  # spiral
        from Test_time_functions import get_velocity_rgb_spiral
        pitch_w = v_frame_w * (1.0 - cw["v_overlap"])
        if water_config["camera_type"] == "RGB":
            v_scan, _ = get_velocity_rgb_spiral(
                v_max, R_base, pitch_w, w_arc_w, cw["gsd"],
                cw["max_blur"], cw["shutter"], cw["h_overlap"], cw["fps"])
        else:
            v_scan = v_max
        v_horiz = v_scan   # spiral has one speed

        num_revs = H_water / pitch_w
        n_pts    = max(int(num_revs * 100), 100)
        za = np.linspace(0.0, -H_water, n_pts)
        ta = (za / (-H_water)) * num_revs * 2.0 * np.pi
        ra = np.full(n_pts, r_inspect)
        n_strips = 0   # used only for print

    # ── Steps 2-6: identical to aerial profiling ───────────────────────────────
    xyz   = np.column_stack([ra * np.cos(ta), ra * np.sin(ta), za])
    N_pts = len(za)
    dists = np.array([np.linalg.norm(xyz[i+1] - xyz[i]) for i in range(N_pts - 1)])

    # Classify gaps: 0=vertical, 1=horizontal
    gap_type = np.full(N_pts - 1, -1, dtype=int)
    for i in range(N_pts - 1):
        if dists[i] > 1e-10:
            vf = abs(xyz[i+1, 2] - xyz[i, 2]) / dists[i]
            gap_type[i] = 0 if vf > 0.7 else 1

    # trap_speeds per contiguous same-type segment
    speed_arr = np.zeros(N_pts)
    i = 0
    while i < N_pts - 1:
        while i < N_pts - 1 and gap_type[i] == -1:
            i += 1
        if i >= N_pts - 1:
            break
        seg_t = gap_type[i]
        v_cru = v_scan if seg_t == 0 else v_horiz
        j = i
        while j < N_pts - 1 and gap_type[j] == seg_t:
            j += 1
        s_arr = arc_lengths(xyz[i:j+1])
        speeds, _ = trap_speeds(s_arr, v_cru, TRAJ_ACCEL_MAX)
        speed_arr[i:j+1] = speeds
        i = j

    # Project velocity components from path tangent x speed
    vr_arr  = np.zeros(N_pts)
    vth_arr = np.zeros(N_pts)
    vz_arr  = np.zeros(N_pts)
    for i in range(N_pts - 1):
        if dists[i] < 1e-10:
            continue
        dv   = xyz[i+1] - xyz[i]
        vx_i = (dv[0] / dists[i]) * speed_arr[i]
        vy_i = (dv[1] / dists[i]) * speed_arr[i]
        vz_i = (dv[2] / dists[i]) * speed_arr[i]
        r_i  = max(ra[i], 1e-6)
        th_i = ta[i]
        vr_arr[i]  =  vx_i * np.cos(th_i) + vy_i * np.sin(th_i)
        vth_arr[i] = (-vx_i * np.sin(th_i) + vy_i * np.cos(th_i)) / r_i
        vz_arr[i]  = vz_i

    vr_arr[-1] = vr_arr[-2];  vth_arr[-1] = vth_arr[-2];  vz_arr[-1] = vz_arr[-2]

    # Timestamps: trapezoidal rule over speed profile
    times = [0.0]
    for i in range(1, N_pts):
        d = dists[i-1]
        if d < 1e-10:
            times.append(times[-1])
            continue
        v_avg = max((speed_arr[i-1] + speed_arr[i]) / 2, 1e-6)
        times.append(times[-1] + d / v_avg)

    times = np.array(times)
    wp = np.column_stack([times, ra, ta, za, vr_arr, vth_arr, vz_arr])
    print(f"[UW traj] {fmode}, {n_strips} strips, {N_pts} pts, "
          f"duration={wp[-1,0]:.0f} s, v_scan={v_scan:.3f} m/s, v_horiz={v_horiz:.3f} m/s")
    return wp


def build_test_time_water_traj(t_arr):
    """Interpolate water trajectory onto simulation time vector."""
    wp = _water_timed_waypoints()
    ref_p = np.zeros((3, len(t_arr)))
    ref_v = np.zeros((3, len(t_arr)))
    for ax in range(3):
        ref_p[ax, :] = np.interp(t_arr, wp[:, 0], wp[:, 1 + ax])
        ref_v[ax, :] = np.interp(t_arr, wp[:, 0], wp[:, 4 + ax])
    r0, th0, z0 = wp[0, 1], wp[0, 2], wp[0, 3]
    start_xyz   = np.array([r0 * np.cos(th0), r0 * np.sin(th0), z0])
    return ref_p, ref_v, float(wp[-1, 0]), start_xyz, wp


# ══════════════════════════════════════════════════════════════════════════════
# DISTURBANCE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def get_disturbance(t_k):
    if not DIST_ENABLED:
        return np.zeros(3), np.zeros(3)
    Fd, taud = np.zeros(3), np.zeros(3)
    for row in DISTURBANCES:
        if row[0] <= t_k < row[1]:
            Fd   += np.asarray(row[2:5], dtype=float)
            taud += np.asarray(row[5:8], dtype=float)
    return Fd, taud


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION SETUP
# ══════════════════════════════════════════════════════════════════════════════

dt = 0.05   # [s] timestep

_t_events = [T_BUFFER]
if DIST_ENABLED and DISTURBANCES:
    _t_events.append(max(row[1] for row in DISTURBANCES))

_wp_uw = None

if TRAJ_MODE == "test_time_water":
    _wp_pre = _water_timed_waypoints()
    _t_events.append(float(_wp_pre[-1, 0]))

t_end = max(_t_events) + T_BUFFER
t     = np.arange(0, t_end + dt, dt)
N     = len(t)

x0_override = None

if TRAJ_MODE == "hold":
    ref_pos = np.zeros((3, N))
    ref_vel = np.zeros((3, N))
    print("Trajectory mode : HOLD at origin (underwater)")

elif TRAJ_MODE == "custom":
    ref_pos, ref_vel = build_custom_traj(TRAJ_SEGMENTS, t)
    if isinstance(TRAJ_SEGMENTS, np.ndarray):
        r0, th0, z0 = TRAJ_SEGMENTS[0][1], TRAJ_SEGMENTS[0][2], TRAJ_SEGMENTS[0][3]
        x0_override  = np.array([r0 * np.cos(th0), r0 * np.sin(th0), z0])
    else:
        x0_override = np.array([TRAJ_SEGMENTS[0][1],
                                 TRAJ_SEGMENTS[0][2],
                                 TRAJ_SEGMENTS[0][3]])
    print(f"Trajectory mode : CUSTOM  ({len(TRAJ_SEGMENTS)} segments)")

elif TRAJ_MODE == "test_time_water":
    ref_pos, ref_vel, _dur, _start, _wp_uw = build_test_time_water_traj(t)
    x0_override = _start
    print(f"Trajectory mode : TEST_TIME_WATER  (duration={_dur:.0f} s, "
          f"t_end={t_end:.1f} s)")
    print(f"  Start          : x={_start[0]:.2f} m  y={_start[1]:.2f} m  "
          f"z={_start[2]:.2f} m")
else:
    raise ValueError(f"Unknown TRAJ_MODE: '{TRAJ_MODE}'")

# Cylindrical reference for custom (ndarray) and test_time_water
_USE_CYL_REF = TRAJ_MODE in ("custom", "test_time_water") and isinstance(
    TRAJ_SEGMENTS if TRAJ_MODE == "custom" else True, (np.ndarray, bool))

# Yaw: always face inward toward tower axis
if _USE_CYL_REF:
    ref_yaw = np.arctan2(np.sin(ref_pos[1, :] + np.pi),
                         np.cos(ref_pos[1, :] + np.pi))
else:
    ref_yaw = np.zeros(N)

# Feedforward acceleration (centripetal + Coriolis in cylindrical mode)
if _USE_CYL_REF:
    r_r  = ref_pos[0];  th_r = ref_pos[1]
    vr_r = ref_vel[0];  vth_r = ref_vel[1];  vz_r = ref_vel[2]
    ar_r  = np.gradient(vr_r,  dt)
    ath_r = np.gradient(vth_r, dt)
    az_r  = np.gradient(vz_r,  dt)
    ref_acc = np.zeros((3, N))
    ref_acc[0] = (ar_r - r_r*vth_r**2)*np.cos(th_r) - (r_r*ath_r + 2*vr_r*vth_r)*np.sin(th_r)
    ref_acc[1] = (ar_r - r_r*vth_r**2)*np.sin(th_r) + (r_r*ath_r + 2*vr_r*vth_r)*np.cos(th_r)
    ref_acc[2] = az_r
else:
    ref_acc = np.gradient(ref_vel, dt, axis=1)

print(f"Disturbances    : {'ON' if DIST_ENABLED else 'OFF'}")
print(f"Plots           : {'ON' if ENABLE_PLOTS else 'OFF'}")


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def rot_ZYX(phi, theta, psi):
    """Body-to-inertial rotation matrix, ZYX Euler."""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi),  np.cos(phi)]])
    Ry = np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [0,              1, 0             ],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi),  np.cos(psi), 0],
                   [0,            0,            1]])
    return Rz @ Ry @ Rx


def euler_kin(phi, theta):
    """Maps body rates [p,q,r] → Euler rates. Singular at theta=±90°."""
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([
        [1, sp*st/ct, cp*st/ct],
        [0, cp,      -sp      ],
        [0, sp/ct,    cp/ct   ],
    ])


def uw_ode(s, wr_vert_cmd, Fh_cmd, Fd, taud):
    """
    Nonlinear 6-DOF underwater ODE.
    s            : state (16,) — [x,y,z, phi,theta,psi, vx,vy,vz, p,q,r, w1,w2,w3,w4]
    wr_vert_cmd  : signed vertical prop speed commands (4,) [rad/s]
    Fh_cmd       : signed horizontal thruster force commands (4,) [N]
    Fd           : disturbance force, inertial (3,)
    taud         : disturbance torque, body (3,)
    """
    euler = s[3:6];   phi, theta, psi = euler
    vel   = s[6:9]
    wb    = s[9:12]
    wr_v  = s[12:16]  # actual signed vertical prop speeds

    R = rot_ZYX(phi, theta, psi)

    # Vertical prop forces: signed squared speed q_i = w_i * |w_i|
    q_v  = wr_v * np.abs(wr_v)
    u_v  = A_mix_vert @ q_v          # [Fz_v, tau_phi, tau_theta, tau_psi_v]
    # Fz_v > 0 = upward (same as aerial T); negative at eq to cancel buoyancy

    # Horizontal thruster forces
    Fx_h      = A_mix_horiz[0, :] @ Fh_cmd
    Fy_h      = A_mix_horiz[1, :] @ Fh_cmd
    tau_psi_h = A_mix_horiz[2, :] @ Fh_cmd

    # Total body force (horizontal from H-thrusters, vertical from V-props)
    F_body = np.array([Fx_h, Fy_h, u_v[0]])

    # External: residual buoyancy upward after ballast
    F_buoy_net = np.array([0.0, 0.0, UW_BALLAST_RESIDUAL])

    # Directional drag
    F_drag = -uw.kd_uw * vel

    pos_ddot = (R @ F_body + F_buoy_net + F_drag + Fd) / p.m

    # Body torques: phi/theta from V-props, psi from both
    tau_body = np.array([u_v[1], u_v[2], u_v[3] + tau_psi_h]) + taud
    I_mat    = np.diag([p.Ixx, p.Iyy, p.Izz])
    wb_dot   = np.linalg.solve(I_mat, tau_body - np.cross(wb, I_mat @ wb))

    euler_dot = euler_kin(phi, theta) @ wb
    wr_dot    = (wr_vert_cmd - wr_v) / uw.tau_m_v

    return np.concatenate([vel, euler_dot, pos_ddot, wb_dot, wr_dot])


def rk4_step_uw(s, wr_vert_cmd, Fh_cmd, Fd, taud, dt_):
    k1 = uw_ode(s,              wr_vert_cmd, Fh_cmd, Fd, taud)
    k2 = uw_ode(s + dt_/2*k1,  wr_vert_cmd, Fh_cmd, Fd, taud)
    k3 = uw_ode(s + dt_/2*k2,  wr_vert_cmd, Fh_cmd, Fd, taud)
    k4 = uw_ode(s + dt_*k3,    wr_vert_cmd, Fh_cmd, Fd, taud)
    return s + (dt_/6) * (k1 + 2*k2 + 2*k3 + k4)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP  (skipped in root_locus mode)
# ══════════════════════════════════════════════════════════════════════════════

X       = np.zeros((16, N))
# Equilibrium: negative speed (downward thrust) to cancel upward buoyancy residual
X[12:16, 0] = -uw.omega_v_eq

if x0_override is not None:
    X[0:3, 0] = x0_override
X[5, 0] = ref_yaw[0]   # initialise yaw to avoid 180° spike

int_att = np.zeros(3)
int_pos = np.zeros(3)

# Logs: virtual control inputs [Fz_v, tau_phi, tau_theta, tau_psi]
U_log_vert = np.zeros((4, N))   # [Fz_v_cmd, tau_phi, tau_theta, tau_psi_v]
U_log_horiz = np.zeros((3, N))  # [Fx_cmd, Fy_cmd, tau_psi_h]
Wr_log = np.zeros((4, N))       # commanded vertical prop speeds (signed)
Fh_log = np.zeros((4, N))       # commanded horizontal thruster forces

_sim_iter = range(N - 1) if PLOT_MODE != "root_locus" else []
if tqdm is not None and PLOT_MODE != "root_locus":
    _sim_iter = tqdm(_sim_iter, desc="Simulating UW", unit="step",
                     mininterval=5, dynamic_ncols=True)

for k in _sim_iter:
    s     = X[:, k]
    pos   = s[0:3]
    euler = s[3:6];  phi, theta, psi = euler
    vel   = s[6:9]
    wb    = s[9:12]

    # ── Outer PID: position error → acceleration commands ─────────────────
    if _USE_CYL_REF:
        r_m = max(np.sqrt(pos[0]**2 + pos[1]**2), 1e-6)
        th_m = np.arctan2(pos[1], pos[0])
        cs   = np.cos(th_m);  sn = np.sin(th_m)

        e_r = ref_pos[0, k] - r_m
        e_th = np.arctan2(np.sin(ref_pos[1, k] - th_m),
                          np.cos(ref_pos[1, k] - th_m))
        e_z  = ref_pos[2, k] - pos[2]
        e_t  = r_m * e_th

        vr_m  =  vel[0]*cs + vel[1]*sn
        vth_m = (-vel[0]*sn + vel[1]*cs) / r_m
        e_vr  = ref_vel[0, k] - vr_m
        e_vt  = r_m * (ref_vel[1, k] - vth_m)
        e_vz  = ref_vel[2, k] - vel[2]

        int_pos = np.clip(int_pos + np.array([e_r, e_t, e_z]) * dt,
                          -cyl_i_lim, cyl_i_lim)

        ra_r = ref_acc[0, k]*cs + ref_acc[1, k]*sn
        ra_t = -ref_acc[0, k]*sn + ref_acc[1, k]*cs
        ra_z = ref_acc[2, k]

        a_r = cyl_Kp[0]*e_r + cyl_Ki[0]*int_pos[0] + cyl_Kd[0]*e_vr + ra_r
        a_t = cyl_Kp[1]*e_t + cyl_Ki[1]*int_pos[1] + cyl_Kd[1]*e_vt + ra_t
        a_z = cyl_Kp[2]*e_z + cyl_Ki[2]*int_pos[2] + cyl_Kd[2]*e_vz + ra_z

        a_cmd = np.array([a_r*cs - a_t*sn, a_r*sn + a_t*cs, a_z])

    else:
        e_pos = ref_pos[:, k] - pos
        e_vel = ref_vel[:, k] - vel
        int_pos = np.clip(int_pos + e_pos * dt, -cyl_i_lim, cyl_i_lim)
        a_cmd = cyl_Kp * e_pos + cyl_Ki * int_pos + cyl_Kd * e_vel + ref_acc[:, k]

    # ── Direct force allocation with drag feedforward ─────────────────────
    # Water drag in the ODE is F_drag = -kd_uw * vel (opposes motion).
    # Adding kd_uw * vel to the force command cancels it, leaving the plant
    # as m*x'' = m*a_cmd (ideal integrator chain).  This eliminates the
    # steady-state lag kd*v_ref/(m*Kp) that would otherwise scale with speed.
    Fx_cmd   = p.m * a_cmd[0] + uw.kd_uw[0] * vel[0]
    Fy_cmd   = p.m * a_cmd[1] + uw.kd_uw[1] * vel[1]
    Fz_v_cmd = p.m * a_cmd[2] - UW_BALLAST_RESIDUAL + uw.kd_uw[2] * vel[2]

    U_log_horiz[0, k] = Fx_cmd
    U_log_horiz[1, k] = Fy_cmd

    # ── Inner attitude PID: keep phi=theta=0, track psi reference ─────────
    phi_d   = 0.0
    theta_d = 0.0
    psi_d   = ref_yaw[k]

    e_att    = np.array([phi_d - phi, theta_d - theta, psi_d - psi])
    e_att[2] = np.arctan2(np.sin(e_att[2]), np.cos(e_att[2]))
    int_att  = np.clip(int_att + e_att * dt, -att_i_lim, att_i_lim)

    tau_cmd  = att_Kp * e_att + att_Ki * int_att - att_Kd * wb

    tau_phi_cmd   = tau_cmd[0]
    tau_theta_cmd = tau_cmd[1]
    tau_psi_cmd   = tau_cmd[2]

    # ── Vertical prop mixing: [Fz_v, tau_phi, tau_theta, 0] → motor speeds ─
    u_vert = np.array([Fz_v_cmd, tau_phi_cmd, tau_theta_cmd, 0.0])
    U_log_vert[:, k] = u_vert

    q_cmd  = A_mix_vert_inv @ u_vert   # signed squared speeds
    wr_v_cmd = np.sign(q_cmd) * np.sqrt(np.abs(q_cmd))
    wr_v_cmd = np.clip(wr_v_cmd, -uw.omega_max_v, uw.omega_max_v)
    Wr_log[:, k] = wr_v_cmd

    # ── Horizontal thruster mixing ────────────────────────────────────────
    # a_cmd and the drag-compensation term are in INERTIAL frame.
    # A_mix_horiz produces BODY-frame forces.  Must rotate inertial→body
    # before thruster allocation, otherwise at psi=π (inspection heading)
    # every x-y force is applied backwards and the drone flies away.
    R_cur = rot_ZYX(phi, theta, psi)
    _Fxy_body = R_cur.T @ np.array([Fx_cmd, Fy_cmd, 0.0])
    u_horiz = np.array([_Fxy_body[0], _Fxy_body[1], tau_psi_cmd])
    U_log_horiz[2, k] = tau_psi_cmd
    Fh_cmd_k = A_mix_horiz_pinv @ u_horiz
    Fh_cmd_k = np.clip(Fh_cmd_k, -uw.F_h_max, uw.F_h_max)
    Fh_log[:, k] = Fh_cmd_k

    # ── Disturbance + RK4 integrate ────────────────────────────────────────
    Fd, taud = get_disturbance(t[k])
    X[:, k+1] = rk4_step_uw(s, wr_v_cmd, Fh_cmd_k, Fd, taud, dt)

U_log_vert[:, -1]  = U_log_vert[:, -2]
U_log_horiz[:, -1] = U_log_horiz[:, -2]


# ══════════════════════════════════════════════════════════════════════════════
# LINEARISATION around UW equilibrium
# ══════════════════════════════════════════════════════════════════════════════

# Equilibrium: props at negative speed (downward thrust cancels buoyancy residual)
# ψ = π: drone faces inward toward tower, matching the inspection ref_yaw.
# This is the correct operating point for the root locus — without it the
# x-y rotation effect (R.T at ψ=π negates x,y forces) is not captured.
_wr_v_eq = -uw.omega_v_eq            # signed equilibrium speed (negative = downward)
_s0_uw   = np.zeros(16)
_s0_uw[2]     = UW_LINEARISE_DEPTH   # depth
_s0_uw[5]     = np.pi                # psi = π (facing inward, inspection heading)
_s0_uw[12:16] = _wr_v_eq             # vertical props at equilibrium (negative)

# Input: [w_vert_cmd(4), Fh_cmd(4)] = 8 inputs
_u0_uw = np.concatenate([np.full(4, _wr_v_eq), np.zeros(4)])
_F0_uw = np.zeros(3);   _td0_uw = np.zeros(3)
_eps   = 1e-5
ns_uw, nu_uw = 16, 8

A_lin_uw = np.zeros((ns_uw, ns_uw))
B_lin_uw = np.zeros((ns_uw, nu_uw))

def _uw_ode_flat(s, u_flat, F0, td0):
    """Wrapper: u_flat = [wr_vert_cmd(4), Fh_cmd(4)]."""
    return uw_ode(s, u_flat[:4], u_flat[4:], F0, td0)

for i in range(ns_uw):
    sp, sm = _s0_uw.copy(), _s0_uw.copy()
    sp[i] += _eps;  sm[i] -= _eps
    A_lin_uw[:, i] = (_uw_ode_flat(sp, _u0_uw, _F0_uw, _td0_uw) -
                      _uw_ode_flat(sm, _u0_uw, _F0_uw, _td0_uw)) / (2 * _eps)

for j in range(nu_uw):
    up, um = _u0_uw.copy(), _u0_uw.copy()
    up[j] += _eps;  um[j] -= _eps
    B_lin_uw[:, j] = (_uw_ode_flat(_s0_uw, up, _F0_uw, _td0_uw) -
                      _uw_ode_flat(_s0_uw, um, _F0_uw, _td0_uw)) / (2 * _eps)

print(f"\nUW linearised system: {ns_uw} states, {nu_uw} inputs  "
      f"(depth={UW_LINEARISE_DEPTH} m, wr_v_eq={_wr_v_eq:.3f} rad/s)")
print(f"Open-loop poles:\n{np.sort_complex(np.linalg.eigvals(A_lin_uw))}")


# ── Closed-loop gain matrix for UW root locus ─────────────────────────────────

def build_K_cl_uw(pKp, pKi, pKd, aKp, aKi, aKd, psi_lin=np.pi):
    """
    Returns K_uw (8 x 16): linearised gain from state → [wr_vert_cmd(4), Fh_cmd(4)].
    A_cl_uw = A_lin_uw + B_lin_uw @ K_uw

    psi_lin: linearisation heading [rad].  Use np.pi for the inspection case
    (drone faces inward).  The simulation rotates inertial force commands to
    body frame via R.T(psi); this function applies the same rotation so the
    root locus is consistent with what the nonlinear sim actually does.
    """
    m   = p.m
    wrv = max(abs(_wr_v_eq), 1e-3)   # magnitude of eq speed for linearisation
    cp, sp = np.cos(psi_lin), np.sin(psi_lin)   # body←inertial rotation at psi_lin

    # Virtual input gains for vertical system (4 x 16)
    # Fz_v_cmd = m*a_cmd[2] - RESIDUAL + kd_z*vz   (drag feedforward included)
    # d(Fz_v)/dz  = -m*Kp[2]
    # d(Fz_v)/dvz = -m*Kd[2] + kd_z   ← drag term shifts the vz gain
    Kv_vert = np.zeros((4, 16))
    Kv_vert[0, 2]  = -(m * pKp[2])                    # Fz_v ← z
    Kv_vert[0, 8]  = -(m * pKd[2]) + uw.kd_uw[2]      # Fz_v ← vz  (+kd cancels drag)
    Kv_vert[1, 3]  = -aKp[0]                           # tau_phi ← phi
    Kv_vert[1, 9]  = -aKd[0]                           # tau_phi ← p
    Kv_vert[2, 4]  = -aKp[1]                           # tau_theta ← theta
    Kv_vert[2, 10] = -aKd[1]                           # tau_theta ← q

    K_vert = (1.0 / (2.0 * wrv)) * A_mix_vert_inv @ Kv_vert   # (4 x 16)

    # Virtual input gains for horizontal system (3 x 16)
    # Compute inertial-frame gains first, then rotate to body frame.
    # d(Fx_inertial)/dx = -m*Kp[0],  d(Fx_inertial)/dvx = -m*Kd[0]+kd_x
    # Body←inertial rotation (R.T at psi_lin) is applied to rows 0 and 1.
    # At psi_lin=π: rows 0,1 simply negate — body x points in -inertial x.
    Kv_iner = np.zeros((3, 16))
    Kv_iner[0, 0]  = -(m * pKp[0])                    # Fx_iner ← x
    Kv_iner[0, 6]  = -(m * pKd[0]) + uw.kd_uw[0]     # Fx_iner ← vx
    Kv_iner[1, 1]  = -(m * pKp[1])                    # Fy_iner ← y
    Kv_iner[1, 7]  = -(m * pKd[1]) + uw.kd_uw[1]     # Fy_iner ← vy
    Kv_iner[2, 5]  = -aKp[2]                           # tau_psi ← psi (no rotation)
    Kv_iner[2, 11] = -aKd[2]                           # tau_psi ← r

    # Rotate rows 0,1 from inertial to body (R.T_z = [[cp,sp],[-sp,cp]])
    Kv_horiz = Kv_iner.copy()
    Kv_horiz[0, :] =  cp * Kv_iner[0, :] + sp * Kv_iner[1, :]
    Kv_horiz[1, :] = -sp * Kv_iner[0, :] + cp * Kv_iner[1, :]

    K_horiz = A_mix_horiz_pinv @ Kv_horiz   # (4 x 16)

    return np.vstack([K_vert, K_horiz])     # (8 x 16)


# ── Convert cylindrical ref to Cartesian for plotting ─────────────────────────
if _USE_CYL_REF:
    _r   = ref_pos[0];  _th = ref_pos[1]
    _vr  = ref_vel[0];  _vth = ref_vel[1]
    ref_pos_cart = np.array([_r*np.cos(_th), _r*np.sin(_th), ref_pos[2]])
    ref_vel_cart = np.array([_vr*np.cos(_th) - _r*_vth*np.sin(_th),
                              _vr*np.sin(_th) + _r*_vth*np.cos(_th),
                              ref_vel[2]])
else:
    ref_pos_cart = ref_pos
    ref_vel_cart = ref_vel


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING ERROR SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

if PLOT_MODE == "sim":
    print(f"\n{'═'*56}")
    print("UNDERWATER TRACKING ERROR SUMMARY")
    print(f"{'═'*56}")
    if _USE_CYL_REF:
        _rm  = np.maximum(np.sqrt(X[0]**2 + X[1]**2), 1e-6)
        _thm = np.arctan2(X[1], X[0])
        _er  = ref_pos[0] - _rm
        _eth = np.arctan2(np.sin(ref_pos[1] - _thm), np.cos(ref_pos[1] - _thm))
        _ez  = ref_pos[2] - X[2]
        print(f"  pos r  : mean={np.abs(_er).mean():.3f} m   max={np.abs(_er).max():.3f} m")
        print(f"  pos th : mean={np.abs(_eth).mean():.4f} rad  max={np.abs(_eth).max():.4f} rad")
        print(f"  pos z  : mean={np.abs(_ez).mean():.3f} m   max={np.abs(_ez).max():.3f} m")
    else:
        _ep = ref_pos_cart - X[0:3]
        for i, ax in enumerate(['x','y','z']):
            print(f"  pos {ax}  : mean={np.abs(_ep[i]).mean():.3f} m   max={np.abs(_ep[i]).max():.3f} m")

    # ── Actuator ceiling requirements ─────────────────────────────────────────
    def _rpm(w): return abs(w) * 60.0 / (2.0 * np.pi)

    _wv  = X[12:16, :]                          # actual signed prop speeds (4, N)
    _wvc = Wr_log                                # commanded (4, N)

    _wv_max  = float(np.max(_wv))               # most positive (upward-thrust direction)
    _wv_min  = float(np.min(_wv))               # most negative (downward-thrust direction)
    _wv_peak = float(np.max(np.abs(_wv)))       # largest absolute speed
    _wv_eq   = -uw.omega_v_eq                   # signed design equilibrium

    # Angular acceleration from first-order motor model: dw/dt = (w_cmd - w) / tau_m_v
    _alpha_v = np.abs(_wvc[:, :-1] - _wv[:, :-1]) / uw.tau_m_v
    _alpha_v_max = float(np.max(_alpha_v))
    # Worst-case time to sweep the full bidirectional range at peak acceleration
    _t_sweep_v = (2.0 * _wv_peak) / _alpha_v_max if _alpha_v_max > 0 else float('inf')

    # Thrust: T_i = kT_v * q_i where q_i = w_i * |w_i| (signed)
    _qv      = _wv * np.abs(_wv)
    _T_up    = float(np.max(_qv)) * uw.kT_v     # max upward thrust per prop
    _T_dn    = abs(float(np.min(_qv))) * uw.kT_v # max downward thrust per prop

    # Reaction torque and power per prop
    _Q_v_max = uw.kQ_v * _wv_peak**2
    _P_v_max = uw.kQ_v * _wv_peak**3            # P ≈ torque × speed

    # Horizontal thrusters (instantaneous model, no angular acceleration)
    _fh      = Fh_log                            # (4, N)  N per thruster
    _fh_max  = float(np.max(_fh))
    _fh_min  = float(np.min(_fh))
    _fh_peak = float(np.max(np.abs(_fh)))
    _fh_util = 100.0 * _fh_peak / uw.F_h_max
    # Reconstruct net Fx, Fy, Mz from thruster forces to get peak resultants
    _Fx_hist = A_mix_horiz[0, :] @ _fh           # (N,)
    _Fy_hist = A_mix_horiz[1, :] @ _fh
    _Mz_hist = A_mix_horiz[2, :] @ _fh
    _Fxy_peak = float(np.max(np.sqrt(_Fx_hist**2 + _Fy_hist**2)))
    _Mz_peak  = float(np.max(np.abs(_Mz_hist)))
    # Force rate of change (fd/dt via finite difference)
    _fh_dot_max = float(np.max(np.abs(np.diff(_fh, axis=1)))) / dt

    W = 64
    print(f"\n{'═'*W}")
    print("  VERTICAL PROP CEILING REQUIREMENTS  (motor / ESC sizing)")
    print(f"{'═'*W}")
    print(f"  {'Design equilibrium':<34}  {_wv_eq:>+9.1f} rad/s  ({_rpm(_wv_eq):>6.0f} RPM)")
    print(f"  {'Saturation limit (bidirectional)':<34}  {'+/-':>4} {uw.omega_max_v:>5.1f} rad/s  ({_rpm(uw.omega_max_v):>6.0f} RPM)")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max speed reached (upward dir)':<34}  {_wv_max:>+9.1f} rad/s  ({_rpm(_wv_max):>6.0f} RPM)")
    print(f"  {'Min speed reached (downward dir)':<34}  {_wv_min:>+9.1f} rad/s  ({_rpm(_wv_min):>6.0f} RPM)")
    print(f"  {'Peak absolute speed':<34}  {_wv_peak:>9.1f} rad/s  ({_rpm(_wv_peak):>6.0f} RPM)")
    print(f"  {'Saturation margin remaining':<34}  {uw.omega_max_v - _wv_peak:>9.1f} rad/s  ({_rpm(uw.omega_max_v - _wv_peak):>6.0f} RPM)")
    print(f"  {'Saturation utilisation':<34}  {100*_wv_peak/uw.omega_max_v:>9.1f} %")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max angular acceleration':<34}  {_alpha_v_max:>9.0f} rad/s²")
    print(f"  {'Min time to sweep full range':<34}  {_t_sweep_v:>9.3f} s  (2*peak / max_alpha)")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max upward thrust   (per prop)':<34}  {_T_up:>9.2f} N")
    print(f"  {'Max downward thrust (per prop)':<34}  {_T_dn:>9.2f} N")
    print(f"  {'Max upward thrust   (4 props)':<34}  {4*_T_up:>9.2f} N")
    print(f"  {'Max downward thrust (4 props)':<34}  {4*_T_dn:>9.2f} N")
    print(f"  {'Equilibrium downward thrust':<34}  {UW_BALLAST_RESIDUAL:>9.2f} N  (= ballast residual)")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Max reaction torque (per prop)':<34}  {_Q_v_max:>9.4f} N·m")
    print(f"  {'Max power estimate  (per prop)':<34}  {_P_v_max:>9.2f} W   (kQ × w³)")
    print(f"\n  HORIZONTAL THRUSTER CEILING REQUIREMENTS")
    print(f"  {'─'*(W-2)}")
    print(f"  {'Saturation limit (per thruster)':<34}  {'+/-':>4} {uw.F_h_max:>5.1f} N")
    print(f"  {'Max force reached (per thruster)':<34}  {_fh_max:>9.2f} N")
    print(f"  {'Min force reached (per thruster)':<34}  {_fh_min:>9.2f} N")
    print(f"  {'Peak absolute (per thruster)':<34}  {_fh_peak:>9.2f} N")
    print(f"  {'Saturation utilisation':<34}  {_fh_util:>9.1f} %")
    print(f"  {'Peak resultant Fx (net, body)':<34}  {float(np.max(np.abs(_Fx_hist))):>9.2f} N")
    print(f"  {'Peak resultant Fy (net, body)':<34}  {float(np.max(np.abs(_Fy_hist))):>9.2f} N")
    print(f"  {'Peak combined Fxy (net, body)':<34}  {_Fxy_peak:>9.2f} N")
    print(f"  {'Peak yaw moment Mz (net, body)':<34}  {_Mz_peak:>9.2f} N·m")
    print(f"  {'Max force rate of change':<34}  {_fh_dot_max:>9.1f} N/s  (per thruster)")
    print(f"{'═'*W}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

if PLOT_MODE not in ("sim", "root_locus"):
    print("Plots suppressed.")

elif PLOT_MODE == "root_locus":
    from matplotlib.widgets import Slider

    _pKp = cyl_Kp.copy().astype(float)
    _pKd = cyl_Kd.copy().astype(float)
    _aKp = att_Kp.copy().astype(float)
    _aKd = att_Kd.copy().astype(float)
    _z3  = np.zeros(3)

    def _inner_poles_uw():
        K = build_K_cl_uw(_z3, _z3, _z3, _aKp, _z3, _aKd, psi_lin=np.pi)
        return np.linalg.eigvals(A_lin_uw + B_lin_uw @ K)

    def _A_inner_cl_uw():
        K = build_K_cl_uw(_z3, _z3, _z3, _aKp, _z3, _aKd, psi_lin=np.pi)
        return A_lin_uw + B_lin_uw @ K

    def _full_poles_uw():
        K = build_K_cl_uw(_pKp, _z3, _pKd, _aKp, _z3, _aKd, psi_lin=np.pi)
        return np.linalg.eigvals(A_lin_uw + B_lin_uw @ K)

    fig_rl = plt.figure(figsize=(20, 13))
    fig_rl.suptitle(
        f"UW Sequential closed-loop pole analysis  —  gains: '{_flight_mode}'  "
        f"|  depth={UW_LINEARISE_DEPTH} m  |  Ki omitted",
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
           "Plant: open-loop UW  |  feedback: att_Kp, att_Kd only")
    _setup(ax_out,
           "OUTER LOOP  (position)\n"
           "Plant: inner-loop-closed  |  feedback: pos_Kp, pos_Kd  +  fixed att gains")

    ol_poles_uw = np.linalg.eigvals(A_lin_uw)

    ax_in.scatter(ol_poles_uw.real, ol_poles_uw.imag,
                  marker='o', s=50, color='lightgray', edgecolors='gray',
                  lw=1, label='Plant (open-loop UW)', zorder=3)
    ip0   = _inner_poles_uw()
    sc_in = ax_in.scatter(ip0.real, ip0.imag,
                          marker='x', s=130, lw=2.5,
                          color='darkorange', label='Attitude closed-loop', zorder=5)
    ax_in.legend(loc='upper right', fontsize=8)

    icp0   = np.linalg.eigvals(_A_inner_cl_uw())
    sc_ref = ax_out.scatter(icp0.real, icp0.imag,
                            marker='o', s=50, color='lightgray', edgecolors='gray',
                            lw=1, label='Plant (inner loop closed)', zorder=3)
    fp0    = _full_poles_uw()
    sc_out = ax_out.scatter(fp0.real, fp0.imag,
                            marker='x', s=130, lw=2.5,
                            color='royalblue', label='Full closed-loop', zorder=5)
    ax_out.legend(loc='upper right', fontsize=8)

    # Inner loop: auto-scale to show all poles (including fast motor poles)
    for ax, poles_list in [(ax_in, [ol_poles_uw, ip0])]:
        all_r = np.concatenate([p_.real for p_ in poles_list])
        all_i = np.concatenate([p_.imag for p_ in poles_list])
        pad_r = max(abs(all_r).max() * 0.15, 1.0)
        pad_i = max(abs(all_i).max() * 0.15, 1.0)
        ax.set_xlim(all_r.min() - pad_r, max(all_r.max() + pad_r, 0.5))
        ax.set_ylim(-max(abs(all_i).max() + pad_i, 0.5),
                     max(abs(all_i).max() + pad_i, 0.5))

    # Outer loop: zoom in on SLOW poles only.  Motor poles (Re ≈ -17) and fast
    # attitude poles are off-screen intentionally — they barely move when outer
    # loop gains change.  The interesting region is near the origin where the
    # position poles migrate as cyl_Kp / cyl_Kd are adjusted.
    _slow = lambda poles: poles[poles.real > -10.0]
    _sp0  = np.concatenate([_slow(icp0), _slow(fp0)])
    _sr   = _sp0.real if len(_sp0) > 0 else np.array([-3.0, 0.0])
    _si   = _sp0.imag if len(_sp0) > 0 else np.array([-2.0, 2.0])
    _pad_r = max(abs(_sr).max() * 0.20, 1.0)
    _pad_i = max(abs(_si).max() * 0.20, 1.5)
    ax_out.set_xlim(_sr.min() - _pad_r, 0.5)
    ax_out.set_ylim(-max(abs(_si).max() + _pad_i, 1.5),
                     max(abs(_si).max() + _pad_i, 1.5))
    ax_out.text(0.02, 0.02, "Motor & fast att. poles (Re < −10) off-screen",
                transform=ax_out.transAxes, fontsize=7, color='gray')

    def _refresh_inner():
        ip = _inner_poles_uw()
        sc_in.set_offsets(np.c_[ip.real, ip.imag])
        fig_rl.canvas.draw_idle()

    def _refresh_outer():
        icp = np.linalg.eigvals(_A_inner_cl_uw())
        fp  = _full_poles_uw()
        sc_ref.set_offsets(np.c_[icp.real, icp.imag])
        sc_out.set_offsets(np.c_[fp.real, fp.imag])
        fig_rl.canvas.draw_idle()   # scatter set_offsets does not change axis limits

    # ── Sliders ───────────────────────────────────────────────────────────
    SL_H = 0.060; SL_W = 0.095; SL_GAP = 0.008
    y_kp = 0.04; y_kd = y_kp + SL_H + 0.015
    att_lbls = ['phi', 'theta', 'psi']
    pos_lbls = ['r', 't', 'z']
    att_cols = ['#FF8C00', '#FFD700', '#FF6347']
    pos_cols = ['#4169E1', '#1E90FF', '#00BFFF']

    sl_data = []

    def _add_sliders(x0, Kp_arr, Kd_arr, lbls, cols, prefix):
        for ci in range(3):
            x_col = x0 + ci * (SL_W + SL_GAP)
            for gname, garr, y_row in [('Kp', Kp_arr, y_kp), ('Kd', Kd_arr, y_kd)]:
                ax_sl = fig_rl.add_axes([x_col, y_row, SL_W, SL_H])
                v0    = float(garr[ci])
                vmax  = max(v0 * 1.5, 2.0)
                sl    = Slider(ax_sl, lbls[ci], 0.0, vmax,
                               valinit=v0, valstep=vmax/500, color=cols[ci])
                sl.label.set_size(8)
                sl_data.append((sl, gname, ci, prefix))

    _add_sliders(0.06, _aKp, _aKd, att_lbls, att_cols, 'att')
    _add_sliders(0.55, _pKp, _pKd, pos_lbls, pos_cols, 'pos')

    fig_rl.text(0.06 + 1*(SL_W+SL_GAP), y_kd + SL_H + 0.010,
                'INNER  —  att_Kp / att_Kd  (phi, theta, psi)', fontsize=9,
                fontweight='bold', ha='center', color='dimgray')
    fig_rl.text(0.55 + 1*(SL_W+SL_GAP), y_kd + SL_H + 0.010,
                'OUTER  —  cyl_Kp / cyl_Kd  (r, t, z)', fontsize=9,
                fontweight='bold', ha='center', color='dimgray')

    def _make_cb(gname, ci, prefix):
        def cb(val):
            if prefix == 'att':
                if gname == 'Kp': _aKp[ci] = val
                else:             _aKd[ci] = val
                _refresh_inner()
                _refresh_outer()   # inner gains shift the plant seen by outer loop
            else:
                if gname == 'Kp': _pKp[ci] = val
                else:             _pKd[ci] = val
                _refresh_outer()
        return cb

    for sl, gname, ci, prefix in sl_data:
        sl.on_changed(_make_cb(gname, ci, prefix))

    plt.show()

else:  # PLOT_MODE == "sim"
    _ps        = max(1, N // 10_000)
    t_p        = t[::_ps]
    X_p        = X[:, ::_ps]
    rp_p       = ref_pos_cart[:, ::_ps]
    rv_p       = ref_vel_cart[:, ::_ps]
    ref_yaw_p  = ref_yaw[::_ps]
    Uv_p       = U_log_vert[:, ::_ps]
    Uh_p       = U_log_horiz[:, ::_ps]
    Wr_p       = Wr_log[:, ::_ps]
    Fh_p       = Fh_log[:, ::_ps]

    if _USE_CYL_REF:
        _r_p  = np.maximum(np.sqrt(X_p[0,:]**2 + X_p[1,:]**2), 1e-6)
        _th_p = np.unwrap(np.arctan2(X_p[1,:], X_p[0,:]))
        _z_p  = X_p[2,:]
        _vr_p =  X_p[6,:]*np.cos(_th_p) + X_p[7,:]*np.sin(_th_p)
        _vth_p = (-X_p[6,:]*np.sin(_th_p) + X_p[7,:]*np.cos(_th_p)) / _r_p
        _vz_p  = X_p[8,:]

        _rr_p  = ref_pos[0, ::_ps];  _thr_p = ref_pos[1, ::_ps]; _zr_p = ref_pos[2, ::_ps]
        _vr_r  = ref_vel[0, ::_ps];  _vthr  = ref_vel[1, ::_ps]; _vzr  = ref_vel[2, ::_ps]

    gc = (0.85, 0.95, 0.85)

    def shade_gusts(ax):
        yl = ax.get_ylim()
        if DIST_ENABLED:
            for row in DISTURBANCES:
                ax.axvspan(row[0], row[1], color=gc, alpha=0.5, zorder=0)
        ax.set_ylim(yl)

    # ── Figure 1: Position + Attitude ────────────────────────────────────────
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig1.suptitle(f"UW Position & Attitude  [{TRAJ_MODE}]", fontsize=13)

    if _USE_CYL_REF:
        pos_labels = ['r  [m]', 'theta  [rad]', 'z  [m]']
        pos_actual = [_r_p, _th_p, _z_p]
        pos_ref    = [_rr_p, _thr_p, _zr_p]
    else:
        pos_labels = ['x  [m]', 'y  [m]', 'z  [m]']
        pos_actual = [X_p[0,:], X_p[1,:], X_p[2,:]]
        pos_ref    = [rp_p[0,:], rp_p[1,:], rp_p[2,:]]

    att_labels = ['phi  [deg]', 'theta  [deg]', 'psi  [deg]']

    for i in range(3):
        ax = axes1[i, 0]
        if PLOT_ACTUAL:    ax.plot(t_p, pos_actual[i], 'b', lw=1.6, label='Actual')
        if PLOT_REFERENCE: ax.plot(t_p, pos_ref[i], 'r--', lw=1.2, label='Reference')
        ax.set_ylabel(pos_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0:
            ax.set_title("Position (cylindrical)" if _USE_CYL_REF else "Position")
            ax.legend(loc='lower right')

        ax = axes1[i, 1]
        ax.plot(t_p, np.degrees(X_p[3+i, :]), 'b', lw=1.6, label='Actual')
        if i == 2 and PLOT_REFERENCE:
            ax.plot(t_p, np.degrees(np.unwrap(ref_yaw_p)), 'r--', lw=1.2, label='Reference')
            ax.legend(loc='lower right')
        ax.set_ylabel(att_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0: ax.set_title("Attitude")

    axes1[2, 0].set_xlabel("Time  [s]")
    axes1[2, 1].set_xlabel("Time  [s]")
    fig1.tight_layout()

    # ── Figure 2: Velocity tracking ──────────────────────────────────────────
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig2.suptitle(f"UW Velocity Tracking  [{TRAJ_MODE}]", fontsize=13)

    if _USE_CYL_REF:
        vel_labels = ['vr  [m/s]', 'vtheta  [rad/s]', 'vz  [m/s]']
        vel_actual = [_vr_p, _vth_p, _vz_p]
        vel_ref    = [_vr_r, _vthr, _vzr]
    else:
        vel_labels = ['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]']
        vel_actual = [X_p[6,:], X_p[7,:], X_p[8,:]]
        vel_ref    = [rv_p[0,:], rv_p[1,:], rv_p[2,:]]

    for i in range(3):
        ax = axes2[i]
        if PLOT_ACTUAL:    ax.plot(t_p, vel_actual[i], 'b',   lw=1.6, label='Actual')
        if PLOT_REFERENCE: ax.plot(t_p, vel_ref[i],    'r--', lw=1.2, label='Reference')
        ax.set_ylabel(vel_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0: ax.legend(loc='lower right')

    axes2[-1].set_xlabel("Time  [s]")
    fig2.tight_layout()

    # ── Figure 3: Virtual control inputs ─────────────────────────────────────
    fig3, axes3 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig3.suptitle("UW Virtual Control Inputs", fontsize=13)

    vert_labels  = ['Fz_v  [N]', 'tau_phi  [N·m]', 'tau_theta  [N·m]']
    horiz_labels = ['Fx  [N]', 'Fy  [N]', 'tau_psi  [N·m]']

    for i in range(3):
        ax = axes3[i, 0]
        ax.plot(t_p, Uv_p[i, :], 'teal', lw=1.6)
        if i == 0:
            ax.axhline(-UW_BALLAST_RESIDUAL, color='r', ls='--', lw=1.0,
                       label=f'Eq Fz_v (−{UW_BALLAST_RESIDUAL:.1f} N)')
            ax.legend(loc='lower right')
        ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.set_ylabel(vert_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0: ax.set_title("Vertical props")

        ax = axes3[i, 1]
        ax.plot(t_p, Uh_p[i, :], 'darkorange', lw=1.6)
        ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.set_ylabel(horiz_labels[i]); ax.grid(True)
        shade_gusts(ax)
        if i == 0: ax.set_title("Horizontal thrusters")

    axes3[2, 0].set_xlabel("Time  [s]")
    axes3[2, 1].set_xlabel("Time  [s]")
    fig3.tight_layout()

    # ── Figure 4: Vertical prop speeds (signed) ───────────────────────────────
    fig4, axes4 = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    fig4.suptitle("UW Vertical Prop Speeds (signed: + = down thrust)", fontsize=13)
    prop_names = ['Prop 0 (45°)', 'Prop 1 (135°)', 'Prop 2 (225°)', 'Prop 3 (315°)']

    for i in range(4):
        ax = axes4[i]
        ax.plot(t_p, X_p[12+i, :], 'b', lw=1.6)
        ax.axhline(-uw.omega_v_eq,  color='r', ls='--', lw=1.0, label='w_eq (downward)')
        ax.axhline( uw.omega_v_eq,  color='r', ls='--', lw=1.0)
        ax.axhline( uw.omega_max_v, color='k', ls='--', lw=1.0, label='max')
        ax.axhline(-uw.omega_max_v, color='k', ls='--', lw=1.0)
        ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.set_ylabel(f"{prop_names[i]}  [rad/s]"); ax.grid(True)
        if i == 0: ax.legend(loc='lower right')
        shade_gusts(ax)

    axes4[-1].set_xlabel("Time  [s]")
    fig4.tight_layout()

    # ── Figure 5: Open-loop pole map ──────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(7, 6))
    poles_uw = np.linalg.eigvals(A_lin_uw)
    ax5.scatter(poles_uw.real, poles_uw.imag, marker='x', s=80, color='teal', zorder=5)
    ax5.axvline(0, color='k', lw=0.8, ls='--')
    ax5.axhline(0, color='k', lw=0.8, ls='--')
    ax5.set_xlabel("Real"); ax5.set_ylabel("Imaginary")
    ax5.set_title(f"UW Open-Loop Poles  (depth={UW_LINEARISE_DEPTH} m)")
    ax5.grid(True)
    fig5.tight_layout()

    # ── Figure 6: 3D flight path ──────────────────────────────────────────────
    fig6 = plt.figure(figsize=(11, 10))
    ax6  = fig6.add_subplot(111, projection='3d')

    # Draw monopile cylinder (underwater section, z negative)
    _th_s = np.linspace(0, 2*np.pi, 60)
    _z_uw = np.linspace(-H_water, 0, 40)
    _TH_uw, _Z_uw = np.meshgrid(_th_s, _z_uw)
    ax6.plot_surface(R_base*np.cos(_TH_uw), R_base*np.sin(_TH_uw), _Z_uw,
                     color='silver', alpha=0.30, edgecolor='none')

    # Sea surface plane
    _xy_s = R_base * 2.5
    _xs, _ys = np.meshgrid(np.linspace(-_xy_s, _xy_s, 2),
                            np.linspace(-_xy_s, _xy_s, 2))
    ax6.plot_surface(_xs, _ys, np.zeros_like(_xs),
                     color='dodgerblue', alpha=0.15)

    if PLOT_REFERENCE:
        if TRAJ_MODE == "test_time_water" and _wp_uw is not None:
            _rx = _wp_uw[:, 1] * np.cos(_wp_uw[:, 2])
            _ry = _wp_uw[:, 1] * np.sin(_wp_uw[:, 2])
            _rz = _wp_uw[:, 3]
        else:
            _rx, _ry, _rz = rp_p[0, :], rp_p[1, :], rp_p[2, :]
        ax6.plot(_rx, _ry, _rz, color='red', lw=1.8, ls='--',
                 label='Reference', zorder=5)

    if PLOT_ACTUAL:
        ax6.plot(X_p[0, :], X_p[1, :], X_p[2, :],
                 color='teal', lw=1.8, label='Actual', zorder=6)

    ax6.set_xlabel("X  [m]")
    ax6.set_ylabel("Y  [m]")
    ax6.set_zlabel("Z  [m]  (negative = depth)")
    ax6.set_title(f"UW 3D Flight Path  [{TRAJ_MODE}]", fontsize=13)
    ax6.legend(loc='upper left')
    fig6.tight_layout()

    plt.show()
