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
from tqdm import tqdm
from Test_time import (air_config, cameras, R_base, R_top, H_air_cyl, H_air_cone,
                       _aerial_timed_waypoints, _turbine_timed_waypoints)
from aerial_props import C_T as _ap_CT, C_Q as _ap_CQ, D as _ap_D, kT as _ap_kT, kQ as _ap_kQ
from geometry import total_mass as _geo_mass, Ixx as _geo_Ixx, Iyy as _geo_Iyy, Izz as _geo_Izz, L_arm as _geo_Larm, L_box as _geo_Lbox, W_box as _geo_Wbox, H_box as _geo_Hbox
from sensors_SC import SensorSuite
from kalman_SC  import KinematicEKF12

# ══════════════════════════════════════════════════════════════
#   SIMULATION CONFIG  —  edit only in this section
# ══════════════════════════════════════════════════════════════

# ── Trajectory ────────────────────────────────────────────────
TRAJ_MODE      = "test_time_air"  # "hold" | "custom" | "test_time_air"
AERIAL_PHASE   = "tower"          # "tower" | "turbine"  (only used when TRAJ_MODE = "test_time_air")
TRAJ_PCT       = 100              # [1-100] percentage of test_time_air waypoints to use
                                   # 100 = full path, 25 = first quarter of lawnmower strips
TRAJ_ACCEL_MAX = 20             # [m/s²] max acceleration for trapezoidal velocity profile
CORNER_RADIUS  = 0                # [m]    corner-rounding radius at strip ends; 0 = sharp turn

# custom waypoint table — only used when TRAJ_MODE = "custom"
# Each row: (t_start, x, y, z)  or  (t_start, x, y, z, vx, vy, vz)
TRAJ_SEGMENTS = [
    (0, 0, 0, 0),
    (2, 0, 0, 1),
]

# ── Waypoint advancement ──────────────────────────────────────
# True:  advance to next waypoint only when ALL THREE cylindrical errors are within tolerance
# False: reference advances in lockstep with simulation time
USE_EVENT_TRIG = True
EVENT_R_TOL    = 1   # [m]   radial standoff tolerance
EVENT_TH_TOL   = 0.2   # [rad] azimuth tolerance (~23°)
EVENT_Z_TOL    = 1   # [m]   height tolerance

# ── Obstacle avoidance ───────────────────────────────────────
OA_MODE = "none"  # "none"    — obstacle avoidance off
                     # "apf_ref" — APF-inspired reference deflection (obstacle_avoidance.py)
                     #             shifts the reference position; PID generates corrective forces
                     # "apf"     — true APF direct force injection (obstacle_avoidance_apf.py)
                     #             repulsive acceleration added to a_cmd at drone's actual position
                     # "mpc"     — receding-horizon MPC planner (mpc_planner.py)
                     #             plans a collision-free trajectory around the obstacle and
                     #             resumes the offline path; fully independent of APF modes

# ── EKF ──────────────────────────────────────────────────────
USE_EKF = True   # True:  EKF runs, estimates feed the controller + waypoint manager,
                   #        and EKF vs truth plots are added to the output figures
                   # False: controller uses true plant states, no EKF plots

# ── Disturbances ──────────────────────────────────────────────
DIST_ENABLED    = False
IMPULSE_ENABLED = False
WIND_ENABLED    = False
WIND_INTERP     = 'linear'   # 'linear' | 'cubic' (cubic needs ≥ 4 rows)

# rows: (t_on, t_off, Fx, Fy, Fz [N inertial], tx, ty, tz [N·m body])
DISTURBANCES = [
    (2.0, 2.1,   1.0, 0.0, 0.0,   0.0, 0.0, 0.0),
]
# rows: (t_impulse, Jx, Jy, Jz [N·s], Jtx, Jty, Jtz [N·m·s])
IMPULSES = [
    (2.0,  50.0, 0.0, 0.0,  0.0, 0.0, 0.0),
]
# wind velocity profile — inertial frame [t, vx, vy, vz]
WIND_PROFILE = np.array([
    [0.00,  0.0, 0.0, 0.0],
    [0.999, 0.0, 0.0, 0.0],
    [1.000, 2.0, 0.0, 0.0],
])

# ── Output ────────────────────────────────────────────────────
SAVE_SIM_DATA  = True    # save state data to sim_data_SC.npz (for kalman_SC.py)
SAVE_PCT       = 100      # % of flight to save (first N%) at full resolution
PLOT_MODE      = "sim"   # "sim" | "root_locus" | "none"
PLOT_REFERENCE = True    # overlay reference trajectory on position plots
PLOT_VIBRATION = True   # show vibration frequency envelope (rotor harmonics)
N_BLADES       = 2       # propeller blade count (for vibration plot)

# ── Timing ────────────────────────────────────────────────────
T_BUFFER = 50   # [s] simulation time added after last trajectory event

# ── Step-response analysis (custom mode only) ─────────────────
STEP_ANALYSIS_WINDOW   = 20.0   # [s]
SETTLING_THRESHOLD_PCT = 5.0    # [%] band around final value

# ══════════════════════════════════════════════════════════════


# linearisation point for root locus — sourced from geometry.py (shared with pole_placement.py)
from geometry import LIN_POS, LIN_EULER, LIN_VEL

# --- Vehicle parameters ---

def run_sim(overrides=None, show_plots=True):
    """Run simulation. overrides dict keys (all optional):
       mass_factor    float  multiplier on total mass              (default 1.0)
       ixx_factor     float  multiplier on Ixx roll inertia        (default 1.0)
       iyy_factor     float  multiplier on Iyy pitch inertia       (default 1.0)
       izz_factor     float  multiplier on Izz yaw inertia         (default 1.0)
       tau_m_factor   float  multiplier on motor time constant     (default 1.0)
       traj_mode      str    override TRAJ_MODE                    (default file value)
       wind_enabled   bool   override WIND_ENABLED                 (default file value)
       wind_speed     float  set constant wind magnitude [m/s]     (enables wind automatically)
       dist_enabled   bool   override DIST_ENABLED                 (default file value)
       gust_force     float  lateral gust [N] for 0.5 s at t=5 s  (enables dist automatically)
       impulse_enabled bool  override IMPULSE_ENABLED              (default file value)
       impulse_force  float  lateral impulse [N·s] at t=5 s       (enables impulse automatically)
       t_buffer       float  override T_BUFFER sim duration [s]    (default file value)
    """
    _ov = overrides or {}
    _mass_factor  = float(_ov.get('mass_factor',  1.0))
    _ixx_factor   = float(_ov.get('ixx_factor',   1.0))
    _iyy_factor   = float(_ov.get('iyy_factor',   1.0))
    _izz_factor   = float(_ov.get('izz_factor',   1.0))
    _tau_m_factor = float(_ov.get('tau_m_factor', 1.0))

    # Shadow module-level config with overridable locals
    _g = globals()
    TRAJ_MODE       = _ov.get('traj_mode',       _g['TRAJ_MODE'])
    TRAJ_SEGMENTS   = _ov.get('traj_segments',   _g['TRAJ_SEGMENTS'])
    USE_EKF         = _ov.get('ekf_enabled',     _g['USE_EKF'])
    DIST_ENABLED    = _ov.get('dist_enabled',    _g['DIST_ENABLED'])
    IMPULSE_ENABLED = _ov.get('impulse_enabled', _g['IMPULSE_ENABLED'])
    WIND_ENABLED    = _ov.get('wind_enabled',    _g['WIND_ENABLED'])
    DISTURBANCES    = list(_g['DISTURBANCES'])
    IMPULSES        = list(_g['IMPULSES'])
    WIND_PROFILE    = _g['WIND_PROFILE'].copy()
    T_BUFFER        = float(_ov.get('t_buffer',  _g['T_BUFFER']))

    # Convenience: constant wind (automatically enables)
    if 'wind_speed' in _ov:
        _ws = float(_ov['wind_speed'])
        WIND_PROFILE = np.array([[0.0, _ws, 0.0, 0.0], [99999.0, _ws, 0.0, 0.0]])
        WIND_ENABLED = True

    # Convenience: sustained lateral gust force for 0.5 s at t=5 s
    if 'gust_force' in _ov:
        DISTURBANCES = [(5.0, 5.5, float(_ov['gust_force']), 0.0, 0.0, 0.0, 0.0, 0.0)]
        DIST_ENABLED = True

    # Convenience: lateral impulse at t=5 s
    if 'impulse_force' in _ov:
        IMPULSES = [(5.0, float(_ov['impulse_force']), 0.0, 0.0, 0.0, 0.0, 0.0)]
        IMPULSE_ENABLED = True

    # ── Obstacle avoidance module — re-imported every run so edits take effect ──
    import sys as _sys
    _mpc = None
    _obs_list = []
    _obs_Rbase = 0.0
    _obs_cyl2xyz = None
    _oa_raw = _oa_speed = _oa_tau = None
    _oa_apf_acc = _oa_apf_speed = None
    if OA_MODE != "none":
        _sys.modules.pop('obstacle_avoidance', None)
        import obstacle_avoidance as _oa_mod
        _obs_list    = _oa_mod.OBSTACLES
        _obs_Rbase   = _oa_mod.R_BASE
        _obs_cyl2xyz = _oa_mod._cyl_to_xyz
        if OA_MODE == "apf_ref":
            _oa_raw   = _oa_mod.compute_raw_deflection
            _oa_speed = _oa_mod.compute_speed_scale
            _oa_tau   = _oa_mod.SMOOTH_TAU
        elif OA_MODE == "apf":
            _sys.modules.pop('obstacle_avoidance_apf', None)
            import obstacle_avoidance_apf as _oa_apf
            _oa_apf_acc   = _oa_apf.compute_apf_acceleration
            _oa_apf_speed = _oa_apf.compute_speed_scale
        elif OA_MODE == "mpc":
            _sys.modules.pop('mpc_planner', None)
            from mpc_planner import MPCPlanner
            _mpc = MPCPlanner(
                obstacles   = _obs_list,
                cyl_to_xyz  = _obs_cyl2xyz,
                use_cyl_ref = True,
                dt_sim      = 0.005,
            )
            print(f"[MPC] Planner ready  (N={_mpc.N} steps × {_mpc.dt}s = "
                  f"{_mpc.N*_mpc.dt:.1f}s horizon, "
                  f"detect={_mpc.det_steps} fine steps = {_mpc.det_steps*0.005:.1f}s)")
        print(f"[OA] mode={OA_MODE!r}  {len(_obs_list)} obstacle(s): "
              + ", ".join(f"r={o['pos'][0]}m θ={o['pos'][1]}° z={o['pos'][2]}m rad={o['radius']}m"
                          for o in _obs_list))

    _step_vis = None   # populated by step-response analysis, used by position plot

    class Params:
        m     = _geo_mass * _mass_factor  # total mass from geometry.py
        Ixx   = _geo_Ixx  * _ixx_factor   # roll inertia
        Iyy   = _geo_Iyy  * _iyy_factor   # pitch inertia
        Izz   = _geo_Izz  * _izz_factor   # yaw inertia
        Ixz   = 0.0        # xz product of inertia (ZX-plane symmetry assumed)
        l     = _geo_Larm  # arm length CoM to rotor
        g     = 9.81
    
        # ── Propeller aerodynamics — fitted from aerial_props.py ────────────
        CT_prop = _ap_CT
        CQ_prop = _ap_CQ
        D_prop  = _ap_D
        kT      = _ap_kT
        kQ      = _ap_kQ
        tau_m = 0.06 * _tau_m_factor      # [s] motor lag
        rho_air = 1.225
        Cd      = np.array([1.28, 1.28, 1.28])       # bluff-body drag coeff [x, y, z]
        A_face  = np.array([
            _geo_Wbox * _geo_Hbox,                 # frontal area in x
            _geo_Lbox * _geo_Hbox,                 # frontal area in y
            _geo_Lbox * _geo_Wbox,                 # frontal area in z
        ])
        Jr      = 6.0e-5   # [kg·m²] rotor spin inertia
    
        omega_max = 569.0  # [rad/s] rotor speed limit
    
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

    # PID gains — EKF-feedback detuned for sensor noise.
    # att_Kp/Kd rescaled from old box-hull values by inertia ratio:
    #   roll I: 0.661/1.791=0.369  pitch I: 1.837/2.687=0.684  yaw I: 2.356/3.693=0.638
    # Inner att_Kp/Kd halved from nominal — EKF theta has ~10-15° residual error which
    # gets amplified by high gains into destabilising torques during vertical climb.
    # Without rescaling: 2.7x (roll) / 1.5x (pitch) excess angular accel per deg of EKF error.
    _g = dict(
        att_Kp    = np.array([11.1,  67.7,   12.8 ]),
        att_Ki    = np.array([0.1,   0.1,    0.05 ]),
        att_Kd    = np.array([3.3,   15.1,   6.4  ]),
        att_i_lim = np.array([5.0,   5.0,    3.0  ]),
        att_lim   = 0.45,
        cyl_Kp    = np.array([0.10,  0.14,   1.40 ]),
        cyl_Ki    = np.array([0.005, 0.005,  0.05 ]),
        cyl_Kd    = np.array([0.64,  0.54,   1.25 ]),
        cyl_i_lim = np.array([5.0,   5.0,   10.0  ]),
    )

 

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

    print(f"PID gains       : lawnmower_ekf set")

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


    def build_test_time_aerial_traj(t_arr, wp=None):
        """Interpolate aerial path onto the sim time array.
        Returns ref_pos (3,N), ref_vel (3,N), duration (s), start_xyz (3,).
        wp: optional pre-sliced waypoint array (M,7); if None the full path is used.
        """
        if wp is None:
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

    ENABLE_EKF       = USE_EKF
    USE_EKF_FEEDBACK = USE_EKF

    dt = 0.005   # [s] plant timestep — EKF sub-steps internally at IMU rate
    if ENABLE_EKF:
        from sensors_SC import IMUSensor as _IMU
        _EKF_SUBSTEPS = max(1, round(dt * _IMU.update_rate))  # EKF steps per plant step
        _DT_EKF       = dt / _EKF_SUBSTEPS                    # actual EKF dt
        if _EKF_SUBSTEPS > 1:
            print(f"[EKF] plant {1/dt:.0f} Hz < IMU {_IMU.update_rate:.0f} Hz "
                  f"→ {_EKF_SUBSTEPS} EKF sub-steps per plant step (dt_ekf={_DT_EKF:.4f} s)")
    else:
        _EKF_SUBSTEPS = 1
        _DT_EKF       = dt

    # end time = latest event + buffer
    _t_events = [T_BUFFER]
    if DIST_ENABLED and DISTURBANCES:
        _t_events.append(max(row[1] for row in DISTURBANCES))

    if TRAJ_MODE == "custom" and len(TRAJ_SEGMENTS):
        _t_events.append(TRAJ_SEGMENTS[-1][0])
    elif TRAJ_MODE == "test_time_air":
        if AERIAL_PHASE == "turbine":
            _wp, _, _ = _turbine_timed_waypoints()
        else:
            _wp, _, _ = _aerial_timed_waypoints()
        _n_wp = max(2, int(round(len(_wp) * min(100, max(1, TRAJ_PCT)) / 100)))
        _wp   = _wp[:_n_wp]
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
        ref_pos, ref_vel, _dur, _start = build_test_time_aerial_traj(t, wp=_wp)
        x0_override = _start
        print(f"Trajectory mode : TEST_TIME_AIR  phase={AERIAL_PHASE}  (path duration={_dur:.0f} s, "
              f"t_end={t_end:.1f} s, TRAJ_PCT={TRAJ_PCT}%  [{_n_wp} waypoints])")
        print(f"  Start waypoint : x={_start[0]:.2f} m  y={_start[1]:.2f} m  "
              f"z={_start[2]:.2f} m")

    else:
        raise ValueError(f"Unknown TRAJ_MODE: '{TRAJ_MODE}'")

    # cylindrical ref used for custom (ndarray) and test_time_air
    _USE_CYL_REF = TRAJ_MODE in ("custom", "test_time_air") and isinstance(
        TRAJ_SEGMENTS if TRAJ_MODE == "custom" else True, (np.ndarray, bool))

    # Patch MPC reference frame now that _USE_CYL_REF is known
    if OA_MODE == "mpc" and _mpc is not None:
        _mpc.cyl = _USE_CYL_REF

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

    # export the exact ref arrays SC uses to the appropriate trajectory .m file
    if TRAJ_MODE == "test_time_air":
        from Test_time import omega_h as _omega_h, write_matlab_traj as _write_traj
        _mat = np.column_stack([ref_pos[0], ref_pos[1], ref_pos[2],
                                ref_vel[0], ref_vel[1], ref_vel[2],
                                ref_acc[0], ref_acc[1], ref_acc[2],
                                ref_yaw])
        _save_base = r'C:\Users\banda\Documents\MATLAB\UAUV Control\Aerial'
        if AERIAL_PHASE == "turbine":
            _fname  = f'{_save_base}/trajectory_turbine_blade.m'
            _label  = 'turbine blade'
            _dur_key = 'duration_turbine'
            _n_key   = 'n_waypoints_turbine'
        else:
            _fname  = f'{_save_base}/trajectory_aerial.m'
            _label  = 'aerial'
            _dur_key = 'duration_aerial'
            _n_key   = 'n_waypoints_aerial'
        _header = [
            ('omega_h',   _omega_h,        'rad/s  hover rotor speed'),
            (_dur_key,    float(t[-1]),    's'),
            (_n_key,      float(len(t)),   'rows'),
        ]
        _footer = ("\nx0        = zeros(16,1);\n"
                   "x0(1)     = trajectory(1, 1) * cos(trajectory(1, 2));\n"
                   "x0(2)     = trajectory(1, 1) * sin(trajectory(1, 2));\n"
                   "x0(3)     = trajectory(1, 3);\n"
                   "x0(6)     = trajectory(1, 10);\n"
                   "x0(13:16) = omega_h;\n")
        _write_traj(_fname, _mat, _header, _label, extra_footer=_footer)
        print(f"MATLAB export   : {_fname}  ({len(t)} rows)")

    print(f"Event trig WP   : {'ON' if USE_EVENT_TRIG else 'OFF'}  "
          f"(r={EVENT_R_TOL} m  θ={EVENT_TH_TOL} rad  z={EVENT_Z_TOL} m)")
    print(f"Disturbances    : {'ON' if DIST_ENABLED else 'OFF'}  "
          f"({len(DISTURBANCES)} row(s) defined)")
    print(f"Plots           : {'ON' if PLOT_MODE != 'none' else 'OFF'}")

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

    U_log        = np.zeros((4, N))   # [T, τ_φ, τ_θ, τ_ψ]
    Wr_log       = np.zeros((4, N))   # commanded rotor speeds
    Ref_log      = np.zeros((3, N))   # event-triggered reference position logged at each step
    Ref_vel_log  = np.zeros((3, N))   # event-triggered reference velocity logged at each step

    k_ref    = 0              # event-triggered reference pointer
    _oa_defl = np.zeros(3)    # smoothed reference deflection [x,y,z] (apf_ref only)
    APF_defl_log = np.zeros((3, N))   # per-step APF deflection vector (apf_ref only)
    if OA_MODE != "none":
        _oa_min_dist = [np.inf] * len(_obs_list)   # min surface distance per obstacle
    _t_complete = None     # sim time when drone first reaches last trajectory waypoint
    _k_end = N - 1        # last valid filled index (updated on early exit)
    _ekf_feedback_x = None # last EKF estimate; used when USE_EKF_FEEDBACK = True

    # Cartesian last-waypoint position for completion detection (immune to theta aliasing)
    if TRAJ_MODE == "test_time_air":
        _lw_x = _wp[-1, 1] * np.cos(_wp[-1, 2])
        _lw_y = _wp[-1, 1] * np.sin(_wp[-1, 2])
        _lw_z = _wp[-1, 3]
        _COMPLETE_RADIUS = 1.5
    else:
        _lw_x = _lw_y = _lw_z = 0.0;  _COMPLETE_RADIUS = 1.0

    # EKF + sensor suite — runs in parallel, controller still uses true states
    if ENABLE_EKF:
        _suite    = SensorSuite(R_turbine=R_base)
        _ekf      = KinematicEKF12()
        _ekf.x[0:3]  = X[0:3, 0]         # seed position
        _ekf.x[3:6]  = X[6:9, 0]         # seed velocity
        _ekf.x[6:9]  = X[3:6, 0]         # seed attitude
        _ekf.x[9:12] = X[9:12, 0]        # seed body rates
        X_ekf = np.zeros((12, N))         # [pos(3), vel(3), euler(3), rates(3)]

    _sim_iter = range(N - 1) if PLOT_MODE != "root_locus" else []
    if tqdm is not None and PLOT_MODE != "root_locus":
        _sim_iter = tqdm(_sim_iter, desc="Simulating", unit="step",
                         mininterval=5, dynamic_ncols=True)
    for k in _sim_iter:
        s     = X[:, k]
        pos   = s[0:3]        # true plant states — always used for EKF sensor model
        euler = s[3:6];  phi, theta, psi = euler
        vel   = s[6:9]
        wb    = s[9:12]
        wr    = s[12:16]

        # Controller input: EKF estimate (if feedback enabled) or true state
        if USE_EKF_FEEDBACK and _ekf_feedback_x is not None:
            c_pos   = _ekf_feedback_x[0:3]
            c_vel   = _ekf_feedback_x[3:6]
            c_euler = _ekf_feedback_x[6:9]
            c_wb    = _ekf_feedback_x[9:12]
        else:
            c_pos, c_vel, c_euler, c_wb = pos, vel, euler, wb
        c_phi, c_theta, c_psi = c_euler

        if USE_EVENT_TRIG and _USE_CYL_REF:
            if k_ref < N - 1:
                _r_m  = np.sqrt(c_pos[0]**2 + c_pos[1]**2)
                _th_m = np.arctan2(c_pos[1], c_pos[0])
                if (abs(_r_m - ref_pos[0, k_ref]) < EVENT_R_TOL and
                    abs(np.arctan2(np.sin(_th_m - ref_pos[1, k_ref]),
                                   np.cos(_th_m - ref_pos[1, k_ref]))) < EVENT_TH_TOL and
                    abs(c_pos[2] - ref_pos[2, k_ref]) < EVENT_Z_TOL):
                    # log achievement error before advancing: how far from the waypoint at trigger time
                    k_ref += 1
        elif not _USE_CYL_REF:
            k_ref = k   # Cartesian custom mode: reference is time-indexed, follow lockstep
        if _t_complete is None and TRAJ_MODE == "test_time_air":
            _d_last = np.sqrt((pos[0]-_lw_x)**2 + (pos[1]-_lw_y)**2 + (pos[2]-_lw_z)**2)
            if _d_last < _COMPLETE_RADIUS:
                _t_complete = t[k]
        # MPC planner query — overrides k_ref and rp/rv when active
        _mpc_active = False
        if OA_MODE == "mpc":
            _rp, _rv, k_ref, _mpc_active = _mpc.query(
                k_ref, ref_pos, ref_vel, pos, vel)
        else:
            _rp = ref_pos[:, k_ref].copy()
            _rv = ref_vel[:, k_ref]
        _ra = ref_acc[:, k_ref]
        _ry = ref_yaw[  k_ref]
        # Cartesian setpoint mode: gradient-of-step gives useless 1000 m/s spikes —
        # zero them out so only the PD position error drives the controller.
        if not _USE_CYL_REF:
            _rv = np.zeros(3)
            _ra = np.zeros(3)

        if OA_MODE == "apf_ref":
            # ── Reference deflection (both modes) ────────────────────────────────
            # apf_ref: this IS the avoidance mechanism.
            # apf:     this prevents the PID attractive force from fighting the APF
            #          repulsion (local minimum) and stops integrator windup while
            #          the drone is held off the planned path by the obstacle field.
            if _USE_CYL_REF:
                _rp_xyz = np.array([_rp[0]*np.cos(_rp[1]),
                                     _rp[0]*np.sin(_rp[1]),
                                     _rp[2]])
            else:
                _rp_xyz = _rp.copy()
            # Asymmetric low-pass: normal tau while force is growing (approach),
            # 4× faster decay once force is shrinking (obstacle cleared).
            _raw     = _oa_raw(_rp_xyz, c_vel)
            _tau_eff = _oa_tau if (np.linalg.norm(_raw) >= np.linalg.norm(_oa_defl)) \
                               else _oa_tau * 0.25
            _alpha      = dt / (dt + _tau_eff)
            _oa_defl    = _oa_defl + _alpha * (_raw - _oa_defl)
            APF_defl_log[:, k] = _oa_defl
            _rp_xyz_mod = _rp_xyz + _oa_defl
            if _USE_CYL_REF:
                _r_mod = max(np.sqrt(_rp_xyz_mod[0]**2 + _rp_xyz_mod[1]**2), 1e-6)
                _rp = np.array([_r_mod,
                                 np.arctan2(_rp_xyz_mod[1], _rp_xyz_mod[0]),
                                 _rp_xyz_mod[2]])
            else:
                _rp = _rp_xyz_mod
            # Braking: reduce feedforward acceleration opposing velocity
            _speed_scale = _oa_speed(_rp_xyz, np.linalg.norm(c_vel))
            if _speed_scale < 1.0:
                _vel_n = np.linalg.norm(c_vel)
                if _vel_n > 0.1:
                    _ra = _ra - (1.0 - _speed_scale) * 3.0 * (c_vel / _vel_n)

        if OA_MODE != "none":
            # Track min surface distance to each obstacle (true drone position, both modes)
            for _oi, _ob in enumerate(_obs_list):
                _op = _obs_cyl2xyz(_ob['pos'])
                _ds = max(np.linalg.norm(pos - _op) - float(_ob['radius']), 0.0)
                if _ds < _oa_min_dist[_oi]:
                    _oa_min_dist[_oi] = _ds

        Ref_log[:, k]     = _rp               # log active reference position
        Ref_vel_log[:, k] = ref_vel[:, k_ref]  # log active reference velocity

        # outer PID + feedforward
        if _USE_CYL_REF:
            r_m = max(np.sqrt(c_pos[0]**2 + c_pos[1]**2), 1e-6)
            θ_m = np.arctan2(c_pos[1], c_pos[0])
            cs  = np.cos(θ_m);  sn = np.sin(θ_m)

            e_r = _rp[0] - r_m
            e_θ = np.arctan2(np.sin(_rp[1] - θ_m), np.cos(_rp[1] - θ_m))
            e_z = _rp[2] - c_pos[2]
            e_t = r_m * e_θ

            ṙ_m  =  c_vel[0] * cs + c_vel[1] * sn
            θ̇_m  = (-c_vel[0] * sn + c_vel[1] * cs) / r_m
            e_ṙ  = _rv[0] - ṙ_m
            e_ṫ  = r_m * (_rv[1] - θ̇_m)
            e_ż  = _rv[2] - c_vel[2]

            int_pos = np.clip(int_pos + np.array([e_r, e_t, e_z]) * dt,
                              -cyl_i_lim, cyl_i_lim)

            a_r = cyl_Kp[0]*e_r + cyl_Ki[0]*int_pos[0] + cyl_Kd[0]*e_ṙ
            a_t = cyl_Kp[1]*e_t + cyl_Ki[1]*int_pos[1] + cyl_Kd[1]*e_ṫ
            a_z = cyl_Kp[2]*e_z + cyl_Ki[2]*int_pos[2] + cyl_Kd[2]*e_ż

            a_cmd = np.array([a_r * cs - a_t * sn,
                              a_r * sn + a_t * cs,
                              a_z]) + _ra
        else:
            e_pos = _rp - c_pos
            e_vel = _rv - c_vel
            int_pos = np.clip(int_pos + e_pos * dt, -cyl_i_lim, cyl_i_lim)
            a_cmd = cyl_Kp * e_pos + cyl_Ki * int_pos + cyl_Kd * e_vel + _ra

        if OA_MODE == "apf":
            # ── True APF: inject repulsive acceleration at drone's actual position ──
            # Evaluated at pos (truth), not the reference — position integrator untouched.
            # Reference passed as Cartesian so the tangential force steers toward the goal.
            if _USE_CYL_REF:
                _rp_apf_xyz = np.array([_rp[0]*np.cos(_rp[1]), _rp[0]*np.sin(_rp[1]), _rp[2]])
            else:
                _rp_apf_xyz = _rp.copy()
            _apf = _oa_apf_acc(pos, c_vel, _rp_apf_xyz)
            a_cmd[0] += _apf[0]
            a_cmd[1] += _apf[1]
            # Braking: direct deceleration feedforward opposing velocity
            _vel_n = np.linalg.norm(c_vel)
            if _vel_n > 0.1:
                _spd = _oa_apf_speed(pos, _vel_n)
                if _spd < 1.0:
                    a_cmd -= (1.0 - _spd) * 3.0 * (c_vel / _vel_n)

        _T_max = 4.0 * p.kT * p.omega_max**2
        T_cmd  = np.clip(p.m * (a_cmd[2] + p.g), 0.1 * p.m * p.g, _T_max)

        theta_d = ( a_cmd[0]*np.cos(c_psi) + a_cmd[1]*np.sin(c_psi)) * p.m / T_cmd
        phi_d   = ( a_cmd[0]*np.sin(c_psi) - a_cmd[1]*np.cos(c_psi)) * p.m / T_cmd
        psi_d   = _ry

        theta_d = np.clip(theta_d, -att_lim, att_lim)
        phi_d   = np.clip(phi_d,   -att_lim, att_lim)

        # inner PID: attitude error -> body torques
        e_att    = np.array([phi_d, theta_d, psi_d]) - c_euler
        e_att[2] = np.arctan2(np.sin(e_att[2]), np.cos(e_att[2]))   # yaw wrap ±π
        int_att  = np.clip(int_att + e_att * dt, -att_i_lim, att_i_lim)

        # Derivative on measured body rate — avoids derivative kick on ref step
        tau_cmd = att_Kp * e_att + att_Ki * int_att - att_Kd * c_wb

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

        # Crash / runaway abort: drone fell below ground plane — sim is clearly broken
        if X[2, k+1] < -5.0:
            print(f"\n[SIM] ABORT at t={t[k]:.2f}s — drone z={X[2, k+1]:.1f}m < -5 m. "
                  "Showing plots up to this point.")
            _k_end = k + 1
            break

        if ENABLE_EKF:
            # Sensors always observe true plant states — independent of USE_EKF_FEEDBACK
            _a_inertial = (X[6:9, k+1] - X[6:9, k]) / dt
            for _j in range(_EKF_SUBSTEPS):
                _t_sub = t[k] + _j * _DT_EKF
                _meas  = _suite.tick(_t_sub, _DT_EKF, _a_inertial, euler, wb, pos, vel)
                _ekf_x = _ekf.update(_meas, _DT_EKF)
            X_ekf[:, k]     = _ekf_x
            _ekf_feedback_x = _ekf_x   # available as controller input next step

        if _t_complete is not None:
            _k_end = k + 1
            break

    if OA_MODE != "none":
        print(f"\n[OA] Obstacle proximity report  (mode={OA_MODE!r}):")
        for _oi, _ob in enumerate(_obs_list):
            _md = _oa_min_dist[_oi]
            _status = "COLLISION" if _md <= 0.0 else "clear"
            print(f"  Obs {_oi+1}  r={_ob['pos'][0]}m θ={_ob['pos'][1]}° z={_ob['pos'][2]}m "
                  f"rad={_ob['radius']}m  →  min dist to surface = {_md:.3f} m  [{_status}]")

    _n_valid = _k_end + 1
    t       = t      [:_n_valid]
    X       = X      [:, :_n_valid]
    U_log   = U_log  [:, :_n_valid]
    Wr_log  = Wr_log [:, :_n_valid]
    Ref_log     = Ref_log[:, :_n_valid]
    Ref_vel_log = Ref_vel_log[:, :_n_valid]
    APF_defl_log = APF_defl_log[:, :_n_valid]
    ref_pos = ref_pos[:, :_n_valid]
    ref_vel = ref_vel[:, :_n_valid]
    ref_acc = ref_acc[:, :_n_valid]
    ref_yaw = ref_yaw[   :_n_valid]
    if ENABLE_EKF:
        X_ekf = X_ekf[:, :_n_valid]
    N = _n_valid

    U_log[:, -1] = U_log[:, -2]
    if ENABLE_EKF:
        X_ekf[:, -1] = X_ekf[:, -2]

    if SAVE_SIM_DATA:
        from pathlib import Path
        _npz_path = Path(__file__).parent / 'sim_data_SC.npz'
        _n_save = max(1, int(len(t) * min(100, max(1, SAVE_PCT)) / 100))
        np.savez(str(_npz_path),
                 t=t[:_n_save], X=X[:, :_n_save],
                 dt=np.float64(dt), R_base=np.float64(R_base))
        print(f"[SIM] State data saved → {_npz_path}  "
              f"({_n_save} steps at full dt={dt:.4f}s, {SAVE_PCT}% of {len(t)})")

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
    # ref_pos_cart: what the controller actually commanded at each step (from Ref_log)
    if _USE_CYL_REF:
        _r  = Ref_log[0];  _θ = Ref_log[1]
        ref_pos_cart = np.array([_r * np.cos(_θ),
                                  _r * np.sin(_θ),
                                  Ref_log[2]])
    else:
        ref_pos_cart = Ref_log

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
        if _t_complete is not None:
            _insp_label = f"{_t_complete:.1f} s  ({_t_complete/60:.2f} min)"
        else:
            _insp_label = f"N/A  [trajectory not completed within {t_end:.0f} s]"
        print(f"\n{'═'*56}")
        print(f"  Total inspection time (SC): {_insp_label}")
        print(f"{'═'*56}")

    if PLOT_MODE == "sim" and TRAJ_MODE == "test_time_air":
        # ── Scan quality metric ─────────────────────────────────────────────
        # Lateral deviation during vertical scan segments only (z changes between
        # consecutive waypoints). Filter by time so points from other strips
        # (which share the same z-range) are not included.
        # Waypoints: _wp[:, 0]=t  [1]=r  [2]=theta  [3]=z  (cylindrical)
        _wp_x = _wp[:, 1] * np.cos(_wp[:, 2])
        _wp_y_arr = _wp[:, 1] * np.sin(_wp[:, 2])
        _wp_z_arr = _wp[:, 3]
        _wp_t_arr = _wp[:, 0]
        _true_x = X[0, :_n_valid]
        _true_y = X[1, :_n_valid]
        _true_t = t[:_n_valid]
        _scan_errors = []
        for _si in range(len(_wp_x) - 1):
            if abs(_wp_z_arr[_si + 1] - _wp_z_arr[_si]) < 1e-3:
                continue   # horizontal segment — skip
            _xr, _yr = _wp_x[_si], _wp_y_arr[_si]
            _mask = (_true_t >= _wp_t_arr[_si]) & (_true_t <= _wp_t_arr[_si + 1])
            if not _mask.any():
                continue
            _scan_errors.append(np.sqrt((_true_x[_mask] - _xr)**2 + (_true_y[_mask] - _yr)**2))
        print(f"\n{'═'*56}")
        print("  SCAN QUALITY  (lateral deviation, vertical strips only)")
        print(f"{'═'*56}")
        if _scan_errors:
            _all_err = np.concatenate(_scan_errors)
            print(f"  {'Scan segments':<34}  {len(_scan_errors)}")
            print(f"  {'Points evaluated':<34}  {len(_all_err)}")
            print(f"  {'Mean lateral error':<34}  {np.mean(_all_err)*1e2:.2f} cm")
            print(f"  {'Max  lateral error':<34}  {np.max(_all_err)*1e2:.2f} cm")
            print(f"  {'Min  lateral error':<34}  {np.min(_all_err)*1e2:.2f} cm")
        else:
            print("  No vertical scan segments found in waypoints.")
        print(f"{'═'*56}")

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
            f"Sequential closed-loop pole analysis  —  gains: 'lawnmower_ekf'  |  Ki omitted",
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

        if show_plots:
            plt.show()

    else:  # PLOT_MODE == "sim"
        _ps        = max(1, N // 10_000)
        def _zclip(a): return np.where(np.abs(a) < 1e-10, 0.0, a)
        t_p        = t[::_ps]
        X_p        = _zclip(X[:, ::_ps])
        rp_p       = _zclip(ref_pos_cart[:, ::_ps])
        Ul_p       = _zclip(U_log[:, ::_ps])
        ref_yaw_p  = _zclip(ref_yaw[::_ps])

        if _USE_CYL_REF:
            _r_p  = np.maximum(np.sqrt(X_p[0,:]**2 + X_p[1,:]**2), 1e-6)
            _θ_p  = np.unwrap(np.arctan2(X_p[1,:], X_p[0,:]))
            _z_p  = X_p[2,:]
            _vr_p =  X_p[6,:]*np.cos(_θ_p) + X_p[7,:]*np.sin(_θ_p)
            _vθ_p = (-X_p[6,:]*np.sin(_θ_p) + X_p[7,:]*np.cos(_θ_p)) / _r_p
            _vz_p =  X_p[8,:]

            _rr_p  = Ref_log[0, ::_ps];     _θr_p = Ref_log[1, ::_ps];     _zr_p = Ref_log[2, ::_ps]
            _vr_r  = Ref_vel_log[0, ::_ps]; _vθ_r = Ref_vel_log[1, ::_ps]; _vz_r = Ref_vel_log[2, ::_ps]

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
            _leg_h = [ax.plot([], [], 'b', lw=1.6, label='Actual')[0]]
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
            vel_ref     = [Ref_vel_log[0, ::_ps], Ref_vel_log[1, ::_ps], Ref_vel_log[2, ::_ps]]

        for i in range(3):
            ax = axes2[i]
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

        # error figure placeholder — error metric to be defined

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

        # fig 4b: thrust per rotor
        _T_p = p.kT * X_p[12:16, :]**2   # (4, N_plot) — always positive
        fig4b, axes4b = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        fig4b.suptitle("Rotor Thrust", fontsize=13)
        _T_hover = p.kT * p.omega_h**2
        for i in range(4):
            ax = axes4b[i]
            ax.plot(t_p, _T_p[i], 'b', lw=1.6)
            ax.axhline(_T_hover, color='r', ls='--', lw=1.0, label=f'Hover T = {_T_hover:.2f} N')
            ax.set_ylabel(f"{motor_names[i]}  [N]"); ax.grid(True)
            if i == 0: ax.legend(loc='lower right')
            shade_gusts(ax)
        axes4b[-1].set_xlabel("Time  [s]")
        for ax in fig4b.axes: ax.tick_params(labelbottom=True)
        fig4b.tight_layout()

        if PLOT_VIBRATION:
            _wr_p   = X_p[12:16, :]                      # (4, N) rotor speeds [rad/s]
            _f1     = np.abs(_wr_p) / (2 * np.pi)        # 1Ω  [Hz]  per motor
            _f_bp   = N_BLADES * _f1                      # blade-pass [Hz] per motor

            _f1_mean   = _f1.mean(axis=0)
            _f1_lo     = _f1.min(axis=0)
            _f1_hi     = _f1.max(axis=0)
            _fbp_mean  = _f_bp.mean(axis=0)
            _fbp_lo    = _f_bp.min(axis=0)
            _fbp_hi    = _f_bp.max(axis=0)

            fig_vib, ax_vib = plt.subplots(figsize=(11, 4))
            fig_vib.suptitle("Vibration Excitation Frequency Envelope  (1Ω  &  Blade-Pass)", fontsize=13)

            ax_vib.fill_between(t_p, _f1_lo,  _f1_hi,  alpha=0.25, color='steelblue')
            ax_vib.fill_between(t_p, _fbp_lo, _fbp_hi, alpha=0.25, color='tomato')
            ax_vib.plot(t_p, _f1_mean,  color='steelblue', lw=1.5, label=f'1Ω  (motor, mean)')
            ax_vib.plot(t_p, _fbp_mean, color='tomato',    lw=1.5, label=f'{N_BLADES}Ω  (blade-pass, mean)')

            ax_vib.set_xlabel("Time  [s]")
            ax_vib.set_ylabel("Frequency  [Hz]")
            ax_vib.legend(loc='upper right')
            ax_vib.grid(True)
            shade_gusts(ax_vib)
            fig_vib.tight_layout()

        fig5, ax5 = plt.subplots(figsize=(7, 6))
        poles = np.linalg.eigvals(A_lin)
        ax5.scatter(poles.real, poles.imag, marker='x', s=80, color='b', zorder=5)
        ax5.axvline(0, color='k', lw=0.8, ls='--')
        ax5.axhline(0, color='k', lw=0.8, ls='--')
        ax5.set_xlabel("Real"); ax5.set_ylabel("Imaginary")
        ax5.set_title("Open-Loop Poles at Hover (linearised)")
        ax5.grid(True)
        fig5.tight_layout()

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

        if ENABLE_EKF:
            # EKF truth vs estimate — 4 figures, one per state group
            # Plant state ordering: [pos(3), euler(3), vel(3), wb(3), rotors(4)]
            # EKF state ordering:   [pos(3), vel(3),   euler(3), wb(3)]

            # Figure EKF-1: Position
            fig_e1, axes_e1 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e1.suptitle("EKF: Position  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['x  [m]', 'y  [m]', 'z  [m]']):
                axes_e1[i].plot(t_p, X_p[i],               color='steelblue', lw=1.5, label='Truth')
                axes_e1[i].plot(t_p, X_ekf[i,   ::_ps],    color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e1[i].set_ylabel(lbl); axes_e1[i].grid(True)
                if i == 0: axes_e1[i].legend(loc='upper right')
                shade_gusts(axes_e1[i])
            axes_e1[-1].set_xlabel("Time  [s]")
            for ax in fig_e1.axes: ax.tick_params(labelbottom=True)
            fig_e1.tight_layout()

            # Figure EKF-2: Velocity
            fig_e2, axes_e2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e2.suptitle("EKF: Velocity  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]']):
                axes_e2[i].plot(t_p, X_p[6+i],             color='steelblue', lw=1.5, label='Truth')
                axes_e2[i].plot(t_p, X_ekf[3+i, ::_ps],    color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e2[i].set_ylabel(lbl); axes_e2[i].grid(True)
                if i == 0: axes_e2[i].legend(loc='upper right')
                shade_gusts(axes_e2[i])
            axes_e2[-1].set_xlabel("Time  [s]")
            for ax in fig_e2.axes: ax.tick_params(labelbottom=True)
            fig_e2.tight_layout()

            # Figure EKF-3: Euler angles (degrees)
            fig_e3, axes_e3 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e3.suptitle("EKF: Euler Angles  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['φ  [°]', 'θ  [°]', 'ψ  [°]']):
                _truth_ang = np.arctan2(np.sin(X_p[3+i]), np.cos(X_p[3+i]))  # wrap to ±π
                axes_e3[i].plot(t_p, np.degrees(_truth_ang),         color='steelblue', lw=1.5, label='Truth')
                axes_e3[i].plot(t_p, np.degrees(X_ekf[6+i, ::_ps]), color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e3[i].set_ylabel(lbl); axes_e3[i].grid(True)
                if i == 0: axes_e3[i].legend(loc='upper right')
                shade_gusts(axes_e3[i])
            axes_e3[-1].set_xlabel("Time  [s]")
            for ax in fig_e3.axes: ax.tick_params(labelbottom=True)
            fig_e3.tight_layout()

            # Figure EKF-4: Body rates
            fig_e4, axes_e4 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e4.suptitle("EKF: Body Rates  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['p  [rad/s]', 'q  [rad/s]', 'r  [rad/s]']):
                axes_e4[i].plot(t_p, X_p[9+i],             color='steelblue', lw=1.5, label='Truth')
                axes_e4[i].plot(t_p, X_ekf[9+i, ::_ps],    color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e4[i].set_ylabel(lbl); axes_e4[i].grid(True)
                if i == 0: axes_e4[i].legend(loc='upper right')
                shade_gusts(axes_e4[i])
            axes_e4[-1].set_xlabel("Time  [s]")
            for ax in fig_e4.axes: ax.tick_params(labelbottom=True)
            fig_e4.tight_layout()

        # ── 3D flight path (created last so it opens on top) ─────────────────────
        fig6 = plt.figure(figsize=(11, 10))
        ax6  = fig6.add_subplot(111, projection='3d')

        if TRAJ_MODE == "test_time_air":
            from Test_time import (R_base as _R_base, R_top as _R_top,
                                   H_air_cyl as _H_cyl, H_air_cone as _H_cone,
                                   blade_chord as _blade_chord, blade_span as _blade_span)
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

            # Blade slab surface (flat rectangle at y=0 in blade-local frame)
            _xb_sg, _zb_sg = np.meshgrid(
                np.linspace(-_blade_chord / 2, _blade_chord / 2, 4),
                np.linspace(0, _blade_span, 30))
            _yb_sg = np.zeros_like(_xb_sg)

            for _i, _ang in enumerate([0, 120, 240]):
                _col  = 'orange' if (AERIAL_PHASE == "turbine" and _i == 0) else 'gold'
                _alph = 0.55     if (AERIAL_PHASE == "turbine" and _i == 0) else 0.20
                _xr, _yr, _zr = _blade_transform(_xb_sg, _yb_sg, _zb_sg, _ang, _H_air)
                ax6.plot_surface(_xr, _yr, _zr, color=_col, alpha=_alph, edgecolor='none')

        if PLOT_REFERENCE:
            if TRAJ_MODE == "test_time_air":
                _rx = _wp[:, 1] * np.cos(_wp[:, 2])
                _ry = _wp[:, 1] * np.sin(_wp[:, 2])
                _rz = _wp[:, 3]
            else:
                _rx, _ry, _rz = rp_p[0, :], rp_p[1, :], rp_p[2, :]
            ax6.plot(_rx, _ry, _rz,
                     color='red', lw=1.8, ls='--', label='Reference', zorder=5)
        ax6.plot(X_p[0, :], X_p[1, :], X_p[2, :],
                 color='blue', lw=1.8, label='Actual (controller)', zorder=6)

        if OA_MODE == "apf_ref":
            # smooth planned ref in Cartesian + per-step APF deflection offset
            _ps_d = X_p.shape[1]  # same downsample length as X_p
            _ds = max(1, X.shape[1] // _ps_d)
            _rp_s = ref_pos[0, ::_ds];  _th_s = ref_pos[1, ::_ds];  _z_s = ref_pos[2, ::_ds]
            _defl_s = APF_defl_log[:, ::_ds]
            _apf_x = _rp_s * np.cos(_th_s) + _defl_s[0]
            _apf_y = _rp_s * np.sin(_th_s) + _defl_s[1]
            _apf_z = _z_s                  + _defl_s[2]
            ax6.plot(_apf_x[:_ps_d], _apf_y[:_ps_d], _apf_z[:_ps_d],
                     color='purple', lw=1.5, ls=':', label='APF-deflected reference', zorder=7)

        # start / end markers on actual path
        ax6.scatter(*X_p[:3,  0], color='green',  s=40, zorder=7)
        ax6.scatter(*X_p[:3, -1], color='orange', s=40, zorder=7)
        ax6.text(X_p[0,  0], X_p[1,  0], X_p[2,  0], '  start', fontsize=7, color='green')
        ax6.text(X_p[0, -1], X_p[1, -1], X_p[2, -1], '  end',   fontsize=7, color='orange')

        if OA_MODE != "none":
            _u_s = np.linspace(0, 2*np.pi, 30)
            _v_s = np.linspace(0, np.pi,   20)
            _legend_obs_added  = False   # add legend proxy patches once only
            _legend_safe_added = False
            for _oi, _obs in enumerate(_obs_list):
                _cx, _cy, _cz = _obs_cyl2xyz(_obs['pos'])
                _r  = _obs['radius']
                _xs = _cx + _r * np.outer(np.cos(_u_s), np.sin(_v_s))
                _ys = _cy + _r * np.outer(np.sin(_u_s), np.sin(_v_s))
                _zs = _cz + _r * np.outer(np.ones_like(_u_s), np.cos(_v_s))
                ax6.plot_surface(_xs, _ys, _zs, color='red', alpha=0.6, edgecolor='none')
                # center marker + per-obstacle label
                ax6.scatter(_cx, _cy, _cz, color='darkred', s=60, zorder=9)
                ax6.text(_cx, _cy, _cz,
                         f'  obs{_oi+1} r={_r}m', fontsize=7, color='darkred')
                # safety / clearance bubble (physical radius + R_BASE)
                _rs = _r + _obs_Rbase
                _xs2 = _cx + _rs * np.outer(np.cos(_u_s), np.sin(_v_s))
                _ys2 = _cy + _rs * np.outer(np.sin(_u_s), np.sin(_v_s))
                _zs2 = _cz + _rs * np.outer(np.ones_like(_u_s), np.cos(_v_s))
                ax6.plot_surface(_xs2, _ys2, _zs2, color='orange', alpha=0.15, edgecolor='none')

            # Legend proxy patches for obstacle layers (matplotlib can't auto-legend
            # plot_surface, so we add invisible Line2D handles with the right colours)
            import matplotlib.lines as _mlines
            import matplotlib.patches as _mpatches
            _obs_patch  = _mpatches.Patch(color='red',    alpha=0.6,
                                           label=f'Obstacle body (actual size)')
            _safe_patch = _mpatches.Patch(color='orange', alpha=0.4,
                                           label=f'Safety clearance shell  '
                                                 f'(body + {_obs_Rbase:.1f} m R_BASE)')
            _extra_handles = [_obs_patch, _safe_patch]

        else:
            _extra_handles = []

        ax6.set_xlabel("X  [m]")
        ax6.set_ylabel("Y  [m]")
        ax6.set_zlabel("Z  [m]")
        ax6.set_title(f"3D Flight Path: Reference vs Actual  [{TRAJ_MODE}]", fontsize=13)
        _handles, _labels = ax6.get_legend_handles_labels()
        ax6.legend(handles=_handles + _extra_handles, loc='upper left', fontsize=8)

        if TRAJ_MODE == "test_time_air":
            # axis limits from reference trajectory + any obstacle spheres
            _all_x = list(_rx) + list(X_p[0])
            _all_y = list(_ry) + list(X_p[1])
            _all_z = list(_rz) + list(X_p[2])
            if OA_MODE != "none":
                for _ob in _obs_list:
                    _ox, _oy, _oz = _obs_cyl2xyz(_ob['pos'])
                    _ro = _ob['radius'] + _obs_Rbase   # include safety shell in axis bounds
                    _all_x += [_ox - _ro, _ox + _ro]
                    _all_y += [_oy - _ro, _oy + _ro]
                    _all_z += [_oz - _ro, _oz + _ro]
            x_mid = (max(_all_x) + min(_all_x)) / 2
            y_mid = (max(_all_y) + min(_all_y)) / 2
            z_mid = (max(_all_z) + min(_all_z)) / 2
            half  = max(max(_all_x) - min(_all_x),
                        max(_all_y) - min(_all_y),
                        max(_all_z) - min(_all_z)) / 2 * 1.3   # 30% margin
            ax6.set_xlim(x_mid - half, x_mid + half)
            ax6.set_ylim(y_mid - half, y_mid + half)
            ax6.set_zlim(z_mid - half, z_mid + half)
            ax6.set_box_aspect([1, 1, 1])

        ax6.view_init(elev=25, azim=45)   # RHR z-up: elev from xy-plane, azim CCW from +x
        fig6.tight_layout()

        if show_plots:
            plt.show()

    if not show_plots:
        plt.close('all')

    _crashed       = (X[2, _k_end] < -4.0)
    _pos_err       = np.linalg.norm(X[:3, :_n_valid] - ref_pos_cart[:3, :_n_valid], axis=0)
    _euler_abs_deg = np.abs(np.degrees(X[3:6, :_n_valid]))
    _yaw_rate_dps  = np.abs(np.degrees(X[11, :_n_valid]))
    _result = {
        'rms_pos_error':    float(np.sqrt(np.mean(_pos_err**2))),
        'max_pos_error':    float(np.max(_pos_err)),
        'crashed':          bool(_crashed),
        't_complete':       float(_t_complete) if _t_complete is not None else None,
        'max_roll_deg':     float(np.max(_euler_abs_deg[0])),
        'max_pitch_deg':    float(np.max(_euler_abs_deg[1])),
        'max_att_deg':      float(np.max(_euler_abs_deg[:2])),
        'max_yaw_rate_dps': float(np.max(_yaw_rate_dps)),
    }
    if _ov.get('return_timeseries', False):
        _result['t'] = t[:_n_valid].copy()
        _result['X'] = X[:, :_n_valid].copy()
    return _result


if __name__ == "__main__":
    run_sim()
