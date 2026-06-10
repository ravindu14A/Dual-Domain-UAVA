"""
quadcopterUW.py  --  UAUV Underwater Phase: Stability & Control
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

20-State 6-DOF nonlinear underwater model + cascaded PID + pseudo-inverse allocation.

STATE (20):  [x  y  z | phi  theta  psi | xd  yd  zd | p  q  r | w1  w2  w3  w4 | b1  b2  b3  b4]
              pos(3)    euler(3)           vel(3)        ang_rate(3) prop_speed(4)    servo_angle(4)

z CONVENTION: z=0 at water surface, NEGATIVE downward.

ACTUATORS  (body frame, corners at CoM height z=0):
  FL(0) r=[ Lx, Ly,0]  FR(1) r=[ Lx,-Ly,0]  RL(2) r=[-Lx, Ly,0]  RR(3) r=[-Lx,-Ly,0]
  U_base = [±cos γ, ±sin γ, 0]  (γ = inward angle from wall)
  v_i = cos(β_i)·U_base_i + sin(β_i)·ez      β=0→horizontal, β=π/2→up, β<0→down
  h_dir = [+1, -1, -1, +1]  (CCW, CW, CW, CCW)

CONTROL:
  Outer cylindrical PID → a_cmd (inertial) → F_body_des = M_ctrl·R^T·a_cmd − F_gb_body
  Inner attitude PID → [τ_φ, τ_θ, τ_ψ]
  Allocation:  T_v = A_VERT_PINV  @ [Fz, τ_φ, τ_θ]    (per-thruster vertical)
               T_h = A_HORIZ_PINV @ [Fx, Fy, τ_ψ]     (per-thruster horizontal)
               β_i = atan2(T_v[i], T_h[i]);  ω_i from |T| via kT

Dependencies: numpy matplotlib tqdm
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from geometry import (
    L_box, W_box, H_box,
    V_pill,
    total_mass  as _geo_mass,
    Ixx_uw      as _geo_Ixx,
    Iyy_uw      as _geo_Iyy,
    Izz_uw      as _geo_Izz,
)
from underwater_props import kT as _kT_fwd, kT_rev as _kT_rev
from Test_time import (
    water_config, cameras,
    R_base, H_water,
    _water_timed_waypoints, write_matlab_traj,
)

# ─────────────────────────────────────────────────────────────────────
#  SIMULATION CONFIG  (edit here)
# ─────────────────────────────────────────────────────────────────────

TRAJ_MODE = "test_time_water"
TRAJ_PCT  = 10   # [1-100] percentage of test_time_water waypoints to use
#   "hold"            -- hold at origin
#   "custom"          -- cylindrical (r, θ, z) segments (list of tuples)
#   "test_time_water" -- full underwater lawnmower from water_config

TRAJ_SEGMENTS = [
    (0.0,  6.0, 0.0,   0.0),
    (30.0, 6.0, 0.0, -60.0),
]   # only used when TRAJ_MODE = "custom"

USE_EVENT_TRIG  = True
EVENT_R_TOL     = 1.25    # [m]   radial standoff tolerance
EVENT_TH_TOL    = 0.15    # [rad] azimuth tolerance
EVENT_Z_TOL     = 1.25    # [m]   height tolerance

USE_DRAG_FF = False   # body-frame drag feedforward
USE_MASS_FF = False   # added-mass compensation in force command
USE_CENT_FF = True    # centripetal/Coriolis feedforward

DIST_ENABLED = False
DISTURBANCES = [
    # t_on  t_off  Fx   Fy   Fz   tx   ty   tz
    (20.0, 22.0,  3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
]

IMPULSE_ENABLED = False
IMPULSES = [
    #  t [s]  Jx    Jy   Jz   Jtx  Jty  Jtz
    (20.0,  10.0, 0.0, 0.0, 0.0, 0.0, 0.0),
]

CURRENT_ENABLED = False
CURRENT_INTERP  = 'linear'
CURRENT_PROFILE = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
])

T_BUFFER = 500    # [s] extra sim time to run after the last waypoint is reached

PLOT_MODE      = "sim"   # "sim" | "root_locus"
PLOT_REFERENCE = True
PLOT_ACTUAL    = True
ENABLE_PLOTS   = PLOT_MODE in ("sim", "root_locus")
PLOT_VIBRATION = False    # vibration frequency envelope (1Ω + blade-pass) vs time
N_BLADES       = 2       # number of propeller blades
SAVE_SIM_DATA  = True   # save state data to sim_data_UW.npz (for kalman_UW.py standalone)
SAVE_PCT       = 100      # % of sim to save (first N%) at full resolution

# ── EKF ──────────────────────────────────────────────────────
USE_EKF = True   # True:  EKF runs, estimates feed controller + waypoint manager,
                   #        and EKF vs truth plots are added to output figures
                   # False: controller uses true plant states, no EKF plots

LIN_DEPTH = -20.0    # [m]   linearisation depth
LIN_PSI   = np.pi   # [rad] linearisation yaw (π = facing tower)
LIN_VEL   = np.array([0.0, 0.0, 0.0])

# ─────────────────────────────────────────────────────────────────────
#  VEHICLE & ENVIRONMENT PARAMETERS
# ─────────────────────────────────────────────────────────────────────


def run_sim(overrides=None, show_plots=True):
    """Run simulation. overrides dict keys (all optional):
       mass_factor      float  multiplier on total mass              (default 1.0)
       ixx_factor       float  multiplier on Ixx roll inertia        (default 1.0)
       iyy_factor       float  multiplier on Iyy pitch inertia       (default 1.0)
       izz_factor       float  multiplier on Izz yaw inertia         (default 1.0)
       tau_m_factor     float  multiplier on motor time constant     (default 1.0)
       traj_mode        str    override TRAJ_MODE                    (default file value)
       dist_enabled     bool   override DIST_ENABLED                 (default file value)
       gust_force       float  lateral gust [N] for 0.5 s at t=5 s  (enables dist automatically)
       impulse_enabled  bool   override IMPULSE_ENABLED              (default file value)
       impulse_force    float  lateral impulse [N·s] at t=5 s       (enables impulse automatically)
       current_enabled  bool   override CURRENT_ENABLED              (default file value)
       current_speed    float  set constant lateral current [m/s]    (enables current automatically)
       t_buffer         float  override T_BUFFER sim duration [s]    (default file value)
    """
    _ov = overrides or {}
    _mass_factor  = float(_ov.get('mass_factor',  1.0))
    _ixx_factor   = float(_ov.get('ixx_factor',   1.0))
    _iyy_factor   = float(_ov.get('iyy_factor',   1.0))
    _izz_factor   = float(_ov.get('izz_factor',   1.0))
    _tau_m_factor = float(_ov.get('tau_m_factor', 1.0))

    # Shadow module-level config with overridable locals
    _g = globals()
    TRAJ_MODE        = _ov.get('traj_mode',        _g['TRAJ_MODE'])
    TRAJ_SEGMENTS    = _ov.get('traj_segments',    _g['TRAJ_SEGMENTS'])
    USE_EKF          = _ov.get('ekf_enabled',      _g['USE_EKF'])
    DIST_ENABLED     = _ov.get('dist_enabled',     _g['DIST_ENABLED'])
    IMPULSE_ENABLED  = _ov.get('impulse_enabled',  _g['IMPULSE_ENABLED'])
    CURRENT_ENABLED  = _ov.get('current_enabled',  _g['CURRENT_ENABLED'])
    DISTURBANCES     = list(_g['DISTURBANCES'])
    IMPULSES         = list(_g['IMPULSES'])
    CURRENT_PROFILE  = _g['CURRENT_PROFILE'].copy()
    T_BUFFER         = float(_ov.get('t_buffer',   _g['T_BUFFER']))

    # Convenience: constant lateral current (automatically enables)
    if 'current_speed' in _ov:
        _cs = float(_ov['current_speed'])
        CURRENT_PROFILE = np.array([[0.0, _cs, 0.0, 0.0], [99999.0, _cs, 0.0, 0.0]])
        CURRENT_ENABLED = True

    # Convenience: sustained lateral gust force for 0.5 s at t=5 s
    if 'gust_force' in _ov:
        DISTURBANCES = [(5.0, 5.5, float(_ov['gust_force']), 0.0, 0.0, 0.0, 0.0, 0.0)]
        DIST_ENABLED = True

    # Convenience: lateral impulse at t=5 s
    if 'impulse_force' in _ov:
        IMPULSES = [(5.0, float(_ov['impulse_force']), 0.0, 0.0, 0.0, 0.0, 0.0)]
        IMPULSE_ENABLED = True

    class Params:
        m   = _geo_mass * _mass_factor
        Ixx = _geo_Ixx * _ixx_factor;  Iyy = _geo_Iyy * _iyy_factor;  Izz = _geo_Izz * _izz_factor
        g   = 9.81;      rho_water = 1025.0

        # Thruster
        kT_fwd = _kT_fwd;  kT_rev = _kT_rev  # [N·s²/rad²] — fitted from thruster datasheet
        kQ_fwd = 1.0e-5;  kQ_rev = 1e-5   # [N·m·s²/rad²]
        tau_m  = 0.06 * _tau_m_factor;    tau_srv = 0.06     # [s]
        omega_max = 500.0                    # [rad/s]
        BETA_MIN  = 0.0;  BETA_MAX = np.pi  # servo range

        gamma = np.radians(45.0)

        # Geometry / buoyancy  (pill hull — full displaced volume, no ballast factor)
        V_sub  = V_pill

        # Drag  (frontal areas approximated from bounding box)
        Cd     = np.array([1.5, 1.5, 2.0])
        A_face = np.array([W_box*H_box, L_box*H_box, L_box*W_box])

        # Added mass fractions (of ρ·V per axis)
        M_a_frac_sim  = np.array([0.10, 0.40, 0.25])
        M_a_frac_ctrl = np.array([0.10, 0.40, 0.25])

    p = Params()

    # Precomputed scalars used in the ODE hot path — avoid repeated property evaluation
    _RHO_V     = p.rho_water * p.V_sub
    _M_SIM     = np.array([p.m]*3) + p.M_a_frac_sim  * _RHO_V
    _M_CTRL    = np.array([p.m]*3) + p.M_a_frac_ctrl * _RHO_V
    _I_MAT     = np.diag([p.Ixx*1.05, p.Iyy*1.05, p.Izz*1.05])
    _F_BUOYANCY    = _RHO_V * p.g                      # full displaced-volume buoyancy
    _F_RESID       = _F_BUOYANCY - p.m * p.g           # net upward force (positive = up)
    _OMEGA_EQ      = -np.sqrt(max(_F_RESID / (4.0 * p.kT_rev), 0.0))  # hover ω (negative)
    _F_GB_VEC      = np.array([0.0, 0.0, _F_RESID])    # body-frame net buoyancy at hover

    if TRAJ_MODE == "test_time_water":
        print("=" * 60)
        print("  UNDERWATER VEHICLE PARAMETERS")
        print("=" * 60)
        print(f"  {'m':<12}  Total mass              {p.m:.4f}   kg")
        print(f"  {'Ixx/Iyy/Izz':<12}  Inertia (folded)        {p.Ixx:.3f} / {p.Iyy:.3f} / {p.Izz:.3f}   kg·m²")
        print(f"  {'kT_fwd/rev':<12}  Thrust coeffs           {p.kT_fwd:.2e} / {p.kT_rev:.2e}  N·s²/rad²")
        print(f"  {'tau_m/srv':<12}  Motor / servo lag       {p.tau_m:.2f} / {p.tau_srv:.2f}   s")
        print(f"  {'gamma':<12}  Inward angle            {np.degrees(p.gamma):.1f}   deg")
        print(f"  {'F_buoy':<12}  Full buoyancy (V_pill)  {_F_BUOYANCY:.2f}   N")
        print(f"  {'F_residual':<12}  Net upward (F_b - mg)   {_F_RESID:.2f}   N")
        print(f"  {'omega_eq':<12}  Hover ω                 {_OMEGA_EQ:.2f}   rad/s ({abs(_OMEGA_EQ)*60/(2*np.pi):.0f} RPM)")
        print(f"  {'M_a_SIM':<12}  Added mass SIM [x,y,z]  {(_M_SIM - p.m).tolist()}  kg")
        print(f"  {'Cd':<12}  Drag coeff [x,y,z]      {p.Cd.tolist()}")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────
    #  THRUSTER GEOMETRY + MIXING MATRICES
    # ─────────────────────────────────────────────────────────────────────

    _cg, _sg = np.cos(p.gamma), np.sin(p.gamma)
    _Lx, _Ly = L_box / 2.0, W_box / 2.0

    # Base horizontal direction per thruster at β=0, shape (3, 4)
    U_BASE = np.array([[ _cg, -_sg, 0],
                       [ _cg,  _sg, 0],
                       [-_cg, -_sg, 0],
                       [-_cg,  _sg, 0]], dtype=float).T

    # Prop positions, shape (3, 4)
    R_PROPS = np.array([[ _Lx,  _Ly, 0],
                        [ _Lx, -_Ly, 0],
                        [-_Lx,  _Ly, 0],
                        [-_Lx, -_Ly, 0]], dtype=float).T

    H_DIR = np.array([1.0, -1.0, -1.0, 1.0])   # CCW/CW handedness

    # Vertical mixing (β=π/2): [Fz, τ_φ, τ_θ] = A_VERT @ T_v
    # cross([rx,ry,0], T·ez) = T·[ry, -rx, 0]
    A_VERT = np.array([
        [ 1.0,  1.0,  1.0,  1.0],
        [_Ly,  -_Ly,  _Ly, -_Ly],
        [-_Lx, -_Lx,  _Lx,  _Lx],
    ], dtype=float)
    A_VERT_PINV = np.linalg.pinv(A_VERT)

    # Horizontal mixing (β=0): [Fx, Fy, τ_ψ] = A_HORIZ @ T_h
    _A_arm = _Lx * _sg + _Ly * _cg
    A_HORIZ = np.array([
        [ _cg,  _cg, -_cg, -_cg],
        [-_sg,  _sg, -_sg,  _sg],
        [-_A_arm, _A_arm, _A_arm, -_A_arm],
    ], dtype=float)
    A_HORIZ_PINV = np.linalg.pinv(A_HORIZ)

    print(f"A_VERT  cond#: {np.linalg.cond(A_VERT):.1f}")
    print(f"A_HORIZ cond#: {np.linalg.cond(A_HORIZ):.1f}")

    # ─────────────────────────────────────────────────────────────────────
    #  PID GAINS PER MODE
    # ─────────────────────────────────────────────────────────────────────

    # PID gains — EKF-feedback detuned for sensor noise.
    # Inner ~50% detuned: 20 Hz plant → 50 ms EKF delay → phase lag ω·0.05 rad,
    # keep crossover below ~6 rad/s for 45° margin.
    # Outer detuned: transponder at 5 Hz (vs GNSS 50 Hz), noisier position fix.
    _g = dict(
        att_Kp    = np.array([10.22,  9.90, 21.07]),
        att_Ki    = np.array([ 0.03,  0.03,  0.03]),
        att_Kd    = np.array([ 2.943, 7.06, 10.00]),
        att_i_lim = np.array([2.0,   2.0,   2.0]),
        att_lim   = 0.25,
        cyl_Kp    = np.array([0.224, 0.55, 0.436]),
        cyl_Ki    = np.array([0.002, 0.002, 0.006]),
        cyl_Kd    = np.array([0.60, 0.968, 1.382]),
        cyl_i_lim = np.array([3.0,  3.0,  8.0]),
    )

    att_Kp = _g["att_Kp"];  att_Ki = _g["att_Ki"];  att_Kd = _g["att_Kd"]
    att_i_lim = _g["att_i_lim"];  att_lim = _g["att_lim"]
    cyl_Kp = _g["cyl_Kp"];  cyl_Ki = _g["cyl_Ki"];  cyl_Kd = _g["cyl_Kd"]
    cyl_i_lim = _g["cyl_i_lim"]
    print(f"PID gains : lawnmower_ekf (UW)")

    # ─────────────────────────────────────────────────────────────────────
    #  TRAJECTORY BUILDERS
    # ─────────────────────────────────────────────────────────────────────

    def build_custom_traj(segments, t_arr):
        ref_p = np.zeros((3, len(t_arr)));  ref_v = np.zeros((3, len(t_arr)))
        if isinstance(segments, np.ndarray):
            for ax in range(3):
                ref_p[ax] = np.interp(t_arr, segments[:, 0], segments[:, 1+ax])
                ref_v[ax] = np.interp(t_arr, segments[:, 0], segments[:, 4+ax])
            return ref_p, ref_v
        for i, row in enumerate(segments):
            t0 = row[0];  x0, y0, z0 = row[1], row[2], row[3]
            vx, vy, vz = (row[4], row[5], row[6]) if len(row) == 7 else (0.0, 0.0, 0.0)
            t1 = segments[i+1][0] if i < len(segments)-1 else t_arr[-1]+1
            mask = (t_arr >= t0) & (t_arr < t1);  dts = t_arr[mask] - t0
            ref_p[0, mask] = x0+vx*dts;  ref_p[1, mask] = y0+vy*dts;  ref_p[2, mask] = z0+vz*dts
            ref_v[0, mask] = vx;          ref_v[1, mask] = vy;          ref_v[2, mask] = vz
        return ref_p, ref_v


    def build_test_time_water_traj(t_arr, wp=None):
        """Interpolate water trajectory onto sim time vector.
        Accepts pre-computed wp to avoid calling _water_timed_waypoints() twice."""
        if wp is None:
            wp, _, _ = _water_timed_waypoints()
        ref_p = np.zeros((3, len(t_arr)));  ref_v = np.zeros((3, len(t_arr)))
        for ax in range(3):
            ref_p[ax] = np.interp(t_arr, wp[:, 0], wp[:, 1+ax])
            ref_v[ax] = np.interp(t_arr, wp[:, 0], wp[:, 4+ax])
        r0, th0, z0 = wp[0, 1], wp[0, 2], wp[0, 3]
        return ref_p, ref_v, float(wp[-1, 0]), np.array([r0*np.cos(th0), r0*np.sin(th0), z0]), wp

    # ─────────────────────────────────────────────────────────────────────
    #  DISTURBANCE / CURRENT HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def get_disturbance(t_k):
        Fd, taud = np.zeros(3), np.zeros(3)
        if DIST_ENABLED:
            for row in DISTURBANCES:
                if row[0] <= t_k < row[1]:
                    Fd += np.array(row[2:5]);  taud += np.array(row[5:8])
        if IMPULSE_ENABLED:
            for row in IMPULSES:
                if t_k <= row[0] < t_k + dt:
                    Fd += np.array(row[1:4]) / dt;  taud += np.array(row[4:7]) / dt
        return Fd, taud


    def get_current(t_k):
        if not CURRENT_ENABLED:
            return np.zeros(3)
        tc = CURRENT_PROFILE[:, 0]
        if CURRENT_INTERP == 'cubic' and len(CURRENT_PROFILE) >= 4:
            from scipy.interpolate import interp1d
            f = interp1d(tc, CURRENT_PROFILE[:, 1:4], axis=0, kind='cubic', bounds_error=False,
                         fill_value=(CURRENT_PROFILE[0, 1:4], CURRENT_PROFILE[-1, 1:4]))
            return f(t_k)
        return np.array([np.interp(t_k, tc, CURRENT_PROFILE[:, c]) for c in [1, 2, 3]])

    # ─────────────────────────────────────────────────────────────────────
    #  SIMULATION SETUP
    # ─────────────────────────────────────────────────────────────────────

    ENABLE_EKF       = USE_EKF
    USE_EKF_FEEDBACK = USE_EKF

    dt = 0.005   # [s]

    if ENABLE_EKF:
        from sensors_UW import IMUSensor as _IMU_UW
        from sensors_UW import SensorSuite as _SensorSuite_UW
        from kalman_UW  import KinematicEKF12 as _EKF_UW
        _EKF_SUBSTEPS = max(1, round(dt * _IMU_UW.update_rate))
        _DT_EKF       = dt / _EKF_SUBSTEPS
        if _EKF_SUBSTEPS > 1:
            print(f"[EKF-UW] plant {1/dt:.0f} Hz < IMU {_IMU_UW.update_rate:.0f} Hz "
                  f"→ {_EKF_SUBSTEPS} EKF sub-steps per plant step (dt_ekf={_DT_EKF:.4f} s)")
    else:
        _EKF_SUBSTEPS = 1
        _DT_EKF       = dt

    _t_events = [0]   # T_BUFFER is added at the end only — no phantom pre-buffer
    if DIST_ENABLED and DISTURBANCES:
        _t_events.append(max(row[1] for row in DISTURBANCES))

    _wp_uw = None;  x0_override = None

    if TRAJ_MODE == "test_time_water":
        _wp_pre, _, _ = _water_timed_waypoints()
        _n_wp  = max(2, int(round(len(_wp_pre) * min(100, max(1, TRAJ_PCT)) / 100)))
        _wp_pre = _wp_pre[:_n_wp]
        _t_events.append(float(_wp_pre[-1, 0]))

    # When EKF feedback is active the inner/outer gains are detuned (~50% of nominal)
    # so the drone traverses the trajectory at roughly half the designed speed.
    # Add the full trajectory duration as extra time so the slower drone finishes.
    _traj_dur   = float(_wp_pre[-1, 0]) if (TRAJ_MODE == "test_time_water") else 0.0
    _ekf_extra  = _traj_dur if ENABLE_EKF else 0.0
    t_end = max(_t_events) + T_BUFFER + _ekf_extra
    t     = np.arange(0, t_end + dt, dt)
    N     = len(t)
    # Index in t[] where the trajectory itself ends (before T_BUFFER / EKF extra).
    # Event trigger is capped here so k_ref never advances into the held-constant buffer.
    _N_traj = min(N, int(round(_traj_dur / dt)) + 1) if (TRAJ_MODE == "test_time_water") else N

    if TRAJ_MODE == "hold":
        ref_pos = np.zeros((3, N));  ref_vel = np.zeros((3, N))
        print("Trajectory mode : HOLD at origin (underwater)")

    elif TRAJ_MODE == "custom":
        ref_pos, ref_vel = build_custom_traj(TRAJ_SEGMENTS, t)
        x0_override = np.array([TRAJ_SEGMENTS[0][1], TRAJ_SEGMENTS[0][2], TRAJ_SEGMENTS[0][3]])
        print(f"Trajectory mode : CUSTOM  ({len(TRAJ_SEGMENTS)} segments)")

    elif TRAJ_MODE == "test_time_water":
        ref_pos, ref_vel, _dur, _start, _wp_uw = build_test_time_water_traj(t, wp=_wp_pre)
        x0_override = _start
        print(f"Trajectory mode : TEST_TIME_WATER  (path duration={_dur:.0f} s, "
              f"t_end={t_end:.1f} s, TRAJ_PCT={TRAJ_PCT}%  [{_n_wp} waypoints])")
        print(f"  Start          : x={_start[0]:.2f} m  y={_start[1]:.2f} m  z={_start[2]:.2f} m")
    else:
        raise ValueError(f"Unknown TRAJ_MODE: '{TRAJ_MODE}'")

    _USE_CYL_REF = TRAJ_MODE in ("custom", "test_time_water")

    if _USE_CYL_REF:
        ref_yaw = np.arctan2(np.sin(ref_pos[1] + np.pi), np.cos(ref_pos[1] + np.pi))
    else:
        ref_yaw = np.zeros(N)

    if _USE_CYL_REF:
        r_r, th_r  = ref_pos[0], ref_pos[1]
        vr_r, vth_r, vz_r = ref_vel[0], ref_vel[1], ref_vel[2]
        ar_r  = np.gradient(vr_r,  dt);  ath_r = np.gradient(vth_r, dt);  az_r = np.gradient(vz_r, dt)
        ref_acc = np.zeros((3, N))
        ref_acc[0] = (ar_r - r_r*vth_r**2)*np.cos(th_r) - (r_r*ath_r + 2*vr_r*vth_r)*np.sin(th_r)
        ref_acc[1] = (ar_r - r_r*vth_r**2)*np.sin(th_r) + (r_r*ath_r + 2*vr_r*vth_r)*np.cos(th_r)
        ref_acc[2] = az_r
    else:
        ref_acc = np.gradient(ref_vel, dt, axis=1)

    if TRAJ_MODE == "test_time_water":
        from Test_time import WATER_ACCEL_MAX as _a_max_uw
        _mat_uw = np.column_stack([ref_pos[0], ref_pos[1], ref_pos[2],
                                   ref_vel[0], ref_vel[1], ref_vel[2],
                                   ref_acc[0], ref_acc[1], ref_acc[2],
                                   ref_yaw])
        _header_uw = [
            ('a_max_underwater',   _a_max_uw,       'm/s^2'),
            ('duration_underwater', float(t[-1]),    's'),
            ('n_waypoints_underwater', float(len(t)), 'rows'),
        ]
        _fname_uw = r'C:\Users\banda\Documents\MATLAB\UAUV Control\Aerial\trajectory_underwater.m'
        write_matlab_traj(_fname_uw, _mat_uw, _header_uw, 'underwater')
        print(f"MATLAB export   : {_fname_uw}  ({len(t)} rows)")

    print(f"Drag FF      : {'ON' if USE_DRAG_FF else 'OFF'}")
    print(f"Mass FF      : {'ON' if USE_MASS_FF else 'OFF'}  (added mass compensation)")
    print(f"Cent/Cor FF  : {'ON' if USE_CENT_FF else 'OFF'}")
    print(f"Event trig   : {'ON' if USE_EVENT_TRIG else 'OFF'}  "
          f"(r={EVENT_R_TOL} m  θ={EVENT_TH_TOL} rad  z={EVENT_Z_TOL} m)")
    print(f"Disturbances : {'ON' if DIST_ENABLED else 'OFF'}")
    print(f"EKF          : {'ON — feedback + plots' if USE_EKF else 'OFF'}")
    print(f"Plots        : {'ON' if ENABLE_PLOTS else 'OFF'}")

    # ─────────────────────────────────────────────────────────────────────
    #  PHYSICS FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────

    def rot_ZYX(phi, theta, psi):
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi), np.sin(psi)
        return np.array([
            [cy*ct, cy*st*sp - sy*cp, cy*st*cp + sy*sp],
            [sy*ct, sy*st*sp + cy*cp, sy*st*cp - cy*sp],
            [  -st, ct*sp,             ct*cp            ],
        ])


    def euler_kin(phi, theta):
        sp, cp = np.sin(phi), np.cos(phi)
        st, ct = np.sin(theta), np.cos(theta)
        return np.array([
            [1, sp*st/ct, cp*st/ct],
            [0, cp,      -sp      ],
            [0, sp/ct,    cp/ct   ],
        ])

    # Precompute kT/kQ sign-split arrays for vectorised thrust (avoids branching in ODE)
    _EZ = np.array([0.0, 0.0, 1.0])


    def uw_ode(s, wr_cmd, beta_cmd, Fd=None, taud=None, v_current=None):
        """
        20-State 6-DOF Underwater ODE (vectorised thruster loop).
        Direct translation of uw_dynamics.m (MATLAB plant).
        """
        if Fd is None:        Fd = np.zeros(3)
        if taud is None:      taud = np.zeros(3)
        if v_current is None: v_current = np.zeros(3)

        euler = s[3:6];  phi, theta, psi = euler
        vel   = s[6:9];  wb = s[9:12];  wr = s[12:16];  beta = s[16:20]

        R = rot_ZYX(phi, theta, psi)

        # Vectorised thrust: T_i = ±kT·ω_i²  (sign from ω direction)
        T = np.where(wr >= 0,  p.kT_fwd * wr**2, -p.kT_rev * wr**2)
        Q = np.where(wr >= 0,  p.kQ_fwd * wr**2, -p.kQ_rev * wr**2)

        # Thrust directions (3×4): v_i = cos(β)·U_base_i + sin(β)·ez
        V = np.cos(beta) * U_BASE + np.outer(_EZ, np.sin(beta))   # (3,4)

        # Body force and torque sums (vectorised)
        F_prop    = V @ T                                          # (3,)
        F_i_all   = V * T                                         # (3,4)
        tau_thrust = np.sum(np.cross(R_PROPS.T, F_i_all.T), axis=0)
        tau_rxn    = -V @ (H_DIR * Q)
        tau_prop   = tau_thrust + tau_rxn + taud

        # Hydrodynamics (drag relative to current, in body frame)
        v_body  = R.T @ (vel - v_current)
        F_drag  = -0.5 * p.rho_water * (p.Cd * p.A_face) * v_body * np.abs(v_body)

        # Gravity + buoyancy net (precomputed constant, body-frame at any attitude)
        F_gb_body = R.T @ _F_GB_VEC

        # Translational dynamics (diagonal added mass via precomputed _M_SIM)
        F_body_net = F_prop + F_drag + F_gb_body + R.T @ Fd
        pos_ddot   = R @ (F_body_net / _M_SIM)

        # Rotational dynamics (precomputed _I_MAT)
        wb_dot = np.linalg.solve(_I_MAT, tau_prop - np.cross(wb, _I_MAT @ wb))

        euler_dot = euler_kin(phi, theta) @ wb
        wr_dot    = (wr_cmd  - wr)   / p.tau_m
        beta_dot  = (beta_cmd - beta) / p.tau_srv

        return np.concatenate([vel, euler_dot, pos_ddot, wb_dot, wr_dot, beta_dot])


    def rk4_step(s, wr_cmd, beta_cmd, dt_, Fd=None, taud=None, v_current=None):
        kw = dict(Fd=Fd, taud=taud, v_current=v_current)
        k1 = uw_ode(s,             wr_cmd, beta_cmd, **kw)
        k2 = uw_ode(s + dt_/2*k1, wr_cmd, beta_cmd, **kw)
        k3 = uw_ode(s + dt_/2*k2, wr_cmd, beta_cmd, **kw)
        k4 = uw_ode(s + dt_*k3,   wr_cmd, beta_cmd, **kw)
        return s + (dt_/6) * (k1 + 2*k2 + 2*k3 + k4)

    # ─────────────────────────────────────────────────────────────────────
    #  CONTROL ALLOCATION  (fully vectorised)
    # ─────────────────────────────────────────────────────────────────────

    def allocate(F_des_body, tau_des):
        """
        Desired body-frame wrench → [wr_cmd(4), beta_cmd(4)].
          T_v = A_VERT_PINV  @ [Fz, τ_φ, τ_θ]  (vertical per thruster)
          T_h = A_HORIZ_PINV @ [Fx, Fy,  τ_ψ]  (horizontal per thruster)
          β_i = atan2(T_v, T_h); if β ∈ [0,π]: ω>0 else reflect β by π and ω<0.
        """
        T_v = A_VERT_PINV  @ np.array([F_des_body[2], tau_des[0], tau_des[1]])
        T_h = A_HORIZ_PINV @ np.array([F_des_body[0], F_des_body[1], tau_des[2]])

        beta_raw = np.arctan2(T_v, T_h)                   # (4,) in (-π, π]
        T_mag    = np.hypot(T_v, T_h)                     # (4,) ≥ 0
        in_range = (beta_raw >= p.BETA_MIN) & (beta_raw <= p.BETA_MAX)

        beta_cmd = np.where(in_range, beta_raw,
                            np.clip(beta_raw + np.pi, p.BETA_MIN, p.BETA_MAX))
        wr_fwd   = np.sqrt(np.maximum(T_mag / p.kT_fwd, 0.0))
        wr_rev   = -np.sqrt(np.maximum(T_mag / p.kT_rev, 0.0))
        wr_cmd   = np.where(T_mag > 1e-10, np.where(in_range, wr_fwd, wr_rev), 0.0)

        return np.clip(wr_cmd, -p.omega_max, p.omega_max), beta_cmd

    # ─────────────────────────────────────────────────────────────────────
    #  MAIN SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────

    X        = np.zeros((20, N))
    Wr_log   = np.zeros((4,  N))
    Beta_log = np.zeros((4,  N))
    U_log    = np.zeros((6,  N))   # virtual wrench [Fx,Fy,Fz, τx,τy,τz]
    Ref_log      = np.zeros((3, N))   # active cylindrical ref per step
    Ref_vel_log  = np.zeros((3, N))   # active cylindrical ref velocity per step
    X_ekf    = np.zeros((12, N)) if ENABLE_EKF else None

    if x0_override is not None:
        X[0:3, 0] = x0_override
    X[5, 0]     = ref_yaw[0]    # match yaw to avoid 180° transient
    X[12:16, 0] = _OMEGA_EQ     # hover prop speed (negative → reverse → down thrust)
    X[16:20, 0] = np.pi / 2.0   # servos pointing up

    # EKF initialisation (seeded from true initial state)
    _ekf_feedback_x = None
    if ENABLE_EKF:
        _suite_uw = _SensorSuite_UW(R_turbine=R_base)   # R_base already imported at top
        _ekf_uw   = _EKF_UW()
        _ekf_uw.x[0:3]  = X[0:3,  0]   # pos
        _ekf_uw.x[3:6]  = X[6:9,  0]   # vel  (EKF x[3:6] ← plant X[6:9])
        _ekf_uw.x[6:9]  = X[3:6,  0]   # euler (EKF x[6:9] ← plant X[3:6])
        _ekf_uw.x[9:12] = X[9:12, 0]   # body rates

    int_att = np.zeros(3);  int_pos = np.zeros(3);  k_ref = 0
    _t_complete = None   # sim time when drone first reaches last trajectory waypoint
    _k_end = N - 1      # last valid filled index (updated on early exit)

    # Cartesian position of the last waypoint — used for completion detection.
    # Using Cartesian (not cylindrical θ) avoids 2π aliasing when the lawnmower
    # completes a full revolution and the last strip coincides with θ≈0.
    if TRAJ_MODE == "test_time_water":
        _lw_x = _wp_pre[-1, 1] * np.cos(_wp_pre[-1, 2])
        _lw_y = _wp_pre[-1, 1] * np.sin(_wp_pre[-1, 2])
        _lw_z = _wp_pre[-1, 3]
        _COMPLETE_RADIUS = 1.5   # [m] 3-D distance threshold to declare completion
    else:
        _lw_x = _lw_y = _lw_z = 0.0;  _COMPLETE_RADIUS = 1.0

    _sim_iter = range(N - 1) if PLOT_MODE != "root_locus" else []
    if tqdm is not None and PLOT_MODE != "root_locus":
        _sim_iter = tqdm(_sim_iter, desc="Simulating UW", unit="step",
                         mininterval=5, dynamic_ncols=True)

    # Precompute force vector to cancel buoyancy (constant at level hover, rotated per step)
    _F_resid_inertial = np.array([0.0, 0.0, _F_RESID])
    _m_eff = _M_CTRL if USE_MASS_FF else np.full(3, p.m)

    for k in _sim_iter:
        s     = X[:, k]
        pos   = s[0:3];   euler = s[3:6];   phi, theta, psi = euler
        vel   = s[6:9];   wb    = s[9:12]

        # Controller input: EKF estimate (if feedback enabled) or true state
        if USE_EKF_FEEDBACK and _ekf_feedback_x is not None:
            c_pos   = _ekf_feedback_x[0:3]
            c_vel   = _ekf_feedback_x[3:6]
            c_euler = _ekf_feedback_x[6:9]
            c_wb    = _ekf_feedback_x[9:12]
        else:
            c_pos, c_vel, c_euler, c_wb = pos, vel, euler, wb
        c_phi, c_theta, c_psi = c_euler

        # Event-triggered reference advancement (uses EKF/true position estimate).
        # Capped at _N_traj-1 so k_ref never advances into the held-constant buffer region.
        if USE_EVENT_TRIG and _USE_CYL_REF:
            if k_ref < _N_traj - 1:
                _r_m  = np.sqrt(c_pos[0]**2 + c_pos[1]**2)
                _th_m = np.arctan2(c_pos[1], c_pos[0])
                _r_ok  = abs(_r_m - ref_pos[0, k_ref]) < EVENT_R_TOL
                _th_ok = abs(np.arctan2(np.sin(_th_m - ref_pos[1, k_ref]),
                                        np.cos(_th_m - ref_pos[1, k_ref]))) < EVENT_TH_TOL
                _z_ok  = abs(c_pos[2] - ref_pos[2, k_ref]) < EVENT_Z_TOL
                if _r_ok and _th_ok and _z_ok:
                    k_ref += 1
        # Completion: check TRUE (not EKF) Cartesian distance to last waypoint.
        # This is immune to 2π theta aliasing and k_ref racing.
        if _t_complete is None and TRAJ_MODE == "test_time_water":
            _d_last = np.sqrt((pos[0]-_lw_x)**2 + (pos[1]-_lw_y)**2 + (pos[2]-_lw_z)**2)
            if _d_last < _COMPLETE_RADIUS:
                _t_complete = t[k]
        kr = k_ref if (USE_EVENT_TRIG and _USE_CYL_REF) else k
        Ref_log[:, k]     = ref_pos[:, kr]
        Ref_vel_log[:, k] = ref_vel[:, kr]

        R_cur = rot_ZYX(c_phi, c_theta, c_psi)

        # ── Outer PID → inertial acceleration command ─────────────────────
        if _USE_CYL_REF:
            r_m  = max(np.sqrt(c_pos[0]**2 + c_pos[1]**2), 1e-6)
            th_m = np.arctan2(c_pos[1], c_pos[0])
            cs, sn = np.cos(th_m), np.sin(th_m)

            e_r  = ref_pos[0, kr] - r_m
            e_th = np.arctan2(np.sin(ref_pos[1, kr] - th_m), np.cos(ref_pos[1, kr] - th_m))
            e_z  = ref_pos[2, kr] - c_pos[2]
            e_t  = r_m * e_th

            vr_m   =  c_vel[0]*cs + c_vel[1]*sn
            vth_m  = (-c_vel[0]*sn + c_vel[1]*cs) / r_m
            e_vr   = ref_vel[0, kr] - vr_m
            e_vt   = r_m * (ref_vel[1, kr] - vth_m)
            e_vz   = ref_vel[2, kr] - c_vel[2]

            int_pos = np.clip(int_pos + np.array([e_r, e_t, e_z]) * dt, -cyl_i_lim, cyl_i_lim)
            a_r = cyl_Kp[0]*e_r + cyl_Ki[0]*int_pos[0] + cyl_Kd[0]*e_vr
            a_t = cyl_Kp[1]*e_t + cyl_Ki[1]*int_pos[1] + cyl_Kd[1]*e_vt
            a_z = cyl_Kp[2]*e_z + cyl_Ki[2]*int_pos[2] + cyl_Kd[2]*e_vz
            a_cmd = np.array([a_r*cs - a_t*sn, a_r*sn + a_t*cs, a_z])
        else:
            e_pos = ref_pos[:, kr] - c_pos;  e_vel = ref_vel[:, kr] - c_vel
            int_pos = np.clip(int_pos + e_pos * dt, -cyl_i_lim, cyl_i_lim)
            a_cmd = cyl_Kp*e_pos + cyl_Ki*int_pos + cyl_Kd*e_vel

        if USE_CENT_FF:
            a_cmd += ref_acc[:, kr]

        # ── Body-frame force command (cancel buoyancy, optional drag FF) ───
        F_body_des = _m_eff * (R_cur.T @ a_cmd) - R_cur.T @ _F_resid_inertial
        if USE_DRAG_FF:
            v_body_cur = R_cur.T @ c_vel
            F_body_des += 0.5 * p.rho_water * p.Cd * p.A_face * v_body_cur * np.abs(v_body_cur)

        # ── Inner PID → attitude torques ──────────────────────────────────
        e_att    = np.array([-c_phi, -c_theta, ref_yaw[kr] - c_psi])
        e_att[2] = np.arctan2(np.sin(e_att[2]), np.cos(e_att[2]))
        int_att  = np.clip(int_att + e_att * dt, -att_i_lim, att_i_lim)
        tau_cmd  = att_Kp * e_att + att_Ki * int_att - att_Kd * c_wb

        U_log[:, k] = np.concatenate([F_body_des, tau_cmd])

        # ── Allocation → integrate ────────────────────────────────────────
        wr_cmd_k, beta_cmd_k = allocate(F_body_des, tau_cmd)
        Wr_log[:, k]   = wr_cmd_k
        Beta_log[:, k] = beta_cmd_k

        Fd_k, taud_k = get_disturbance(t[k])
        X[:, k+1] = rk4_step(s, wr_cmd_k, beta_cmd_k, dt,
                              Fd=Fd_k, taud=taud_k, v_current=get_current(t[k]))

        # ── EKF update (uses true states for sensor models) ───────────────
        if ENABLE_EKF:
            _a_inertial = (X[6:9, k+1] - X[6:9, k]) / dt
            for _j in range(_EKF_SUBSTEPS):
                _t_sub = t[k] + _j * _DT_EKF
                _meas  = _suite_uw.tick(_t_sub, _DT_EKF, _a_inertial, euler, wb, pos, vel)
                _ekf_x = _ekf_uw.update(_meas, _DT_EKF)
            X_ekf[:, k]     = _ekf_x
            _ekf_feedback_x = _ekf_x

        if _t_complete is not None:
            _k_end = k + 1
            break

    _n_valid = _k_end + 1
    t        = t       [:_n_valid]
    X        = X       [:, :_n_valid]
    U_log    = U_log   [:, :_n_valid]
    Wr_log   = Wr_log  [:, :_n_valid]
    Beta_log = Beta_log[:, :_n_valid]
    Ref_log     = Ref_log    [:, :_n_valid]
    Ref_vel_log = Ref_vel_log[:, :_n_valid]
    ref_pos  = ref_pos [:, :_n_valid]
    ref_vel  = ref_vel [:, :_n_valid]
    ref_acc  = ref_acc [:, :_n_valid]
    ref_yaw  = ref_yaw [   :_n_valid]
    if X_ekf is not None:
        X_ekf = X_ekf[:, :_n_valid]
    N = _n_valid

    Wr_log[:, -1]   = Wr_log[:, -2]
    Beta_log[:, -1] = Beta_log[:, -2]
    U_log[:, -1]    = U_log[:, -2]
    if ENABLE_EKF and X_ekf is not None:
        X_ekf[:, -1] = X_ekf[:, -2]

    if SAVE_SIM_DATA:
        from pathlib import Path
        _npz_path = Path(__file__).parent / 'sim_data_UW.npz'
        _n_save = max(1, int(len(t) * min(100, max(1, SAVE_PCT)) / 100))
        np.savez(str(_npz_path),
                 t=t[:_n_save], X=X[:, :_n_save],
                 dt=np.float64(dt), R_base=np.float64(R_base))
        print(f"[SIM] State data saved → {_npz_path}  "
              f"({_n_save} steps at full dt={dt:.4f}s, {SAVE_PCT}% of {len(t)})")

    # ─────────────────────────────────────────────────────────────────────
    #  NUMERICAL LINEARISATION
    # ─────────────────────────────────────────────────────────────────────

    _s0 = np.zeros(20)
    _s0[2] = LIN_DEPTH;  _s0[5] = LIN_PSI;  _s0[6:9] = LIN_VEL
    _s0[12:16] = _OMEGA_EQ;  _s0[16:20] = np.pi / 2.0

    _u0  = np.concatenate([np.full(4, _OMEGA_EQ), np.full(4, np.pi/2.0)])
    _eps = 1e-5
    ns_uw, nu_uw = 20, 8

    A_lin_uw = np.zeros((ns_uw, ns_uw))
    B_lin_uw = np.zeros((ns_uw, nu_uw))

    def _ode_flat(s, u):
        return uw_ode(s, u[:4], u[4:])

    for i in range(ns_uw):
        sp, sm = _s0.copy(), _s0.copy();  sp[i] += _eps;  sm[i] -= _eps
        A_lin_uw[:, i] = (_ode_flat(sp, _u0) - _ode_flat(sm, _u0)) / (2*_eps)

    for j in range(nu_uw):
        up, um = _u0.copy(), _u0.copy();  up[j] += _eps;  um[j] -= _eps
        B_lin_uw[:, j] = (_ode_flat(_s0, up) - _ode_flat(_s0, um)) / (2*_eps)

    print(f"\nUW linearised system: {ns_uw} states, {nu_uw} inputs  "
          f"(depth={LIN_DEPTH} m, ω_eq={_OMEGA_EQ:.2f} rad/s)")
    print(f"Open-loop poles:\n{np.sort_complex(np.linalg.eigvals(A_lin_uw))}")


    def build_K_cl_uw(pKp, pKd, aKp, aKd, psi_lin=np.pi):
        """K (8×20): linearised state-feedback gain at hover."""
        T_eq   = _F_RESID / 4.0          # hover thrust magnitude per prop
        w_eq   = abs(_OMEGA_EQ)
        m_eff  = _M_CTRL if USE_MASS_FF else np.full(3, p.m)
        cp, sp = np.cos(psi_lin), np.sin(psi_lin)

        Kv = np.zeros((6, 20))
        # Position → force (body frame at psi_lin)
        Kv[0, 0]  = -m_eff[0]*(pKp[0]*cp + pKp[1]*sp);  Kv[0, 6]  = -m_eff[0]*(pKd[0]*cp + pKd[1]*sp)
        Kv[1, 1]  = -m_eff[1]*(pKp[0]*sp - pKp[1]*cp);  Kv[1, 7]  = -m_eff[1]*(pKd[0]*sp - pKd[1]*cp)
        Kv[2, 2]  = -m_eff[2]*pKp[2];                    Kv[2, 8]  = -m_eff[2]*pKd[2]
        # Attitude → torque
        Kv[3, 3]  = -aKp[0];  Kv[3, 9]  = -aKd[0]
        Kv[4, 4]  = -aKp[1];  Kv[4, 10] = -aKd[1]
        Kv[5, 5]  = -aKp[2];  Kv[5, 11] = -aKd[2]

        # Linearised allocation Jacobian (8×6) at hover
        s_wr   = w_eq / (2.0 * max(T_eq, 1e-6))    # d(ω)/d(T_v)
        s_beta = 1.0  / max(T_eq, 1e-6)            # d(β)/d(T_h)
        J_wr   = np.zeros((4, 6))
        J_wr[:, 2] = s_wr * A_VERT_PINV[:, 0]
        J_wr[:, 3] = s_wr * A_VERT_PINV[:, 1]
        J_wr[:, 4] = s_wr * A_VERT_PINV[:, 2]
        J_beta = np.zeros((4, 6))
        J_beta[:, 0] = s_beta * A_HORIZ_PINV[:, 0]
        J_beta[:, 1] = s_beta * A_HORIZ_PINV[:, 1]
        J_beta[:, 5] = s_beta * A_HORIZ_PINV[:, 2]

        return np.vstack([J_wr, J_beta]) @ Kv   # (8×20)


    # ref_pos_cart: what the controller actually commanded at each step (from Ref_log)
    if _USE_CYL_REF:
        _r, _th = Ref_log[0], Ref_log[1]
        ref_pos_cart = np.array([_r*np.cos(_th), _r*np.sin(_th), Ref_log[2]])
    else:
        ref_pos_cart = Ref_log

    # ─────────────────────────────────────────────────────────────────────
    #  ANALYSIS OUTPUT
    # ─────────────────────────────────────────────────────────────────────

    if PLOT_MODE == "sim":
        if _t_complete is not None:
            _insp_label = f"{_t_complete:.1f} s  ({_t_complete/60:.2f} min)"
        else:
            _insp_label = f"N/A  [trajectory not completed within {t_end:.0f} s]"
        print(f"\n{'═'*56}")
        print(f"  Total inspection time (UW): {_insp_label}")
        print(f"{'═'*56}")
        # error metric placeholder — to be defined

        if TRAJ_MODE == "test_time_water":
            # ── Scan quality metric ──────────────────────────────────────
            # Lateral deviation during vertical scan segments only.
            # Filter by time so points from other strips (same z-range) are excluded.
            # Waypoints: _wp_pre[:, 0]=t  [1]=r  [2]=theta  [3]=z
            _wp_x  = _wp_pre[:, 1] * np.cos(_wp_pre[:, 2])
            _wp_y_arr = _wp_pre[:, 1] * np.sin(_wp_pre[:, 2])
            _wp_z_arr = _wp_pre[:, 3]
            _wp_t_arr = _wp_pre[:, 0]
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

            def _rpm(w): return abs(w)*60.0/(2.0*np.pi)
            _wr     = X[12:16, :];  _wrc = Wr_log;  _bt = X[16:20, :]
            _wc_peak = float(np.max(np.abs(_wr)));  _bt_rng = float(np.max(_bt)) - float(np.min(_bt))
            _alpha_max = float(np.max(np.abs(_wrc[:, :-1] - _wr[:, :-1]))) / p.tau_m
            _T_peak = p.kT_fwd * _wc_peak**2
            W = 60
            print(f"\n{'═'*W}")
            print("  ACTUATOR CEILING  (motor / servo sizing)")
            print(f"{'═'*W}")
            print(f"  {'Hover prop speed':<34}  {_OMEGA_EQ:>+9.1f} rad/s  ({_rpm(_OMEGA_EQ):>5.0f} RPM)")
            print(f"  {'Saturation limit':<34}  +/- {p.omega_max:>5.1f} rad/s  ({_rpm(p.omega_max):>5.0f} RPM)")
            print(f"  {'Peak absolute speed':<34}  {_wc_peak:>9.1f} rad/s  ({_rpm(_wc_peak):>5.0f} RPM)")
            print(f"  {'Saturation utilisation':<34}  {100*_wc_peak/p.omega_max:>9.1f} %")
            print(f"  {'Max angular accel':<34}  {_alpha_max:>9.0f} rad/s²")
            print(f"  {'Peak thrust per prop':<34}  {_T_peak:>9.2f} N")
            print(f"  {'Peak total thrust (4 props)':<34}  {4*_T_peak:>9.2f} N")
            print(f"  {'Residual buoyancy':<34}  {_F_RESID:>9.2f} N  (= hover load)")
            print(f"  {'Servo angle range observed':<34}  {_bt_rng:>9.3f} rad  ({np.degrees(_bt_rng):.1f} deg)")
            print(f"{'═'*W}")

    # ─────────────────────────────────────────────────────────────────────
    #  PLOTS
    # ─────────────────────────────────────────────────────────────────────

    if PLOT_MODE not in ("sim", "root_locus"):
        print("Plots suppressed.")

    elif PLOT_MODE == "root_locus":
        from matplotlib.widgets import Slider

        _pKp = cyl_Kp.copy().astype(float);  _pKd = cyl_Kd.copy().astype(float)
        _aKp = att_Kp.copy().astype(float);  _aKd = att_Kd.copy().astype(float)
        _z3  = np.zeros(3)

        def _make_K(pKp, pKd, aKp, aKd):
            return build_K_cl_uw(pKp, pKd, aKp, aKd, psi_lin=LIN_PSI)

        def _poles_of(K):
            return np.linalg.eigvals(A_lin_uw + B_lin_uw @ K)

        fig_rl = plt.figure(figsize=(20, 13))
        fig_rl.suptitle(
            f"UW Sequential closed-loop pole analysis  —  gains: 'lawnmower_ekf'  "
            f"|  depth={LIN_DEPTH} m  |  Ki omitted",
            fontsize=12, fontweight='bold')

        ax_in  = fig_rl.add_axes([0.06, 0.42, 0.40, 0.50])
        ax_out = fig_rl.add_axes([0.55, 0.42, 0.40, 0.50])

        def _setup_ax(ax, title):
            ax.axvline(0, color='k', lw=0.9, ls='--');  ax.axhline(0, color='k', lw=0.9, ls='--')
            ax.set_xlabel("Real  [rad/s]", fontsize=10);  ax.set_ylabel("Imaginary  [rad/s]", fontsize=10)
            ax.set_title(title, fontsize=10, fontweight='bold');  ax.grid(True)

        _setup_ax(ax_in,  "INNER LOOP  (attitude + motor)\nPlant: open-loop UW  |  att_Kp, att_Kd only")
        _setup_ax(ax_out, "OUTER LOOP  (position)\nPlant: inner-closed  |  pos_Kp, pos_Kd  +  fixed att gains")

        ol_poles = np.linalg.eigvals(A_lin_uw)
        ax_in.scatter(ol_poles.real, ol_poles.imag, marker='x', s=80, color='red', lw=1.5,
                      label='Open-loop', zorder=7)

        K_in0  = _make_K(_z3, _z3, _aKp, _aKd)
        ip0    = _poles_of(K_in0)
        sc_in  = ax_in.scatter(ip0.real, ip0.imag, marker='x', s=130, lw=2.5,
                               color='darkorange', label='Att. closed-loop', zorder=5)
        ax_in.legend(loc='upper right', fontsize=8)

        A_in_cl0 = A_lin_uw + B_lin_uw @ K_in0
        icp0     = np.linalg.eigvals(A_in_cl0)
        sc_ref   = ax_out.scatter(icp0.real, icp0.imag, marker='o', s=50, color='lightgray',
                                  edgecolors='gray', lw=1, label='Inner-cl plant', zorder=3)
        K_full0  = _make_K(_pKp, _pKd, _aKp, _aKd)
        fp0      = _poles_of(K_full0)
        sc_out   = ax_out.scatter(fp0.real, fp0.imag, marker='x', s=130, lw=2.5,
                                  color='royalblue', label='Full closed-loop', zorder=5)
        ax_out.legend(loc='upper right', fontsize=8)

        def _set_lims(ax, *pole_sets):
            r  = np.concatenate([ps.real for ps in pole_sets])
            im = np.concatenate([ps.imag for ps in pole_sets])
            pr = max(abs(r).max()*0.15, 1.0);  pi = max(abs(im).max()*0.15, 1.0)
            ax.set_xlim(r.min()-pr, max(r.max()+pr, 0.5));  ax.set_ylim(im.min()-pi, im.max()+pi)

        _set_lims(ax_in, ol_poles, ip0)
        _slow  = lambda ps: ps[ps.real > -10.0]
        _sp0   = np.concatenate([_slow(icp0), _slow(fp0)])
        if len(_sp0):
            _pr = max(abs(_sp0.real).max()*0.2, 1.0);  _pi = max(abs(_sp0.imag).max()*0.2, 1.5)
            ax_out.set_xlim(_sp0.real.min()-_pr, 0.5)
            ax_out.set_ylim(-max(abs(_sp0.imag).max()+_pi, 1.5), max(abs(_sp0.imag).max()+_pi, 1.5))
        ax_out.text(0.02, 0.02, "Fast poles (Re < −10) off-screen",
                    transform=ax_out.transAxes, fontsize=7, color='gray')

        def _refresh_inner():
            # Single K build for inner loop, reused for both scatter updates
            K_in = _make_K(_z3, _z3, _aKp, _aKd)
            ip   = _poles_of(K_in)
            A_cl = A_lin_uw + B_lin_uw @ K_in
            icp  = np.linalg.eigvals(A_cl)
            K_full = _make_K(_pKp, _pKd, _aKp, _aKd)
            fp   = _poles_of(K_full)
            sc_in.set_offsets(np.c_[ip.real,  ip.imag])
            sc_ref.set_offsets(np.c_[icp.real, icp.imag])
            sc_out.set_offsets(np.c_[fp.real,  fp.imag])
            fig_rl.canvas.draw_idle()

        def _refresh_outer():
            K_full = _make_K(_pKp, _pKd, _aKp, _aKd)
            fp     = _poles_of(K_full)
            sc_out.set_offsets(np.c_[fp.real, fp.imag])
            fig_rl.canvas.draw_idle()

        SL_H = 0.060;  SL_W = 0.115;  SL_GAP = 0.020
        y_kd = 0.05;   y_kp = y_kd + SL_H + 0.05
        att_lbls = ['phi', 'theta', 'psi'];  pos_lbls = ['r', 't', 'z']
        att_cols = ['#FF8C00', '#FFD700', '#FF6347'];  pos_cols = ['#4169E1', '#1E90FF', '#00BFFF']
        all_sl_refs = [];  sl_data = []

        def _add_sliders(x0, Kp_arr, Kd_arr, lbls, cols, prefix):
            for ci in range(3):
                xc = x0 + ci*(SL_W+SL_GAP)
                for gname, garr, yr in [('Kp', Kp_arr, y_kp), ('Kd', Kd_arr, y_kd)]:
                    ax_sl = fig_rl.add_axes([xc, yr, SL_W, SL_H])
                    v0 = float(garr[ci]);  vmax = max(v0*1.5, 2.0)
                    sl = Slider(ax_sl, lbls[ci], 0.0, vmax, valinit=v0, valstep=vmax/500, color=cols[ci])
                    sl.label.set_fontsize(11);  sl.label.set_position((0.03, 0.5));  sl.label.set_horizontalalignment('left')
                    sl.valtext.set_fontsize(8);  sl.valtext.set_position((0.97, 0.5));  sl.valtext.set_horizontalalignment('right')
                    all_sl_refs.append(sl);  sl_data.append((sl, gname, ci, prefix))

        _add_sliders(0.06, _aKp, _aKd, att_lbls, att_cols, 'att')
        _add_sliders(0.55, _pKp, _pKd, pos_lbls, pos_cols, 'pos')
        for yr, lbl in [(y_kp,'Kp'), (y_kd,'Kd')]:
            fig_rl.text(0.005, yr+SL_H/2, lbl, fontsize=12, fontweight='bold', va='center', color='dimgray')
            fig_rl.text(0.505, yr+SL_H/2, lbl, fontsize=12, fontweight='bold', va='center', color='dimgray')
        fig_rl.text(0.06+1*(SL_W+SL_GAP), y_kp+SL_H+0.010,
                    'INNER  —  att_Kp / att_Kd  (phi, theta, psi)', fontsize=9, fontweight='bold', ha='center', color='dimgray')
        fig_rl.text(0.55+1*(SL_W+SL_GAP), y_kp+SL_H+0.010,
                    'OUTER  —  cyl_Kp / cyl_Kd  (r, t, z)', fontsize=9, fontweight='bold', ha='center', color='dimgray')

        def _make_cb(gname, ci, prefix):
            def cb(val):
                if prefix == 'att':
                    if gname == 'Kp': _aKp[ci] = val
                    else:             _aKd[ci] = val
                    _refresh_inner()
                else:
                    if gname == 'Kp': _pKp[ci] = val
                    else:             _pKd[ci] = val
                    _refresh_outer()
            return cb

        for sl, gname, ci, prefix in sl_data:
            sl.on_changed(_make_cb(gname, ci, prefix))
        if show_plots:
            plt.show()

    else:  # PLOT_MODE == "sim"
        _ps  = max(1, N // 10_000)
        def _zc(a): return np.where(np.abs(a) < 1e-10, 0.0, a)
        t_p  = t[::_ps];   X_p  = _zc(X[:, ::_ps])
        rp_p = _zc(ref_pos_cart[:, ::_ps])
        ry_p = _zc(ref_yaw[::_ps])
        Ul_p = _zc(U_log[:, ::_ps]);  Wr_p = _zc(Wr_log[:, ::_ps]);  Bt_p = _zc(Beta_log[:, ::_ps])

        if _USE_CYL_REF:
            _r_p   = np.maximum(np.sqrt(X_p[0]**2 + X_p[1]**2), 1e-6)
            _th_p  = np.unwrap(np.arctan2(X_p[1], X_p[0]))
            _z_p   = X_p[2]
            _vr_p  =  X_p[6]*np.cos(_th_p) + X_p[7]*np.sin(_th_p)
            _vth_p = (-X_p[6]*np.sin(_th_p) + X_p[7]*np.cos(_th_p)) / _r_p
            _vz_p  = X_p[8]
            _rr_p  = Ref_log[0, ::_ps];     _thr_p = Ref_log[1, ::_ps];     _zr_p = Ref_log[2, ::_ps]
            _vrr   = Ref_vel_log[0, ::_ps]; _vthr  = Ref_vel_log[1, ::_ps]; _vzr  = Ref_vel_log[2, ::_ps]

        gc = (0.85, 0.95, 0.85)
        def shade_gusts(ax):
            if not DIST_ENABLED: return
            yl = ax.get_ylim()
            for row in DISTURBANCES: ax.axvspan(row[0], row[1], color=gc, alpha=0.5, zorder=0)
            ax.set_ylim(yl)

        # fig 1: position + attitude
        fig1, axes1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig1.suptitle(f"UW Position & Attitude  [{TRAJ_MODE}]", fontsize=13)
        pos_lbl = ['r  [m]', 'theta  [rad]', 'z  [m]'] if _USE_CYL_REF else ['x  [m]', 'y  [m]', 'z  [m]']
        pos_act = [_r_p, _th_p, _z_p]   if _USE_CYL_REF else [X_p[0], X_p[1], X_p[2]]
        pos_ref = [_rr_p, _thr_p, _zr_p] if _USE_CYL_REF else [rp_p[0], rp_p[1], rp_p[2]]
        att_lbl = ['phi  [deg]', 'theta  [deg]', 'psi  [deg]']
        for i in range(3):
            ax = axes1[i, 0]
            if PLOT_ACTUAL:    ax.plot(t_p, pos_act[i], 'b',   lw=1.6, label='Actual')
            if PLOT_REFERENCE: ax.plot(t_p, pos_ref[i], 'r--', lw=1.2, label='Reference')
            ax.set_ylabel(pos_lbl[i]); ax.grid(True); shade_gusts(ax)
            if i == 0: ax.set_title("Position (cylindrical)" if _USE_CYL_REF else "Position")
            ax.legend(loc='lower right')
            ax = axes1[i, 1]
            ax.plot(t_p, np.degrees(X_p[3+i]), 'b', lw=1.6)
            if i == 2 and PLOT_REFERENCE:
                ax.plot(t_p, np.degrees(np.unwrap(ry_p)), 'r--', lw=1.2, label='Ref yaw')
                ax.legend(loc='lower right')
            ax.set_ylabel(att_lbl[i]); ax.grid(True); shade_gusts(ax)
            if i == 0: ax.set_title("Attitude")
        axes1[2, 0].set_xlabel("Time  [s]");  axes1[2, 1].set_xlabel("Time  [s]")
        for ax in fig1.axes: ax.tick_params(labelbottom=True)
        fig1.tight_layout()

        # fig 2: velocity tracking
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig2.suptitle(f"UW Velocity Tracking  [{TRAJ_MODE}]", fontsize=13)
        vel_lbl = ['vr  [m/s]', 'vtheta  [rad/s]', 'vz  [m/s]'] if _USE_CYL_REF else ['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]']
        vel_act = [_vr_p, _vth_p, _vz_p]  if _USE_CYL_REF else [X_p[6], X_p[7], X_p[8]]
        vel_ref = [_vrr,  _vthr,  _vzr ]  if _USE_CYL_REF else [rv_p[0], rv_p[1], rv_p[2]]
        for i in range(3):
            ax = axes2[i]
            if PLOT_ACTUAL:    ax.plot(t_p, vel_act[i], 'b',   lw=1.6, label='Actual')
            if PLOT_REFERENCE: ax.plot(t_p, vel_ref[i], 'r--', lw=1.2, label='Reference')
            ax.set_ylabel(vel_lbl[i]); ax.grid(True); shade_gusts(ax)
            if i == 0: ax.legend(loc='lower right')
        axes2[-1].set_xlabel("Time  [s]")
        for ax in fig2.axes: ax.tick_params(labelbottom=True)
        fig2.tight_layout()

        # error figure placeholder — to be defined

        # fig 3: virtual wrench
        fig3, axes3 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig3.suptitle("UW Virtual Wrench", fontsize=13)
        f_lbl = ['Fx  [N]', 'Fy  [N]', 'Fz  [N]']
        t_lbl = ['tau_phi  [N·m]', 'tau_theta  [N·m]', 'tau_psi  [N·m]']
        for i in range(3):
            ax = axes3[i, 0];  ax.plot(t_p, Ul_p[i], 'teal', lw=1.6)
            ax.axhline(0, color='k', ls=':', lw=0.8);  ax.set_ylabel(f_lbl[i]);  ax.grid(True);  shade_gusts(ax)
            if i == 0: ax.set_title("Body-frame forces")
            ax = axes3[i, 1];  ax.plot(t_p, Ul_p[3+i], 'darkorange', lw=1.6)
            ax.axhline(0, color='k', ls=':', lw=0.8);  ax.set_ylabel(t_lbl[i]);  ax.grid(True);  shade_gusts(ax)
            if i == 0: ax.set_title("Body-frame torques")
        axes3[2, 0].set_xlabel("Time  [s]");  axes3[2, 1].set_xlabel("Time  [s]")
        for ax in fig3.axes: ax.tick_params(labelbottom=True)
        fig3.tight_layout()

        # fig 4: prop speeds + servo angles
        fig4, axes4 = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
        fig4.suptitle("UW Prop Speeds & Servo Angles", fontsize=13)
        prop_names = ['FL (0)', 'FR (1)', 'RL (2)', 'RR (3)']
        for i in range(4):
            ax = axes4[i, 0]
            ax.plot(t_p, X_p[12+i], 'b', lw=1.6, label='Actual')
            ax.plot(t_p, Wr_p[i],   'r--', lw=1.0, label='Commanded')
            ax.axhline(_OMEGA_EQ, color='gray', ls=':', lw=1.0, label='ω_eq')
            ax.set_ylabel(f"ω_{prop_names[i]}  [rad/s]");  ax.grid(True);  shade_gusts(ax)
            if i == 0: ax.set_title("Prop speeds");  ax.legend(loc='lower right')
            ax = axes4[i, 1]
            ax.plot(t_p, np.degrees(X_p[16+i]), 'b', lw=1.6, label='Actual')
            ax.plot(t_p, np.degrees(Bt_p[i]),   'r--', lw=1.0, label='Commanded')
            ax.axhline(90, color='gray', ls=':', lw=1.0, label='90° (vertical)')
            ax.set_ylabel(f"β_{prop_names[i]}  [deg]");  ax.grid(True);  shade_gusts(ax)
            if i == 0: ax.set_title("Servo angles");  ax.legend(loc='lower right')
        axes4[3, 0].set_xlabel("Time  [s]");  axes4[3, 1].set_xlabel("Time  [s]")
        for ax in fig4.axes: ax.tick_params(labelbottom=True)
        fig4.tight_layout()

        # fig 4b: signed thrust per thruster
        # T_i = +kT_fwd·ω² when ω≥0 (forward),  −kT_rev·ω² when ω<0 (reverse)
        _wr_p = X_p[12:16, :]   # (4, N_plot) actual prop speeds
        _T_p  = np.where(_wr_p >= 0, p.kT_fwd * _wr_p**2, -p.kT_rev * _wr_p**2)
        _T_eq = -p.kT_rev * _OMEGA_EQ**2   # hover equilibrium thrust (negative = downward)
        fig4b, axes4b = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        fig4b.suptitle("UW Thruster Thrust  (+ = up, − = down)", fontsize=13)
        for i in range(4):
            ax = axes4b[i]
            ax.plot(t_p, _T_p[i], 'teal', lw=1.6)
            ax.axhline(_T_eq, color='r', ls='--', lw=1.0, label=f'Eq T = {_T_eq:.2f} N')
            ax.axhline(0, color='k', ls=':', lw=0.8)
            ax.set_ylabel(f"{prop_names[i]}  [N]"); ax.grid(True); shade_gusts(ax)
            if i == 0: ax.legend(loc='lower right')
        axes4b[-1].set_xlabel("Time  [s]")
        for ax in fig4b.axes: ax.tick_params(labelbottom=True)
        fig4b.tight_layout()

        if PLOT_VIBRATION:
            _wr_vib  = X_p[12:16, :]                      # (4, N) rotor speeds [rad/s]
            _f1      = np.abs(_wr_vib) / (2 * np.pi)      # 1Ω  [Hz]  per thruster
            _f_bp    = N_BLADES * _f1                      # blade-pass [Hz] per thruster

            _f1_mean  = _f1.mean(axis=0)
            _f1_lo    = _f1.min(axis=0)
            _f1_hi    = _f1.max(axis=0)
            _fbp_mean = _f_bp.mean(axis=0)
            _fbp_lo   = _f_bp.min(axis=0)
            _fbp_hi   = _f_bp.max(axis=0)

            fig_vib, ax_vib = plt.subplots(figsize=(11, 4))
            fig_vib.suptitle("UW Vibration Excitation Frequency Envelope  (1Ω  &  Blade-Pass)", fontsize=13)

            ax_vib.fill_between(t_p, _f1_lo,  _f1_hi,  alpha=0.25, color='teal')
            ax_vib.fill_between(t_p, _fbp_lo, _fbp_hi, alpha=0.25, color='darkorange')
            ax_vib.plot(t_p, _f1_mean,  color='teal',       lw=1.5, label='1Ω  (thruster, mean)')
            ax_vib.plot(t_p, _fbp_mean, color='darkorange',  lw=1.5, label=f'{N_BLADES}Ω  (blade-pass, mean)')

            ax_vib.set_xlabel("Time  [s]")
            ax_vib.set_ylabel("Frequency  [Hz]")
            ax_vib.legend(loc='upper right')
            ax_vib.grid(True)
            shade_gusts(ax_vib)
            fig_vib.tight_layout()

        # fig 5: open-loop poles
        fig5, ax5 = plt.subplots(figsize=(7, 6))
        _poles_ol = np.linalg.eigvals(A_lin_uw)
        ax5.scatter(_poles_ol.real, _poles_ol.imag, marker='x', s=80, color='teal', zorder=5)
        ax5.axvline(0, color='k', lw=0.8, ls='--');  ax5.axhline(0, color='k', lw=0.8, ls='--')
        ax5.set_xlabel("Real");  ax5.set_ylabel("Imaginary")
        ax5.set_title(f"UW Open-Loop Poles  (depth={LIN_DEPTH} m)");  ax5.grid(True)
        fig5.tight_layout()

        # fig 6: 3D flight path
        fig6 = plt.figure(figsize=(11, 10));  ax6 = fig6.add_subplot(111, projection='3d')
        _th_s = np.linspace(0, 2*np.pi, 60);  _z_uw = np.linspace(-H_water, 0, 40)
        _TH, _ZZ = np.meshgrid(_th_s, _z_uw)
        ax6.plot_surface(R_base*np.cos(_TH), R_base*np.sin(_TH), _ZZ,
                         color='silver', alpha=0.30, edgecolor='none')
        _xys = R_base*2.5
        _xs, _ys = np.meshgrid(np.linspace(-_xys, _xys, 2), np.linspace(-_xys, _xys, 2))
        ax6.plot_surface(_xs, _ys, np.zeros_like(_xs), color='dodgerblue', alpha=0.15)
        if PLOT_REFERENCE and _wp_uw is not None:
            _rx = _wp_uw[:, 1]*np.cos(_wp_uw[:, 2]);  _ry = _wp_uw[:, 1]*np.sin(_wp_uw[:, 2])
            ax6.plot(_rx, _ry, _wp_uw[:, 3], 'r--', lw=1.8, label='Reference', zorder=5)
        if PLOT_ACTUAL:
            ax6.plot(X_p[0], X_p[1], X_p[2], color='teal', lw=1.8, label='Actual', zorder=6)
            ax6.scatter(*X_p[:3,  0], color='green',  s=40, zorder=7)
            ax6.scatter(*X_p[:3, -1], color='orange', s=40, zorder=7)
            ax6.text(X_p[0,  0], X_p[1,  0], X_p[2,  0], '  start', fontsize=7, color='green')
            ax6.text(X_p[0, -1], X_p[1, -1], X_p[2, -1], '  end',   fontsize=7, color='orange')
        ax6.set_xlabel("X  [m]");  ax6.set_ylabel("Y  [m]");  ax6.set_zlabel("Z  [m]  (neg=depth)")
        ax6.set_title(f"UW 3D Flight Path  [{TRAJ_MODE}]", fontsize=13);  ax6.legend(loc='upper left')
        ax6.view_init(elev=20, azim=45)    # RHR z-up: elev from xy-plane, azim CCW from +x
        fig6.tight_layout()

        # ── EKF vs truth plots (only when USE_EKF=True) ──────────────────
        if ENABLE_EKF and X_ekf is not None:
            _ek = X_ekf[:, ::_ps]

            fig_e1, axes_e1 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e1.suptitle("EKF-UW: Position  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['x  [m]', 'y  [m]', 'z  [m]']):
                axes_e1[i].plot(t_p, X_p[i],      color='steelblue', lw=1.5, label='Truth')
                axes_e1[i].plot(t_p, _ek[i],      color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e1[i].set_ylabel(lbl);  axes_e1[i].grid(True)
                if i == 0: axes_e1[i].legend(loc='upper right')
            axes_e1[-1].set_xlabel("Time  [s]")
            for ax in fig_e1.axes: ax.tick_params(labelbottom=True)
            fig_e1.tight_layout()

            fig_e2, axes_e2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e2.suptitle("EKF-UW: Velocity  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]']):
                axes_e2[i].plot(t_p, X_p[6+i],    color='steelblue', lw=1.5, label='Truth')
                axes_e2[i].plot(t_p, _ek[3+i],    color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e2[i].set_ylabel(lbl);  axes_e2[i].grid(True)
                if i == 0: axes_e2[i].legend(loc='upper right')
            axes_e2[-1].set_xlabel("Time  [s]")
            for ax in fig_e2.axes: ax.tick_params(labelbottom=True)
            fig_e2.tight_layout()

            fig_e3, axes_e3 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e3.suptitle("EKF-UW: Euler Angles  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['φ  [deg]', 'θ  [deg]', 'ψ  [deg]']):
                _tr = np.degrees(np.arctan2(np.sin(X_p[3+i]), np.cos(X_p[3+i])))
                axes_e3[i].plot(t_p, _tr,                  color='steelblue', lw=1.5, label='Truth')
                axes_e3[i].plot(t_p, np.degrees(_ek[6+i]), color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e3[i].set_ylabel(lbl);  axes_e3[i].grid(True)
                if i == 0: axes_e3[i].legend(loc='upper right')
            axes_e3[-1].set_xlabel("Time  [s]")
            for ax in fig_e3.axes: ax.tick_params(labelbottom=True)
            fig_e3.tight_layout()

            fig_e4, axes_e4 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            fig_e4.suptitle("EKF-UW: Body Rates  —  Truth vs Estimate", fontsize=13)
            for i, lbl in enumerate(['p  [rad/s]', 'q  [rad/s]', 'r  [rad/s]']):
                axes_e4[i].plot(t_p, X_p[9+i],    color='steelblue', lw=1.5, label='Truth')
                axes_e4[i].plot(t_p, _ek[9+i],    color='tomato',    lw=1.2, ls='--', label='EKF')
                axes_e4[i].set_ylabel(lbl);  axes_e4[i].grid(True)
                if i == 0: axes_e4[i].legend(loc='upper right')
            axes_e4[-1].set_xlabel("Time  [s]")
            for ax in fig_e4.axes: ax.tick_params(labelbottom=True)
            fig_e4.tight_layout()

        if show_plots:
            plt.show()

    if not show_plots:
        plt.close('all')

    _crashed       = bool(X[2, _k_end] > 0.0)
    _pos_err       = np.linalg.norm(X[:3, :_n_valid] - ref_pos_cart[:3, :_n_valid], axis=0)
    _depth_err     = np.abs(X[2, :_n_valid] - ref_pos_cart[2, :_n_valid])
    _euler_abs_deg = np.abs(np.degrees(X[3:6, :_n_valid]))
    _yaw_rate_dps  = np.abs(np.degrees(X[11, :_n_valid]))
    _result = {
        'rms_pos_error':    float(np.sqrt(np.mean(_pos_err**2))),
        'max_pos_error':    float(np.max(_pos_err)),
        'crashed':          _crashed,
        't_complete':       float(_t_complete) if _t_complete is not None else None,
        'max_depth_error':  float(np.max(_depth_err)),
        'rms_depth_error':  float(np.sqrt(np.mean(_depth_err**2))),
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
