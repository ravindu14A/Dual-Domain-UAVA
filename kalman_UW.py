"""
kalman_UW.py  --  12-State Kinematic EKF for the underwater phase (quadcopterUW)
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

Direct mirror of kalman_SC.py — identical architecture and maths.
Only substitutions:
  GNSSSensor  → TransponderSensor  (same R matrix structure, 5 Hz)
  BaroSensor  → DepthSensor        (same R matrix structure, same sigma)
  correct_gnss  → correct_transponder
  correct_baro  → correct_depth

State vector (12):
  x = [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
       pos(3)     vel(3)        euler(3)            body_rates(3)

Prediction  : strapdown at 200 Hz (10 EKF sub-steps per 0.05 s plant step)
Corrections : gyro (200 Hz), AHRS (100 Hz), transponder pos+vel (5 Hz), depth (50 Hz)

Convention: ENU inertial frame, FLU body, ZYX Euler.
"""

import numpy as np
from sensors_UW import IMUSensor, DepthSensor, TransponderSensor, AHRSSensor


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _rot(phi, theta, psi):
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [  -st,          ct*sp,             ct*cp    ],
    ])


class KinematicEKF12:
    """
    12-state kinematic EKF — identical to kalman_SC.KinematicEKF12.

    Usage:
        ekf  = KinematicEKF12()
        meas = suite.tick(t, dt, ...)   # SensorSuite from sensors_UW.py
        est  = ekf.update(meas)         # returns [pos, vel, euler, rates]
    """

    _G = 9.81
    _accel_k   = 1.0
    _THETA_LIM = np.radians(80)

    def __init__(self):
        self.x = np.zeros(12)
        self.P = np.eye(12) * 10.0
        self._build_R()

    # ── R matrices ────────────────────────────────────────────────────────────

    def _build_R(self):
        self.R_gyro        = np.diag(np.full(3, IMUSensor.sigma_gyro**2))
        self.R_ahrs        = np.diag([
            AHRSSensor.sigma_phi**2,
            AHRSSensor.sigma_theta**2,
            AHRSSensor.sigma_psi**2,
        ])
        self.R_transponder = np.diag([
            *([TransponderSensor.sigma_pos**2] * 3),
            *([TransponderSensor.sigma_vel**2] * 3),
        ])
        self.R_depth = DepthSensor.sigma_depth**2

    # ── Process noise ─────────────────────────────────────────────────────────

    def _Q(self, dt):
        q = np.concatenate([
            np.full(3, 1e-4),
            np.full(3, IMUSensor.sigma_accel),
            np.full(3, IMUSensor.sigma_gyro),
            np.full(3, IMUSensor.sigma_gyro),
        ])
        return np.diag(q**2) * dt

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, accel, gyro, dt=None):
        if dt is None:
            dt = 1.0 / IMUSensor.update_rate

        x   = self.x.copy()
        phi, theta, psi = x[6], x[7], x[8]
        p, q, r         = x[9], x[10], x[11]

        theta = np.clip(theta, -self._THETA_LIM, self._THETA_LIM)
        x[7]  = theta

        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        tan_t  = st / ct

        R = _rot(phi, theta, psi)
        ax, ay, az = accel

        v_dot     = R @ accel + np.array([0.0, 0.0, -self._G])
        phi_dot   = p + (q*sp + r*cp) * tan_t
        theta_dot = q*cp - r*sp
        psi_dot   = (q*sp + r*cp) / ct

        x[0:3] += x[3:6] * dt
        x[3:6] += v_dot * dt
        x[6]    = _wrap(phi   + phi_dot   * dt)
        x[7]    = _wrap(theta + theta_dot * dt)
        x[8]    = _wrap(psi   + psi_dot   * dt)

        cy, sy = np.cos(psi), np.sin(psi)
        dvx_dphi = ay*(sy*sp + cy*cp*st) + az*(sy*cp - cy*sp*st)
        dvx_dth  = ax*(-cy*st) + ay*(cy*sp*ct) + az*(cy*cp*ct)
        dvx_dpsi = ax*(-sy*ct) + ay*(-sy*st*sp - cy*cp) + az*(-sy*st*cp + cy*sp)
        dvy_dphi = ay*(-cy*sp + sy*cp*st) + az*(-cy*cp - sy*sp*st)
        dvy_dth  = ax*(-sy*st) + ay*(sy*sp*ct) + az*(sy*cp*ct)
        dvy_dpsi = ax*(cy*ct)  + ay*(cy*st*sp - sy*cp) + az*(cy*st*cp + sy*sp)
        dvz_dphi = ay*(ct*cp) + az*(-ct*sp)
        dvz_dth  = ax*(-ct)   + ay*(-st*sp) + az*(-st*cp)

        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = np.array([
            [dvx_dphi, dvx_dth, dvx_dpsi],
            [dvy_dphi, dvy_dth, dvy_dpsi],
            [dvz_dphi, dvz_dth, 0.0     ],
        ]) * dt
        F[6,  9:12] = np.array([1.0, sp*tan_t, cp*tan_t]) * dt
        F[7,  9:12] = np.array([0.0, cp,       -sp      ]) * dt
        F[8,  9:12] = np.array([0.0, sp/ct,    cp/ct    ]) * dt

        self.x = x
        self.P = F @ self.P @ F.T + self._Q(dt)

    # ── Generic update ────────────────────────────────────────────────────────

    def _update(self, H, R_mat, innov):
        S = H @ self.P @ H.T + R_mat
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(12) - K @ H) @ self.P
        self.x[6] = _wrap(self.x[6])
        self.x[7] = np.clip(_wrap(self.x[7]), -self._THETA_LIM, self._THETA_LIM)
        self.x[8] = _wrap(self.x[8])

    # ── Corrections ───────────────────────────────────────────────────────────

    def correct_gyro(self, gyro):
        H = np.zeros((3, 12));  H[:, 9:12] = np.eye(3)
        self._update(H, self.R_gyro, gyro - H @ self.x)

    def correct_ahrs(self, ahrs_euler):
        H = np.zeros((3, 12));  H[:, 6:9] = np.eye(3)
        innov = np.array([
            _wrap(ahrs_euler[0] - self.x[6]),
            _wrap(ahrs_euler[1] - self.x[7]),
            _wrap(ahrs_euler[2] - self.x[8]),
        ])
        self._update(H, self.R_ahrs, innov)

    def correct_transponder(self, trans_pos, trans_vel):
        """Mirrors correct_gnss exactly — absolute pos+vel fix."""
        H = np.zeros((6, 12));  H[0:6, 0:6] = np.eye(6)
        z = np.concatenate([trans_pos, trans_vel])
        self._update(H, self.R_transponder, z - H @ self.x)

    def correct_depth(self, depth_z):
        """Mirrors correct_baro exactly — absolute z fix."""
        H = np.zeros((1, 12));  H[0, 2] = 1.0
        R = np.array([[self.R_depth]])
        self._update(H, R, np.array([depth_z - self.x[2]]))

    # ── Main entry point ──────────────────────────────────────────────────────

    def update(self, sensor_data, dt=None):
        imu         = sensor_data.get('imu')
        depth       = sensor_data.get('depth')
        transponder = sensor_data.get('transponder')
        ahrs        = sensor_data.get('ahrs')

        if imu is not None:
            accel, gyro, mag = imu
            self.predict(accel, gyro, dt)
            self.correct_gyro(gyro)

        if ahrs is not None:
            self.correct_ahrs(ahrs)

        if transponder is not None:
            self.correct_transponder(*transponder)

        if depth is not None:
            self.correct_depth(depth)

        return self.x[:12].copy()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def pos(self):    return self.x[0:3].copy()
    @property
    def vel(self):    return self.x[3:6].copy()
    @property
    def euler(self):  return self.x[6:9].copy()
    @property
    def rates(self):  return self.x[9:12].copy()


# ── Standalone runner (needs sim_data_UW.npz from quadcopterUW.py) ───────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sensors_UW import SensorSuite, IMUSensor

    _npz = Path(__file__).parent / 'sim_data_UW.npz'
    if not _npz.exists():
        raise FileNotFoundError(
            f"{_npz} not found — run quadcopterUW.py with SAVE_SIM_DATA=True first.")

    data   = np.load(str(_npz))
    t      = data['t']
    X      = data['X']
    dt     = float(data['dt'])
    R_base = float(data['R_base'])
    N      = len(t)
    print(f"[EKF-UW] Loaded {N} steps  dt={dt}s  T={t[-1]:.1f}s  R_base={R_base}m")

    _EKF_SUBSTEPS = max(1, round(dt * IMUSensor.update_rate))
    _DT_EKF       = dt / _EKF_SUBSTEPS
    if _EKF_SUBSTEPS > 1:
        print(f"[EKF-UW] {_EKF_SUBSTEPS} sub-steps/plant step  dt_ekf={_DT_EKF:.4f}s")

    suite = SensorSuite(R_turbine=R_base)
    ekf   = KinematicEKF12()
    ekf.x[0:3]  = X[0:3,  0]
    ekf.x[3:6]  = X[6:9,  0]   # vel  (plant X[6:9], EKF x[3:6])
    ekf.x[6:9]  = X[3:6,  0]   # euler (plant X[3:6], EKF x[6:9])
    ekf.x[9:12] = X[9:12, 0]

    X_ekf = np.zeros((12, N))

    try:
        from tqdm import tqdm as _tqdm
        _iter = _tqdm(range(N - 1), desc="EKF-UW", unit="step",
                      mininterval=5, dynamic_ncols=True)
    except ImportError:
        _iter = range(N - 1)

    for k in _iter:
        pos   = X[0:3,  k]
        euler = X[3:6,  k]
        vel   = X[6:9,  k]
        wb    = X[9:12, k]
        _a_inertial = (X[6:9, k+1] - X[6:9, k]) / dt
        for _j in range(_EKF_SUBSTEPS):
            _t_sub = t[k] + _j * _DT_EKF
            _meas  = suite.tick(_t_sub, _DT_EKF, _a_inertial, euler, wb, pos, vel)
            _ekf_x = ekf.update(_meas, _DT_EKF)
        X_ekf[:, k] = _ekf_x

    X_ekf[:, -1] = X_ekf[:, -2]

    _ps = max(1, N // 10_000)
    t_p = t[::_ps];  X_p = X[:, ::_ps]

    fig_e1, axes_e1 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e1.suptitle("EKF-UW: Position  —  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['x  [m]', 'y  [m]', 'z  [m]']):
        axes_e1[i].plot(t_p, X_p[i],               color='steelblue', lw=1.5, label='Truth')
        axes_e1[i].plot(t_p, X_ekf[i,   ::_ps],    color='tomato',    lw=1.2, ls='--', label='EKF')
        axes_e1[i].set_ylabel(lbl);  axes_e1[i].grid(True)
        if i == 0: axes_e1[i].legend(loc='upper right')
    axes_e1[-1].set_xlabel("Time  [s]");  fig_e1.tight_layout()

    fig_e2, axes_e2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e2.suptitle("EKF-UW: Velocity  —  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]']):
        axes_e2[i].plot(t_p, X_p[6+i],             color='steelblue', lw=1.5, label='Truth')
        axes_e2[i].plot(t_p, X_ekf[3+i, ::_ps],    color='tomato',    lw=1.2, ls='--', label='EKF')
        axes_e2[i].set_ylabel(lbl);  axes_e2[i].grid(True)
        if i == 0: axes_e2[i].legend(loc='upper right')
    axes_e2[-1].set_xlabel("Time  [s]");  fig_e2.tight_layout()

    fig_e3, axes_e3 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e3.suptitle("EKF-UW: Euler Angles  —  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['phi  [deg]', 'theta  [deg]', 'psi  [deg]']):
        _tr = np.arctan2(np.sin(X_p[3+i]), np.cos(X_p[3+i]))
        axes_e3[i].plot(t_p, np.degrees(_tr),               color='steelblue', lw=1.5, label='Truth')
        axes_e3[i].plot(t_p, np.degrees(X_ekf[6+i, ::_ps]), color='tomato',    lw=1.2, ls='--', label='EKF')
        axes_e3[i].set_ylabel(lbl);  axes_e3[i].grid(True)
        if i == 0: axes_e3[i].legend(loc='upper right')
    axes_e3[-1].set_xlabel("Time  [s]");  fig_e3.tight_layout()

    fig_e4, axes_e4 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e4.suptitle("EKF-UW: Body Rates  —  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['p  [rad/s]', 'q  [rad/s]', 'r  [rad/s]']):
        axes_e4[i].plot(t_p, X_p[9+i],             color='steelblue', lw=1.5, label='Truth')
        axes_e4[i].plot(t_p, X_ekf[9+i, ::_ps],    color='tomato',    lw=1.2, ls='--', label='EKF')
        axes_e4[i].set_ylabel(lbl);  axes_e4[i].grid(True)
        if i == 0: axes_e4[i].legend(loc='upper right')
    axes_e4[-1].set_xlabel("Time  [s]");  fig_e4.tight_layout()

    plt.show()
