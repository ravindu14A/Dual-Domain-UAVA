"""
kalman_SC.py  --  12-State Kinematic EKF for the aerial phase (quadcopterSC)
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

State vector (12):
  x = [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
       pos(3)     vel(3)        euler(3)            body_rates(3)

Prediction  : strapdown at 200 Hz — accel propagates velocity, gyro propagates attitude
Corrections : gyro (200 Hz), AHRS euler angles (100 Hz), GNSS (5 Hz), baro (50 Hz).

Convention: ENU inertial frame, body x=forward y=left z=up (FLU), ZYX Euler.
"""

import numpy as np
from sensors_SC import IMUSensor, BaroSensor, GNSSSensor, AHRSSensor


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def _rot(phi, theta, psi):
    """Body-to-inertial rotation matrix (ZYX Euler, ENU)."""
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
    12-state kinematic EKF.

    Usage:
        ekf  = KinematicEKF12()
        meas = suite.tick(t, dt, ...)   # SensorSuite from sensors_SC.py
        est  = ekf.update(meas)         # returns 12-state estimate [pos,vel,euler,rates]
    """

    _G = 9.81

    # ── Process noise (Q) ─────────────────────────────────────────────────────
    # Q represents prediction model uncertainty, derived from sensor sigmas where
    # the sensor drives the prediction, plus small tuning values for the rest.
    #   vel   is predicted from accel  → use IMUSensor.sigma_accel
    #   euler is predicted from gyro   → use IMUSensor.sigma_gyro
    #   pos   integrates vel (good model) → small constant
    #   rates are a random walk        → tuning choice
    _accel_k = 1.0   # adaptive accel R tuning: R_eff = R0*(1 + k*e²), e = |a|-g

    _THETA_LIM = np.radians(80)

    def __init__(self):
        self.x = np.zeros(12)
        self.P = np.eye(12) * 10.0
        self._build_R()
        self.x_prior = np.zeros(12)   # state after predict+gyro, before slow corrections

    # ── R matrices ────────────────────────────────────────────────────────────

    def _build_R(self):
        self.R_gyro = np.diag(np.full(3, IMUSensor.sigma_gyro**2))
        self.R_ahrs = np.diag([
            AHRSSensor.sigma_phi**2,
            AHRSSensor.sigma_theta**2,
            AHRSSensor.sigma_psi**2,
        ])
        self.R_gnss = np.diag([
            *([GNSSSensor.sigma_pos**2] * 3),
            *([GNSSSensor.sigma_vel**2] * 3),
        ])
        self.R_baro = BaroSensor.sigma_baro**2

    # ── Process noise matrix ──────────────────────────────────────────────────

    def _Q(self, dt):
        q = np.concatenate([
            np.full(3, 1e-4),                  # pos:   near-perfect integration model
            np.full(3, IMUSensor.sigma_accel),  # vel:   driven by accel measurement noise
            np.full(3, IMUSensor.sigma_gyro),   # euler: driven by gyro measurement noise
            np.full(3, IMUSensor.sigma_gyro),   # rates: also from gyro (corrected every step)
        ])
        return np.diag(q**2) * dt

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, accel, gyro, dt=None):
        """Propagate state and covariance driven by body-frame IMU accel and gyro."""
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

        # ── Jacobian F (12x12) ────────────────────────────────────────────────
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
        F[0:3, 3:6]   = np.eye(3) * dt
        F[3:6, 6:9]   = np.array([
            [dvx_dphi, dvx_dth, dvx_dpsi],
            [dvy_dphi, dvy_dth, dvy_dpsi],
            [dvz_dphi, dvz_dth, 0.0     ],
        ]) * dt
        F[6,  9:12]   = np.array([1.0, sp*tan_t, cp*tan_t]) * dt
        F[7,  9:12]   = np.array([0.0, cp,       -sp      ]) * dt
        F[8,  9:12]   = np.array([0.0, sp/ct,    cp/ct    ]) * dt

        self.x = x
        self.P = F @ self.P @ F.T + self._Q(dt)

    # ── Generic EKF update ────────────────────────────────────────────────────

    def _update(self, H, R_mat, innov):
        S = H @ self.P @ H.T + R_mat
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(12) - K @ H) @ self.P
        self.x[6] = _wrap(self.x[6])
        self.x[7] = np.clip(_wrap(self.x[7]), -self._THETA_LIM, self._THETA_LIM)
        self.x[8] = _wrap(self.x[8])

    # ── Individual corrections ────────────────────────────────────────────────

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

    def correct_accel_tilt(self, accel):
        """
        Accelerometer attitude correction with adaptive R.
        R is inflated by (1 + k*e²) where e = |accel| - g, so corrections
        are automatically down-weighted during manoeuvres without a hard gate.
        """
        e = np.linalg.norm(accel) - self._G
        R_eff = self.R_accel * (1.0 + self._accel_k * e**2)

        phi_k, theta_k = self.x[6], self.x[7]
        cp, sp = np.cos(phi_k), np.sin(phi_k)
        ct, st = np.cos(theta_k), np.sin(theta_k)

        h = np.array([
            -self._G * st,
             self._G * ct * sp,
             self._G * ct * cp,
        ])

        H = np.zeros((3, 12))
        H[0, 7]  = -self._G * ct
        H[1, 6]  =  self._G * ct * cp;  H[1, 7] = -self._G * st * sp
        H[2, 6]  = -self._G * ct * sp;  H[2, 7] = -self._G * st * cp

        self._update(H, R_eff, accel - h)
        self.x[6] = _wrap(self.x[6])
        self.x[7] = _wrap(self.x[7])

    def correct_gnss(self, gnss_pos, gnss_vel):
        H = np.zeros((6, 12));  H[0:6, 0:6] = np.eye(6)
        z = np.concatenate([gnss_pos, gnss_vel])
        self._update(H, self.R_gnss, z - H @ self.x)

    def correct_baro(self, baro_z):
        H = np.zeros((1, 12));  H[0, 2] = 1.0
        R = np.array([[self.R_baro]])
        self._update(H, R, np.array([baro_z - self.x[2]]))

    def correct_mag(self, mag):
        """
        Tilt-compensated magnetometer yaw correction.
        Reference field [1, 0, 0] in ENU (pointing East).
        """
        phi_k, theta_k = self.x[6], self.x[7]
        cp, sp = np.cos(phi_k), np.sin(phi_k)
        ct, st = np.cos(theta_k), np.sin(theta_k)

        h_north  = mag[0]*ct + mag[1]*st*sp + mag[2]*st*cp
        h_east   = mag[1]*cp - mag[2]*sp
        meas_yaw = np.arctan2(-h_east, h_north)

        H = np.zeros((1, 12));  H[0, 8] = 1.0
        R = np.array([[self.R_mag]])
        innov = _wrap(meas_yaw - self.x[8])
        self._update(H, R, np.array([innov]))
        self.x[8] = _wrap(self.x[8])

    # ── Main entry point ──────────────────────────────────────────────────────

    def update(self, sensor_data, dt=None):
        """
        Call once per IMU step with dict from SensorSuite.tick().
        Returns 12-state estimate [pos, vel, euler, rates].
        """
        imu  = sensor_data.get('imu')
        baro = sensor_data.get('baro')
        gnss = sensor_data.get('gnss')

        ahrs = sensor_data.get('ahrs')

        if imu is not None:
            accel, gyro, mag = imu
            self.predict(accel, gyro, dt)
            self.correct_gyro(gyro)
            self.x_prior = self.x.copy()   # prior: after IMU, before slow corrections

        if ahrs is not None:
            self.correct_ahrs(ahrs)

        if gnss is not None:
            self.correct_gnss(*gnss)

        if baro is not None:
            self.correct_baro(baro)

        return self.x[:12].copy()

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def pos(self):    return self.x[0:3].copy()
    @property
    def vel(self):    return self.x[3:6].copy()
    @property
    def euler(self):  return self.x[6:9].copy()
    @property
    def rates(self):  return self.x[9:12].copy()


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sensors_SC import SensorSuite, IMUSensor

    _npz = Path(__file__).parent / 'sim_data_SC.npz'
    if not _npz.exists():
        raise FileNotFoundError(
            f"{_npz} not found — run quadcopterSC.py with SAVE_SIM_DATA=True first.")

    data   = np.load(str(_npz))
    t      = data['t']
    X      = data['X']
    dt     = float(data['dt'])
    R_base = float(data['R_base'])
    N      = len(t)
    print(f"[EKF] Loaded {N} steps  dt={dt}s  T={t[-1]:.1f}s  R_base={R_base}m")

    _EKF_SUBSTEPS = max(1, round(dt * IMUSensor.update_rate))
    _DT_EKF       = dt / _EKF_SUBSTEPS
    if _EKF_SUBSTEPS > 1:
        print(f"[EKF] {_EKF_SUBSTEPS} sub-steps/plant step  dt_ekf={_DT_EKF:.4f}s")

    suite = SensorSuite(R_turbine=R_base)
    ekf   = KinematicEKF12()
    ekf.x[0:3]  = X[0:3,  0]
    ekf.x[3:6]  = X[6:9,  0]
    ekf.x[6:9]  = X[3:6,  0]
    ekf.x[9:12] = X[9:12, 0]

    X_ekf  = np.zeros((12, N))

    # sparse measurement logs — NaN where sensor didn't fire this step
    meas_gnss_pos = np.full((3, N), np.nan)
    meas_gnss_vel = np.full((3, N), np.nan)
    meas_baro_z   = np.full(N,      np.nan)
    meas_ahrs     = np.full((3, N), np.nan)

    try:
        from tqdm import tqdm as _tqdm
        _ekf_iter = _tqdm(range(N - 1), desc="EKF", unit="step",
                          mininterval=5, dynamic_ncols=True)
    except ImportError:
        _ekf_iter = range(N - 1)

    for k in _ekf_iter:
        pos   = X[0:3,  k]
        euler = X[3:6,  k]
        vel   = X[6:9,  k]
        wb    = X[9:12, k]
        _a_inertial = (X[6:9, k+1] - X[6:9, k]) / dt
        for _j in range(_EKF_SUBSTEPS):
            _t_sub = t[k] + _j * _DT_EKF
            _meas  = suite.tick(_t_sub, _DT_EKF, _a_inertial, euler, wb, pos, vel)
            _ekf_x = ekf.update(_meas, _DT_EKF)
            # log measurements from whichever sub-step they fire on
            if _meas.get('gnss') is not None:
                meas_gnss_pos[:, k] = _meas['gnss'][0]
                meas_gnss_vel[:, k] = _meas['gnss'][1]
            if _meas.get('baro') is not None:
                meas_baro_z[k] = _meas['baro']
            if _meas.get('ahrs') is not None:
                meas_ahrs[:, k] = _meas['ahrs']
        X_ekf[:, k] = _ekf_x

    X_ekf[:, -1] = X_ekf[:, -2]

    _ps = max(1, N // 10_000)
    t_p = t[::_ps]
    X_p = X[:, ::_ps]

    # Figure EKF-1: Position
    fig_e1, axes_e1 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e1.suptitle("EKF: Position  --  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['x  [m]', 'y  [m]', 'z  [m]']):
        ax = axes_e1[i]
        _gm = np.isfinite(meas_gnss_pos[i])
        ax.scatter(t[_gm], meas_gnss_pos[i, _gm], s=10, color='limegreen', zorder=1, alpha=0.6,
                   label='GNSS meas' if i == 0 else '_')
        if i == 2:
            _bm = np.isfinite(meas_baro_z)
            ax.scatter(t[_bm][::5], meas_baro_z[_bm][::5], s=8, color='orange', zorder=1, alpha=0.6,
                       label='Baro meas')
        ax.plot(t_p, X_ekf[i, ::_ps],  color='tomato',    lw=1.2, ls='--', zorder=4, label='EKF')
        ax.plot(t_p, X_p[i],           color='steelblue', lw=1.5,          zorder=3, label='Truth')
        ax.set_ylabel(lbl); ax.grid(True)
        if i == 0 or i == 2: ax.legend(loc='upper right', fontsize=8)
    axes_e1[-1].set_xlabel("Time  [s]")
    for ax in fig_e1.axes: ax.tick_params(labelbottom=True)
    fig_e1.tight_layout()

    # Figure EKF-2: Velocity
    fig_e2, axes_e2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e2.suptitle("EKF: Velocity  --  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['vx  [m/s]', 'vy  [m/s]', 'vz  [m/s]']):
        ax = axes_e2[i]
        _gm = np.isfinite(meas_gnss_vel[i])
        ax.scatter(t[_gm], meas_gnss_vel[i, _gm], s=10, color='limegreen', zorder=1, alpha=0.6,
                   label='GNSS vel meas' if i == 0 else '_')
        ax.plot(t_p, X_ekf[3+i, ::_ps], color='tomato',    lw=1.2, ls='--', zorder=4, label='EKF')
        ax.plot(t_p, X_p[6+i],          color='steelblue', lw=1.5,          zorder=3, label='Truth')
        ax.set_ylabel(lbl); ax.grid(True)
        if i == 0: ax.legend(loc='upper right', fontsize=8)
    axes_e2[-1].set_xlabel("Time  [s]")
    for ax in fig_e2.axes: ax.tick_params(labelbottom=True)
    fig_e2.tight_layout()

    # Figure EKF-3: Euler angles
    fig_e3, axes_e3 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e3.suptitle("EKF: Euler Angles  --  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['phi  [deg]', 'theta  [deg]', 'psi  [deg]']):
        ax = axes_e3[i]
        _am = np.isfinite(meas_ahrs[i])
        ax.scatter(t[_am], np.degrees(meas_ahrs[i, _am]), s=8, color='mediumpurple', zorder=1, alpha=0.6,
                   label='AHRS meas' if i == 0 else '_')
        _truth_ang = np.arctan2(np.sin(X_p[3+i]), np.cos(X_p[3+i]))
        ax.plot(t_p, np.degrees(X_ekf[6+i, ::_ps]), color='tomato',    lw=1.2, ls='--', zorder=4, label='EKF')
        ax.plot(t_p, np.degrees(_truth_ang),         color='steelblue', lw=1.5,          zorder=3, label='Truth')
        ax.set_ylabel(lbl); ax.grid(True)
        if i == 0: ax.legend(loc='upper right', fontsize=8)
    axes_e3[-1].set_xlabel("Time  [s]")
    for ax in fig_e3.axes: ax.tick_params(labelbottom=True)
    fig_e3.tight_layout()

    # Figure EKF-4: Body rates  (no slow-sensor dots — gyro corrects every step)
    fig_e4, axes_e4 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig_e4.suptitle("EKF: Body Rates  --  Truth vs Estimate", fontsize=13)
    for i, lbl in enumerate(['p  [rad/s]', 'q  [rad/s]', 'r  [rad/s]']):
        ax = axes_e4[i]
        ax.plot(t_p, X_ekf[9+i, ::_ps], color='tomato',    lw=1.2, ls='--', zorder=4, label='EKF')
        ax.plot(t_p, X_p[9+i],          color='steelblue', lw=1.5,          zorder=3, label='Truth')
        ax.set_ylabel(lbl); ax.grid(True)
        if i == 0: ax.legend(loc='upper right', fontsize=8)
    axes_e4[-1].set_xlabel("Time  [s]")
    for ax in fig_e4.axes: ax.tick_params(labelbottom=True)
    fig_e4.tight_layout()

    plt.show()
