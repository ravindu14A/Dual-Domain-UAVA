"""
sensors_UW.py  --  Discrete sensor models for the underwater phase (quadcopterUW)
DSE Team 30 | UAUV for Offshore Wind Turbine Inspection

Direct mirror of sensors_SC.py — same architecture, same noise model.
Only substitutions made:
  BaroSensor    → DepthSensor       (pressure depth,  same sigma/rate)
  GNSSSensor    → TransponderSensor (acoustic USBL/LBL, same sigma, 5 Hz)
  LiDAR2DSensor → SonarSensor       (horizontal sonar, same ray model)
  AHRSSensor / IMUSensor            unchanged

Convention: ENU inertial frame, body x=forward y=left z=up (FLU), ZYX Euler.
z = 0 at water surface, NEGATIVE downward (depth).
"""

import numpy as np


# ── rotation helper ───────────────────────────────────────────────────────────

def _rot_ZYX(phi, theta, psi):
    """Body-to-inertial (R).  R.T maps inertial → body."""
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)
    return np.array([
        [cy*ct,        cy*st*sp - sy*cp,   cy*st*cp + sy*sp],
        [sy*ct,        sy*st*sp + cy*cp,   sy*st*cp - cy*sp],
        [  -st,               ct*sp,              ct*cp     ],
    ])


# ── base class ────────────────────────────────────────────────────────────────

class _DiscreteSensor:
    update_rate: float = 1.0

    def __init__(self):
        self._last_tick = -1
        self._cache     = None

    def _step_index(self, t, dt):
        return int(round(t / dt))

    def _due(self, t, dt):
        sim_hz  = 1.0 / dt
        every_n = max(1, round(sim_hz / self.update_rate))
        step    = self._step_index(t, dt)
        return step % every_n == 0

    def tick(self, t, dt, *args, **kwargs):
        if self._due(t, dt):
            self._cache = self._measure(*args, **kwargs)
            return self._cache
        return None


# ── IMU — identical to aerial ─────────────────────────────────────────────────

class IMUSensor(_DiscreteSensor):
    update_rate = 200.0
    sigma_accel = 0.05
    sigma_gyro  = 0.002
    sigma_mag   = 0.01

    _G_INERTIAL = np.array([0.0, 0.0, -9.81])
    _MAG_REF    = np.array([1.0, 0.0,  0.0])

    def _measure(self, accel_inertial, euler, wb):
        phi, theta, psi = euler
        R_T = _rot_ZYX(phi, theta, psi).T
        specific_force = accel_inertial - self._G_INERTIAL
        accel_body = R_T @ specific_force + self.sigma_accel * np.random.randn(3)
        gyro_body  = wb                   + self.sigma_gyro  * np.random.randn(3)
        mag_body   = R_T @ self._MAG_REF  + self.sigma_mag   * np.random.randn(3)
        return accel_body, gyro_body, mag_body


# ── Depth sensor — same as BaroSensor ────────────────────────────────────────

class DepthSensor(_DiscreteSensor):
    """Pressure-based depth sensor.  Mirrors BaroSensor: same sigma, same rate."""
    update_rate  = 50.0
    sigma_depth  = 0.5    # [m]  matches BaroSensor.sigma_baro

    def _measure(self, true_z):
        return true_z + self.sigma_depth * np.random.randn()


# ── Transponder — same as GNSSSensor, 5 Hz ───────────────────────────────────

class TransponderSensor(_DiscreteSensor):
    """
    Acoustic USBL/LBL positioning.  Same output format as GNSSSensor;
    update rate lowered to 5 Hz (realistic for offshore acoustic systems).
    """
    update_rate = 5.0
    sigma_pos   = 0.5
    sigma_vel   = 0.1

    def _measure(self, true_pos, true_vel):
        trans_pos = true_pos + self.sigma_pos * np.random.randn(3)
        trans_vel = true_vel + self.sigma_vel * np.random.randn(3)
        return trans_pos, trans_vel


# ── AHRS — identical to aerial ────────────────────────────────────────────────

class AHRSSensor(_DiscreteSensor):
    update_rate  = 100.0
    sigma_phi    = np.radians(0.5)
    sigma_theta  = np.radians(0.5)
    sigma_psi    = np.radians(1.0)

    def _measure(self, euler):
        phi, theta, psi = euler
        return np.array([
            phi   + self.sigma_phi   * np.random.randn(),
            theta + self.sigma_theta * np.random.randn(),
            psi   + self.sigma_psi   * np.random.randn(),
        ])


# ── Sonar — same ray model as LiDAR2DSensor ───────────────────────────────────

class SonarSensor(_DiscreteSensor):
    """
    Horizontal 2-D scanning sonar.  Identical ray-cylinder model to LiDAR2DSensor;
    only sigma_range is slightly higher (multipath spreading).
    """
    update_rate = 10.0
    n_beams     = 360
    max_range   = 30.0
    sigma_range = 0.05    # [m]  slightly noisier than LiDAR (0.02 m)

    def _measure(self, drone_pos, psi, R_turbine):
        x_d, y_d = drone_pos[0], drone_pos[1]
        beam_angles_inertial = np.linspace(0, 2*np.pi, self.n_beams, endpoint=False)
        dx = np.cos(beam_angles_inertial)
        dy = np.sin(beam_angles_inertial)
        b    = 2.0 * (x_d * dx + y_d * dy)
        c    = x_d**2 + y_d**2 - R_turbine**2
        disc = b**2 - 4.0 * c
        ranges = np.full(self.n_beams, self.max_range)
        hit    = disc >= 0
        t_hit  = (-b[hit] - np.sqrt(disc[hit])) / 2.0
        valid  = t_hit > 0.0
        ranges[np.where(hit)[0][valid]] = np.minimum(t_hit[valid], self.max_range)
        ranges[hit] += self.sigma_range * np.random.randn(hit.sum())
        ranges = np.clip(ranges, 0.0, self.max_range)
        i_min    = np.argmin(ranges)
        standoff = ranges[i_min]
        bearing_inertial = np.arctan2(-y_d, -x_d)
        bearing_body     = np.arctan2(np.sin(bearing_inertial - psi),
                                      np.cos(bearing_inertial - psi))
        beam_angles_body = np.arctan2(np.sin(beam_angles_inertial - psi),
                                      np.cos(beam_angles_inertial - psi))
        return standoff, bearing_body, ranges, beam_angles_body


# ── Sensor suite ──────────────────────────────────────────────────────────────

class SensorSuite:
    def __init__(self, R_turbine=4.0):
        self.imu         = IMUSensor()
        self.depth       = DepthSensor()
        self.transponder = TransponderSensor()
        self.ahrs        = AHRSSensor()
        self.sonar       = SonarSensor()
        self.R_turbine   = R_turbine

    def tick(self, t, dt, accel_inertial, euler, wb, pos, vel):
        phi, theta, psi = euler
        return {
            'imu'        : self.imu.tick(t, dt, accel_inertial, euler, wb),
            'depth'      : self.depth.tick(t, dt, pos[2]),
            'transponder': self.transponder.tick(t, dt, pos, vel),
            'ahrs'       : self.ahrs.tick(t, dt, euler),
            'sonar'      : self.sonar.tick(t, dt, pos, psi, self.R_turbine),
        }
